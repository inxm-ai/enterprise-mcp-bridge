function inlineSchema(schema, topLevelSchema, seen = new Set()) {
  // If the schema is not an object or is null, return it as is.
  if (typeof schema !== 'object' || schema === null) {
    return schema;
  }

  // If the object is an array, process each item in the array.
  if (Array.isArray(schema)) {
    return schema.map(item => inlineSchema(item, topLevelSchema, seen));
  }

  // If the object is a reference, find its definition and process it.
  if (schema.$ref) {
    const refPath = schema.$ref;
    if (seen.has(refPath)) {
      // Prevent infinite recursion for cyclic refs
      return {}; // or return schema as-is, or null, depending on your needs
    }
    seen.add(refPath);

    if (refPath.startsWith('#/')) {
      // Remove the leading "#/" and split by "/"
      const pathParts = refPath.slice(2).split('/');
      // Traverse the topLevelSchema following the path
      let definition = topLevelSchema;
      for (const part of pathParts) {
        if (definition && typeof definition === 'object') {
          definition = definition[part];
        } else {
          throw new Error(`Schema definition not found for ref: ${refPath}`);
        }
      }
      if (!definition) {
        throw new Error(`Schema definition not found for ref: ${refPath}`);
      }
      // Recursively inline any refs that might be inside the definition itself.
      return inlineSchema(definition, topLevelSchema, seen);
    } else {
      const defName = refPath.split('/').pop();
      const definition = topLevelSchema.$defs?.[defName];
      if (!definition) {
        throw new Error(`Schema definition not found for ref: ${refPath}`);
      }
      // Recursively inline any refs that might be inside the definition itself.
      return inlineSchema(definition, topLevelSchema, seen);
    }
  }

  // For regular objects, create a new object and process all its values.
  const newSchema = {};
  for (const key in schema) {
    newSchema[key] = inlineSchema(schema[key], topLevelSchema, seen);
  }

  return newSchema;
}

function mapTools(tools) {
  return tools.map(tool => {
    // Make a deep copy to avoid modifying the original tool object
    const inputSchemaCopy = JSON.parse(JSON.stringify(tool.inputSchema));
    
    // Process the schema to inline all references
    const processedSchema = inlineSchema(inputSchemaCopy, inputSchemaCopy);
    
    // The top-level $defs is no longer needed after inlining
    delete processedSchema.$defs;

    return {
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        // The parameters are the entire processed input schema
        parameters: processedSchema
      }
    };
  });
}
const onExecuteTool = (mcpServer) => async (toolName, toolArgs) => {
    const res = await fetch(`${mcpServer}/${toolName}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify(toolArgs)
    });
    if (!res.ok) {
        const errorText = await res.text();
        throw new Error(`Tool execution failed: ${res.status} ${errorText}`);
    }
    return res.json();
}

export class TGI {
    
    constructor(toolDetails, mcpServer, tgiUrl = '/api/tgi/model/v1/chat/completions') {
        this.tools = mapTools(toolDetails);
        this.url = tgiUrl;
        this.onExecuteTool = onExecuteTool(mcpServer);
        this.messages = []; // Stores the conversation history for a session
    }

    /**
     * Clears the conversation history.
     */
    clearHistory() {
        this.messages = [];
    }

    /**
     * Sends a prompt to the model and manages the entire conversation flow, including tool calls.
     * @param {string} prompt - The user's prompt to send to the model.
     * @param {string|'auto'} [toolChoice='auto'] - The tool selection strategy. Can be 'auto' or the name of a specific function.
     * @param {function(string): void} [onStream] - A callback function that receives chunks of the streaming text response.
     * @returns {Promise<string>} A promise that resolves with the final text response from the assistant.
     */
    async send(prompt, toolChoice = 'auto', onStream = () => {}, newMessage = () => {}) {
        // Add the user's new prompt to the history
        this.messages.push({ role: 'user', content: prompt });

        // The main loop to handle the conversation, including multiple tool calls if needed.
        while (true) {
            toolChoice = toolChoice !== 'auto'
                ? { type: "function", function: { name: toolChoice } }
                : "auto";

            // Get the assistant's response (which could be text or a tool call)
            const assistantMessage = await this._chat(this.messages, {
                toolChoice,
                onStream,
            });

            // Add the assistant's response to the history
            if (assistantMessage) {
                 this.messages.push(assistantMessage);
            }

            // If the response was a tool call, handle it
            if (assistantMessage && assistantMessage.tool_calls) {
                if (!this.onExecuteTool) {
                    throw new Error("Model requested a tool call, but no 'onExecuteTool' callback was provided in the TGI constructor.");
                }

                const toolResults = await Promise.all(
                    assistantMessage.tool_calls.map(async (toolCall) => {
                        const toolName = toolCall.function.name;
                        const toolArgs = JSON.parse(toolCall.function.arguments);

                        let messages = (this.messages.length > 5) ? [this.messages[0], ...this.messages.slice(-4)] : this.messages;
                        const userMessage = `${messages.map(m => m.content ?? m.tool_calls.map(tc => `${tc.function.name}(${Object.entries(JSON.parse(tc.function.arguments)).map(([k, v]) => `${k}:${v}`).join(', ')})`)).join('\n\n')}`;
                        console.log(`getting more information for tool call: ${toolName} with args:`, toolArgs, "from user message:", userMessage);
                        const reason = await this._chat([
                            { role: 'system', content: `You're an expert explaining why a tool will be called! Take the user information on the current history and come back with a super short 1 sentences 10 words max explainer on why the last tool chosen was based on previous data. If there are no results yet then this is the first tool call. Explain why to start with that.` },
                            { role: 'user', content: userMessage }
                        ]);
                        newMessage({
                            role: "thinking",
                            content: reason.content,
                        });
                        
                        // Execute the tool using the provided callback
                        const result = await this.onExecuteTool(toolName, toolArgs);

                        const jsonResult = typeof result === 'string' ? JSON.parse(result) : result;

                        return {
                            role: 'tool',
                            tool_call_id: toolCall.id,
                            name: toolName,
                            content: JSON.stringify(jsonResult.structuredContent || jsonResult.content || result),
                        };
                    })
                );
                
                // Add all tool results to the message history and continue the loop
                this.messages.push(...toolResults);
                // Continue to the next iteration to get the final text response
                
            } else {
                // If there were no tool calls, the conversation is done for this turn.
                // Return the final text content.
                return assistantMessage ? assistantMessage.content : "";
            }
        }
    }

    /**
     * Private method to make a single fetch request and process the streaming response.
     * @private
     */
    async _chat(messages, options) {
        const { toolChoice, onStream = () => {} } = options || { toolChoice: 'none', onStream: () => {} };

        // limit the messages to the last 5 for performance, but keep the first user message
        if (messages.length > 5) {
            messages = [messages[0], ...messages.slice(-4)];
        }

        const payload = {
            messages,
            tools: this.tools,
            tool_choice: toolChoice,
            stream: true,
        };

        const response = await fetch(this.url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'text/event-stream' },
            credentials: 'include',
            body: JSON.stringify(payload),
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, message: ${errorText}`);
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        
        let fullContent = '';
        let toolCalls = [];

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep the last, possibly incomplete, line

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const data = line.substring(6);
                    if (data.trim() === '[DONE]') break;
                    
                    try {
                        const chunk = JSON.parse(data);
                        if (chunk.error) {
                            onStream(`[${chunk.error.http_status_code ?? 500}] ${chunk.error.message}`);
                            fullContent += `[${chunk.error.http_status_code ?? 500}] ${chunk.error.message}`;
                            break;
                        }

                        const delta = chunk.choices[0].delta;

                        if (delta.content) {
                            fullContent += delta.content;
                            //onStream(delta.content);
                        }

                        if (delta.tool_calls) {
                            delta.tool_calls.forEach(tcDelta => {
                                if (toolCalls.length <= tcDelta.index) {
                                    toolCalls.push({ id: '', type: 'function', function: { name: '', arguments: '' } });
                                }
                                const currentTool = toolCalls[tcDelta.index];
                                if (tcDelta.id) currentTool.id = tcDelta.id;
                                if (tcDelta.function.name) currentTool.function.name = tcDelta.function.name;
                                if (tcDelta.function.arguments) currentTool.function.arguments += tcDelta.function.arguments;
                            });
                        }
                    } catch (error) {
                        console.error('Error parsing stream data chunk:', error, 'Chunk:', data);
                    }
                }
            }
        }
        
        const finalMessage = { role: 'assistant', content: fullContent || null };
        if (toolCalls.length > 0) {
            finalMessage.tool_calls = toolCalls;
        }
        
        return finalMessage.content === null && toolCalls.length === 0 ? null : finalMessage;
    }
}