import { otelFetch } from "./otelFetch.js";

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
export class TGI {
  /**
   * @param {Array<Object>} [tools=[]] - Array of OpenAI tool objects
   * @param {string} [tgiUrl='/api/tgi/chat/completions'] - Backend endpoint
   * @param {string} [modelName] - Model name
   */
  constructor(tools = [], modelName = "Qwen/Qwen3-Coder-480B-A35B-Instruct:fireworks-ai", tgiUrl = '/api/mcp/m365/tgi/v1/chat/completions') {
    this.url = tgiUrl;
    this.modelName = modelName;
    this.messages = [];
    this.tools = mapTools(tools);
  }

  clearHistory() {
    this.messages = [];
  }

  /**
   * Sends a prompt to the backend and returns the assistant's response.
   * @param {string} prompt - The user's prompt to send to the model.
   * @param {function(string): void} [onStream] - A callback function that receives chunks of the streaming text response.
   * @param {Array<Object>} [toolsOverride] - Optional array of tools to use for this request only
   * @param {Object} [toolChoice] - Optional tool selection strategy (OpenAI format)
   * @returns {Promise<string>} A promise that resolves with the final text response from the assistant.
   */
  async send(prompt, toolChoice = undefined, onStream = () => {}) {
    this.messages.push({ role: 'user', content: prompt });

    // Only keep the first and last 4 messages for performance
    let messages = this.messages.length > 5 ? [this.messages[0], ...this.messages.slice(-4)] : this.messages;

    const payload = {
      messages,
      model: this.modelName,
      stream: true,
    };
    // Use override tools if provided, else default
    if (this.tools && this.tools.length > 0) {
      payload.tools = this.tools;
    }
    // Optionally specify tool_choice
    if (toolChoice !== undefined) {
      payload.tool_choice = toolChoice;
    }

    const response = await otelFetch(this.url, {
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

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();

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
              onStream(delta.content);
            }
          } catch (error) {
            console.error('Error parsing stream data chunk:', error, 'Chunk:', data);
          }
        }
      }
    }

    this.messages.push({ role: 'assistant', content: fullContent });
    return fullContent;
  }
}