import { otelFetch } from "./otelFetch.js";

function inlineSchema(schema, topLevelSchema, seen = new Set()) {
  if (typeof schema !== 'object' || schema === null) {
    return schema;
  }

  if (Array.isArray(schema)) {
    return schema.map(item => inlineSchema(item, topLevelSchema, seen));
  }

  if (schema.$ref) {
    const refPath = schema.$ref;
    if (seen.has(refPath)) {
      return {};
    }
    seen.add(refPath);

    if (refPath.startsWith('#/')) {
      const pathParts = refPath.slice(2).split('/');
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
      return inlineSchema(definition, topLevelSchema, seen);
    } else {
      const defName = refPath.split('/').pop();
      const definition = topLevelSchema.$defs?.[defName];
      if (!definition) {
        throw new Error(`Schema definition not found for ref: ${refPath}`);
      }
      return inlineSchema(definition, topLevelSchema, seen);
    }
  }

  const newSchema = {};
  for (const key in schema) {
    newSchema[key] = inlineSchema(schema[key], topLevelSchema, seen);
  }

  return newSchema;
}

function mapTools(tools) {
  return tools.map(tool => {
    const inputSchemaCopy = JSON.parse(JSON.stringify(tool.inputSchema));
    const processedSchema = inlineSchema(inputSchemaCopy, inputSchemaCopy);
    delete processedSchema.$defs;

    return {
      type: "function",
      function: {
        name: tool.name,
        description: tool.description,
        parameters: processedSchema
      }
    };
  });
}

export class TGI {
  constructor(tools = [], modelName = "Qwen/Qwen3-Coder-480B-A35B-Instruct:fireworks-ai", tgiUrl = '/api/mcp/atlassian/tgi/v1/chat/completions') {
    this.url = tgiUrl;
    this.modelName = modelName;
    this.messages = [];
    this.tools = mapTools(tools);
  }

  clearHistory() {
    this.messages = [];
  }

  async send(promptOrMessages, options = undefined, onStream = () => {}) {
    let toolChoice = undefined;
    let useWorkflow = undefined;
    let workflowExecutionId = undefined;

    if (typeof options === 'function') {
      onStream = options;
    } else if (typeof options === 'string') {
      toolChoice = options;
    } else if (options && typeof options === 'object') {
      toolChoice = options.toolChoice;
      useWorkflow = options.useWorkflow;
      workflowExecutionId = options.workflowExecutionId;
    }

    if (Array.isArray(promptOrMessages)) {
      this.messages = promptOrMessages;
    } else {
      this.messages.push({ role: 'user', content: promptOrMessages });
    }

    let messages = this.messages.length > 5 ? [this.messages[0], ...this.messages.slice(-4)] : this.messages;

    const payload = {
      messages,
      model: this.modelName,
      stream: true,
    };

    if (this.tools && this.tools.length > 0) {
      payload.tools = this.tools;
    }
    if (toolChoice !== undefined) {
      payload.tool_choice = toolChoice;
    }
    if (useWorkflow !== undefined) {
      payload.use_workflow = useWorkflow;
    }
    if (workflowExecutionId !== undefined) {
      payload.workflow_execution_id = workflowExecutionId;
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
              onStream(`[${chunk.error.http_status_code ?? 500}] ${chunk.error.message}`, {});
              fullContent += `[${chunk.error.http_status_code ?? 500}] ${chunk.error.message}`;
              break;
            }
            const delta = chunk.choices[0].delta;
            const metadata = chunk.metadata || {};
            if (delta.content) {
              fullContent += delta.content;
              onStream(delta.content, metadata);
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
