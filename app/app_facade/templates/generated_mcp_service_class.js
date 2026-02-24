const __mcpExtractJsonBlock = (text) => {
  if (typeof text !== 'string') return null;
  let start = -1;
  let depth = 0;
  let inString = false;
  let escape = false;
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (start < 0) {
      if (char === '{') {
        start = index;
        depth = 1;
      }
      continue;
    }
    if (escape) {
      escape = false;
      continue;
    }
    if (char === '\\') {
      escape = true;
      continue;
    }
    if (char === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (char === '{') depth += 1;
    if (char === '}') depth -= 1;
    if (depth === 0) return text.slice(start, index + 1);
  }
  return null;
};

const __mcpParseJson = (text) => {
  if (typeof text !== 'string') return null;
  const trimmed = text.trim();
  if (!trimmed) return null;
  try {
    return JSON.parse(trimmed);
  } catch (_err) {
  }
  const block = __mcpExtractJsonBlock(trimmed);
  if (!block) return null;
  try {
    return JSON.parse(block);
  } catch (_err) {
    return null;
  }
};

const __mcpStringify = (value) => {
  try {
    return JSON.stringify(value);
  } catch (_err) {
    return String(value);
  }
};

const __MCP_RUNTIME_EXCHANGE_LIMIT = 120;
const __MCP_RUNTIME_FIELD_LIMIT = 2000;
const __MCP_EXTRACTION_TEXT_CAP = 4000;

const __mcpTrimText = (value) => {
  if (value == null) return '';
  const text = String(value);
  if (text.length <= __MCP_RUNTIME_FIELD_LIMIT) return text;
  return `${text.slice(0, __MCP_RUNTIME_FIELD_LIMIT)}...[truncated]`;
};

const __mcpCapExtractionText = (value) => {
  const text = typeof value === 'string' ? value : __mcpStringify(value);
  if (text.length <= __MCP_EXTRACTION_TEXT_CAP) {
    return { text, truncated: false };
  }
  return {
    text: text.slice(0, __MCP_EXTRACTION_TEXT_CAP),
    truncated: true,
  };
};

const __mcpSanitizeRuntimeValue = (value, depth = 0) => {
  if (value == null) return value;
  if (typeof value === 'string') return __mcpTrimText(value);
  if (typeof value === 'number' || typeof value === 'boolean') return value;
  if (depth >= 3) return __mcpTrimText(__mcpStringify(value));
  if (Array.isArray(value)) {
    return value.slice(0, 30).map((item) => __mcpSanitizeRuntimeValue(item, depth + 1));
  }
  if (typeof value === 'object') {
    const out = {};
    for (const [key, entry] of Object.entries(value).slice(0, 40)) {
      out[String(key)] = __mcpSanitizeRuntimeValue(entry, depth + 1);
    }
    return out;
  }
  return __mcpTrimText(value);
};

const __mcpCreateRuntimeStore = () => ({
  entries: [],
  nextCursor: 1,
});

const __mcpRuntimeStore = (() => {
  if (typeof globalThis === 'undefined') return __mcpCreateRuntimeStore();
  const existing = globalThis.__generatedUiServiceRuntime;
  if (existing && Array.isArray(existing.entries) && typeof existing.nextCursor === 'number') {
    return existing;
  }
  const fresh = __mcpCreateRuntimeStore();
  globalThis.__generatedUiServiceRuntime = fresh;
  return fresh;
})();

const __mcpAppendRuntimeExchange = (entry) => {
  const normalized = {
    cursor: __mcpRuntimeStore.nextCursor++,
    timestamp: Date.now(),
    tool: __mcpTrimText(entry?.tool || ''),
    request_body: __mcpSanitizeRuntimeValue(entry?.request_body ?? {}),
    request_options: __mcpSanitizeRuntimeValue(entry?.request_options ?? {}),
    response_payload: __mcpSanitizeRuntimeValue(entry?.response_payload ?? null),
    error: __mcpTrimText(entry?.error || ''),
    mocked: Boolean(entry?.mocked),
  };
  __mcpRuntimeStore.entries.push(normalized);
  if (__mcpRuntimeStore.entries.length > __MCP_RUNTIME_EXCHANGE_LIMIT) {
    __mcpRuntimeStore.entries = __mcpRuntimeStore.entries.slice(-__MCP_RUNTIME_EXCHANGE_LIMIT);
  }
  return normalized;
};

const __mcpCollectRuntimeExchanges = (sinceCursor = 0, limit = 20) => {
  const safeSince = Number.isFinite(Number(sinceCursor)) ? Number(sinceCursor) : 0;
  const safeLimit = Math.max(1, Math.min(100, Number(limit) || 20));
  const entries = __mcpRuntimeStore.entries
    .filter((entry) => Number(entry.cursor) > safeSince)
    .slice(-safeLimit);
  const cursor = entries.length > 0
    ? Number(entries[entries.length - 1].cursor)
    : Math.max(safeSince, Number(__mcpRuntimeStore.nextCursor || 1) - 1);
  return { cursor, entries };
};

const __mcpClearRuntimeExchanges = () => {
  __mcpRuntimeStore.entries = [];
  __mcpRuntimeStore.nextCursor = 1;
  return { cursor: 0, cleared: true };
};

const __mcpIsObject = (value) =>
  value !== null && typeof value === 'object' && !Array.isArray(value);

const __mcpDeepEqual = (left, right) => {
  if (left === right) return true;
  if (typeof left !== typeof right) return false;
  if (Array.isArray(left) && Array.isArray(right)) {
    if (left.length !== right.length) return false;
    for (let index = 0; index < left.length; index += 1) {
      if (!__mcpDeepEqual(left[index], right[index])) return false;
    }
    return true;
  }
  if (__mcpIsObject(left) && __mcpIsObject(right)) {
    const leftKeys = Object.keys(left);
    const rightKeys = Object.keys(right);
    if (leftKeys.length !== rightKeys.length) return false;
    for (const key of leftKeys) {
      if (!Object.prototype.hasOwnProperty.call(right, key)) return false;
      if (!__mcpDeepEqual(left[key], right[key])) return false;
    }
    return true;
  }
  return false;
};

const __mcpTypeMatches = (value, type) => {
  switch (type) {
    case 'object':
      return __mcpIsObject(value);
    case 'array':
      return Array.isArray(value);
    case 'string':
      return typeof value === 'string';
    case 'number':
      return typeof value === 'number' && Number.isFinite(value);
    case 'integer':
      return typeof value === 'number' && Number.isInteger(value);
    case 'boolean':
      return typeof value === 'boolean';
    case 'null':
      return value === null;
    default:
      return true;
  }
};

const __mcpMatchesSchema = (value, schema) => {
  if (!__mcpIsObject(schema)) return false;

  if (Object.prototype.hasOwnProperty.call(schema, 'const')) {
    if (!__mcpDeepEqual(value, schema.const)) return false;
  }
  if (Array.isArray(schema.enum)) {
    if (!schema.enum.some((candidate) => __mcpDeepEqual(candidate, value))) return false;
  }

  if (Array.isArray(schema.allOf)) {
    if (!schema.allOf.every((item) => __mcpMatchesSchema(value, item))) return false;
  }
  if (Array.isArray(schema.anyOf)) {
    if (!schema.anyOf.some((item) => __mcpMatchesSchema(value, item))) return false;
  }
  if (Array.isArray(schema.oneOf)) {
    const matching = schema.oneOf.filter((item) => __mcpMatchesSchema(value, item));
    if (matching.length !== 1) return false;
  }

  const schemaTypes = Array.isArray(schema.type)
    ? schema.type
    : typeof schema.type === 'string'
      ? [schema.type]
      : [];
  if (schemaTypes.length > 0 && !schemaTypes.some((item) => __mcpTypeMatches(value, item))) {
    return false;
  }

  const hasObjectRules =
    Array.isArray(schema.required) ||
    __mcpIsObject(schema.properties) ||
    schema.additionalProperties === false;
  if (hasObjectRules) {
    if (!__mcpIsObject(value)) return false;

    const required = Array.isArray(schema.required) ? schema.required : [];
    for (const key of required) {
      if (!Object.prototype.hasOwnProperty.call(value, key)) return false;
    }

    const properties = __mcpIsObject(schema.properties) ? schema.properties : {};
    for (const [key, propSchema] of Object.entries(properties)) {
      if (!Object.prototype.hasOwnProperty.call(value, key)) continue;
      if (__mcpIsObject(propSchema) && !__mcpMatchesSchema(value[key], propSchema)) {
        return false;
      }
    }

    if (schema.additionalProperties === false) {
      for (const key of Object.keys(value)) {
        if (!Object.prototype.hasOwnProperty.call(properties, key)) return false;
      }
    }
  }

  const hasArrayRules = __mcpIsObject(schema.items);
  if (hasArrayRules) {
    if (!Array.isArray(value)) return false;
    if (!value.every((item) => __mcpMatchesSchema(item, schema.items))) return false;
  }

  return true;
};

class __GeneratedMcpService {
  constructor({
    baseUrl = '{{MCP_BASE_PATH}}/tools',
    llmUrl = '{{MCP_BASE_PATH}}/tgi/v1/chat/completions',
    model = 'tgi',
  } = {}) {
    this.baseUrl = baseUrl;
    this.llmUrl = llmUrl;
    this.model = model;
    this.__testResponses = new Map();
    this.__testCalls = [];
    this.test = {
      addResponse: (name, payload, options = {}) => {
        this._addTestResponse(name, payload, options);
        return this;
      },
      addResolved: (name, value) => {
        this._addTestResponse(name, value, { mode: 'resolved' });
        return this;
      },
      clearResponses: (name) => {
        const key = String(name || '');
        if (key) this.__testResponses.delete(key);
        return this;
      },
      reset: () => {
        this.__testResponses.clear();
        this.__testCalls = [];
        return this;
      },
      getCalls: () => [...this.__testCalls],
    };
  }

  async call(name, body = {}, options = {}) {
    const mockedEntry = this._consumeTestResponse(name);
    if (mockedEntry !== undefined) {
      this.__testCalls.push({
        name,
        body: body || {},
        options: options || {},
        mocked: true,
        timestamp: Date.now(),
      });
      try {
        const resolved = await this._resolveMockedResponse(mockedEntry, options);
        __mcpAppendRuntimeExchange({
          tool: name,
          request_body: body || {},
          request_options: options || {},
          response_payload: resolved,
          mocked: true,
        });
        return resolved;
      } catch (error) {
        __mcpAppendRuntimeExchange({
          tool: name,
          request_body: body || {},
          request_options: options || {},
          error: error?.message || String(error),
          mocked: true,
        });
        throw error;
      }
    }

    const response = await fetch(`${this.baseUrl}/${name}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
      mode: 'cors',
      credentials: 'include',
    });
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    this.__testCalls.push({
      name,
      body: body || {},
      options: options || {},
      mocked: false,
      timestamp: Date.now(),
    });
    try {
      const resolved = await this._resolveToolPayload(payload, options);
      __mcpAppendRuntimeExchange({
        tool: name,
        request_body: body || {},
        request_options: options || {},
        response_payload: resolved,
        mocked: false,
      });
      return resolved;
    } catch (error) {
      __mcpAppendRuntimeExchange({
        tool: name,
        request_body: body || {},
        request_options: options || {},
        error: error?.message || String(error),
        mocked: false,
      });
      throw error;
    }
  }

  async _resolveMockedResponse(entry, options = {}) {
    if (entry && typeof entry === 'object' && entry.__testMockMode) {
      if (entry.__testMockMode === 'resolved') {
        return this._pick(entry.payload, options?.resultKey);
      }
      return this._resolveToolPayload(entry.payload, options);
    }
    return this._resolveToolPayload(entry, options);
  }

  async _resolveToolPayload(payload, options = {}) {
    if (payload?.isError) {
      throw new Error(this._toolErrorMessage(payload));
    }
    if (
      payload
      && typeof payload === 'object'
      && !Array.isArray(payload)
      && Object.prototype.hasOwnProperty.call(payload, 'structuredContent')
      && payload.structuredContent != null
    ) {
      return this._pick(payload.structuredContent, options.resultKey);
    }

    const text = this._toolText(payload);
    if (text) {
      const extracted = await this.extract(text, options);
      return this._pick(extracted, options.resultKey);
    }

    return this._resolvePayloadFallback(payload, options);
  }

  _resolvePayloadFallback(payload, options = {}) {
    if (this._looksLikeToolEnvelope(payload)) {
      const fromEnvelopeBody = this._extractEnvelopeBody(payload);
      const pickedEnvelopeBody = this._pick(fromEnvelopeBody, options.resultKey);
      if (pickedEnvelopeBody != null) return pickedEnvelopeBody;
      return fromEnvelopeBody ?? null;
    }

    const fromContent = this._pick(payload?.content ?? null, options.resultKey);
    if (fromContent != null) return fromContent;

    const fromPayload = this._pick(payload, options.resultKey);
    if (fromPayload != null) return fromPayload;

    return payload ?? null;
  }

  _extractEnvelopeBody(payload) {
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return payload;

    if (Object.prototype.hasOwnProperty.call(payload, 'structuredContent')) {
      if (payload.structuredContent != null) return payload.structuredContent;
    }

    for (const key of ['result', 'data', 'body', 'output', 'value']) {
      if (Object.prototype.hasOwnProperty.call(payload, key)) {
        return payload[key];
      }
    }

    const envelopeKeys = new Set(['isError', 'structuredContent', 'content', 'error']);
    const remaining = {};
    for (const [key, value] of Object.entries(payload)) {
      if (!envelopeKeys.has(key)) remaining[key] = value;
    }
    if (Object.keys(remaining).length > 0) return remaining;
    return null;
  }

  _addTestResponse(name, payload, options = {}) {
    const key = String(name || '').trim();
    if (!key) throw new Error('test.addResponse(name, payload) requires a non-empty name');
    const requestedMode = String(options?.mode || 'auto').toLowerCase();
    const mode = requestedMode === 'auto'
      ? (this._looksLikeToolEnvelope(payload) ? 'raw' : 'resolved')
      : requestedMode;
    if (mode !== 'raw' && mode !== 'resolved') {
      throw new Error('test.addResponse(name, payload, options) mode must be "auto", "raw", or "resolved"');
    }
    const queue = this.__testResponses.get(key) || [];
    queue.push({
      __testMockMode: mode,
      payload,
    });
    this.__testResponses.set(key, queue);
  }

  _looksLikeToolEnvelope(payload) {
    if (!payload || typeof payload !== 'object' || Array.isArray(payload)) return false;
    return (
      Object.prototype.hasOwnProperty.call(payload, 'isError')
      || Object.prototype.hasOwnProperty.call(payload, 'structuredContent')
      || Object.prototype.hasOwnProperty.call(payload, 'content')
      || Object.prototype.hasOwnProperty.call(payload, 'error')
    );
  }

  _consumeTestResponse(name) {
    const key = String(name || '').trim();
    const queue = this.__testResponses.get(key);
    const direct = this._consumeTestResponseQueueEntry(key, queue);
    if (direct !== undefined) return direct;

    const wildcardKey = '*';
    const wildcard = this.__testResponses.get(wildcardKey);
    const fallback = this._consumeTestResponseQueueEntry(wildcardKey, wildcard);
    if (fallback !== undefined) return fallback;

    return undefined;
  }

  _consumeTestResponseQueueEntry(key, queue) {
    if (!Array.isArray(queue) || queue.length === 0) return undefined;
    // Keep the last configured mock sticky so repeated refetches stay deterministic.
    if (queue.length === 1) return queue[0];
    const payload = queue.shift();
    if (queue.length === 0) {
      this.__testResponses.delete(key);
    } else {
      this.__testResponses.set(key, queue);
    }
    return payload;
  }

  async extract(text, options = {}) {
    const sourceText = typeof text === 'string' ? text : __mcpStringify(text);
    const direct = __mcpParseJson(sourceText);
    const schema = options?.schema;
    if (direct !== null) {
      if (!__mcpIsObject(schema) || __mcpMatchesSchema(direct, schema)) {
        return direct;
      }
    }
    const capped = __mcpCapExtractionText(sourceText);
    if (capped.truncated && direct === null) {
      return { text: capped.text, truncated: true };
    }
    try {
      return await this._extractWithLlm(capped.text, options);
    } catch (_err) {
      console.error('Extraction failed, returning original text. Error:', _err);
      return { text: capped.text, truncated: capped.truncated };
    }
  }

  _pick(value, key) {
    if (!key) return value;
    if (value == null) return value;
    // Be resilient when callers provide a resultKey that is absent.
    // In that case, return the full payload instead of collapsing to null.
    if (value && typeof value === 'object' && !Array.isArray(value)) {
      if (Object.prototype.hasOwnProperty.call(value, key)) {
        return value[key];
      }
      return value;
    }
    return value;
  }

  _toolText(payload) {
    const content = payload?.content;
    if (typeof content === 'string') return content.trim();
    if (Array.isArray(content)) {
      return content
        .map((entry) => {
          if (typeof entry === 'string') return entry;
          if (entry && typeof entry === 'object') {
            const candidate = entry.text ?? entry.content ?? entry.value ?? '';
            if (typeof candidate === 'string') return candidate;
            return __mcpStringify(candidate);
          }
          return '';
        })
        .filter(Boolean)
        .join('\n')
        .trim();
    }
    return '';
  }

  _toolErrorMessage(payload) {
    const text = this._toolText(payload);
    return text || payload?.error || 'Tool call failed';
  }

  async _extractWithLlm(text, options = {}) {
    const schemaHint = options?.schema
      ? `Schema: ${__mcpStringify(options.schema)}`
      : '';
    const keyHint = options?.resultKey
      ? `If possible, return an object containing key "${options.resultKey}".`
      : '';
    const instruction =
      options?.extractionPrompt ||
      [
        'Extract structured JSON from the tool output text.',
        keyHint,
        schemaHint,
        'Return valid JSON only.',
      ]
        .filter(Boolean)
        .join('\n');

    const requestBody = {
      messages: [
        { role: 'system', content: instruction },
        { role: 'user', content: text },
      ],
      model: options?.model || this.model,
      stream: true,
      tool_choice: 'auto',
    };

    if (!!options?.schema) {
      requestBody.response_format = {
        type: 'json_object',
        json_schema: {
          name: 'ExtractedData',
          schema: options.schema
        }
      };
    }

    const response = await fetch(this.llmUrl, {
      headers: {
        accept: 'text/event-stream',
        'content-type': 'application/json',
      },
      body: JSON.stringify(requestBody),
      method: 'POST',
      mode: 'cors',
      credentials: 'include',
    });
    if (!response.ok) {
      throw new Error(`Extractor HTTP ${response.status}`);
    }

    const completion = await this._readCompletionResponse(response);
    const parsed = __mcpParseJson(completion);
    if (parsed !== null) return parsed;
    throw new Error('Extractor did not return valid JSON');
  }

  async _readCompletionResponse(response) {
    if (response?.body && typeof response.body.getReader === 'function') {
      return await this._readSseResponse(response);
    }
    if (typeof response?.json === 'function') {
      const payload = await response.json();
      const token = this._extractCompletionToken(payload);
      if (token) return token;
      return typeof payload === 'string' ? payload : __mcpStringify(payload);
    }
    if (typeof response?.text === 'function') {
      return await response.text();
    }
    return '';
  }

  async _readSseResponse(response) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let content = '';

    const consumeLine = (line) => {
      const trimmed = line.trim();
      if (!trimmed.startsWith('data:')) return;
      const payload = trimmed.slice(5).trim();
      if (!payload || payload === '[DONE]') return;
      const parsed = __mcpParseJson(payload);
      if (parsed !== null) {
        const token = this._extractCompletionToken(parsed);
        if (token) {
          content += token;
          return;
        }
      }
      content += payload;
    };

    while (true) {
      const chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';
      lines.forEach(consumeLine);
    }

    const tail = buffer.trim();
    if (tail) consumeLine(tail);
    return content.trim();
  }

  _extractCompletionToken(payload) {
    if (typeof payload === 'string') return payload;
    const choice = payload?.choices?.[0];
    return (
      choice?.delta?.content ||
      choice?.message?.content ||
      payload?.content ||
      null
    );
  }
}

if (typeof globalThis !== 'undefined') {
  globalThis.__generatedUiCollectServiceExchanges = __mcpCollectRuntimeExchanges;
  globalThis.__generatedUiClearServiceExchanges = __mcpClearRuntimeExchanges;
}
