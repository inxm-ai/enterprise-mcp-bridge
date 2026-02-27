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
    gatewayListServers = '{{GENERATED_UI_GATEWAY_LIST_SERVERS}}',
    gatewayListTools = '{{GENERATED_UI_GATEWAY_LIST_TOOLS}}',
    gatewayGetTool = '{{GENERATED_UI_GATEWAY_GET_TOOL}}',
    gatewayCallTool = '{{GENERATED_UI_GATEWAY_CALL_TOOL}}',
    gatewayRoleArgs = {{GENERATED_UI_GATEWAY_ROLE_ARGS_JSON}},
    gatewayPromptArgMaxChars = {{GENERATED_UI_GATEWAY_PROMPT_ARG_MAX_CHARS}},
    gatewayServerIdFields = {{GENERATED_UI_GATEWAY_SERVER_ID_FIELDS_JSON}},
    gatewayServerIdUrlRegex = {{GENERATED_UI_GATEWAY_SERVER_ID_URL_REGEX_JSON}},
  } = {}) {
    this.baseUrl = baseUrl;
    this.llmUrl = llmUrl;
    this.model = model;
    this.gatewayListServers = this._normalizeGatewayToolName(gatewayListServers, 'get_servers');
    this.gatewayListTools = this._normalizeGatewayToolName(gatewayListTools, 'get_tools');
    this.gatewayGetTool = this._normalizeGatewayToolName(gatewayGetTool, 'get_tool');
    this.gatewayCallTool = this._normalizeGatewayToolName(gatewayCallTool, 'call_tool');
    this.gatewayRoleArgs = this._normalizeGatewayRoleArgs(gatewayRoleArgs);
    this.gatewayPromptArgMaxChars = this._normalizeGatewayPromptLimit(gatewayPromptArgMaxChars, 800);
    this.gatewayServerIdFields = this._normalizeGatewayServerIdFields(gatewayServerIdFields);
    this.gatewayServerIdRegex = this._compileGatewayServerIdRegex(gatewayServerIdUrlRegex);
    this.__testResponses = new Map();
    this.__testCalls = [];
    this.__gatewayToolRouteCache = new Map();
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
    const callName = String(name || '').trim();
    const callBody = __mcpIsObject(body) ? body : {};
    const callOptions = __mcpIsObject(options) ? options : {};

    const mockedEntry = this._consumeTestResponse(callName);
    if (mockedEntry !== undefined) {
      this.__testCalls.push({
        name: callName,
        body: callBody,
        options: callOptions,
        mocked: true,
        timestamp: Date.now(),
      });
      try {
        const resolved = await this._resolveMockedResponse(mockedEntry, callOptions);
        __mcpAppendRuntimeExchange({
          tool: callName,
          request_body: callBody,
          request_options: callOptions,
          response_payload: resolved,
          mocked: true,
        });
        return resolved;
      } catch (error) {
        __mcpAppendRuntimeExchange({
          tool: callName,
          request_body: callBody,
          request_options: callOptions,
          error: error?.message || String(error),
          mocked: true,
        });
        throw error;
      }
    }

    try {
      let payload = null;
      const routeHint = this._resolveGatewayRouteHint({
        name: callName,
        body: callBody,
        options: callOptions,
      });

      if (routeHint && !this._isGatewayMetaToolName(callName)) {
        const routed = await this._callThroughGateway(routeHint, callBody);
        payload = routed.payload;
      } else {
        const direct = await this._invokeToolEndpoint(callName, callBody);
        const shouldDiscover = (
          !this._isGatewayMetaToolName(callName)
          && (this._isUnknownToolResult(direct) || Number(direct?.status || 0) === 404)
        );
        if (shouldDiscover) {
          const discovered = await this._discoverGatewayRoute(callName, callBody, callOptions);
          if (discovered) {
            const routed = await this._callThroughGateway(discovered, callBody);
            payload = routed.payload;
          }
        }
        if (payload == null) {
          payload = this._unwrapToolFetchResultOrThrow(direct);
        }
      }

      this.__testCalls.push({
        name: callName,
        body: callBody,
        options: callOptions,
        mocked: false,
        timestamp: Date.now(),
      });

      const resolved = await this._resolveToolPayload(payload, callOptions);
      __mcpAppendRuntimeExchange({
        tool: callName,
        request_body: callBody,
        request_options: callOptions,
        response_payload: resolved,
        mocked: false,
      });
      return resolved;
    } catch (error) {
      const lastCall = this.__testCalls[this.__testCalls.length - 1];
      const alreadyRecorded = Boolean(
        lastCall
        && lastCall.name === callName
        && lastCall.mocked === false
        && lastCall.timestamp
      );
      if (!alreadyRecorded) {
        this.__testCalls.push({
          name: callName,
          body: callBody,
          options: callOptions,
          mocked: false,
          timestamp: Date.now(),
        });
      }
      __mcpAppendRuntimeExchange({
        tool: callName,
        request_body: callBody,
        request_options: callOptions,
        error: error?.message || String(error),
        mocked: false,
      });
      throw error;
    }
  }

  async callTool(name, body = {}, options = {}) {
    return this.call(name, body, options);
  }

  _normalizeGatewayToolName(value, fallback) {
    const text = String(value || '').trim();
    if (!text || text.includes('{{')) return fallback;
    return text;
  }

  _normalizeGatewayRoleArgs(value) {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return {};
    }
    const out = {};
    for (const [role, template] of Object.entries(value)) {
      if (typeof role !== 'string') continue;
      if (!template || typeof template !== 'object' || Array.isArray(template)) continue;
      out[role] = template;
    }
    return out;
  }

  _normalizeGatewayPromptLimit(value, fallback = 800) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed <= 0) return fallback;
    return Math.trunc(parsed);
  }

  _normalizeGatewayServerIdFields(value) {
    const defaults = [
      'server_id',
      'server',
      'meta.server_id',
      'meta.server',
      'mcp_server_id',
      'meta.mcp_server_id',
      'url',
      'meta.url',
    ];
    if (!Array.isArray(value)) return defaults;
    const normalized = value
      .map((item) => String(item || '').trim())
      .filter(Boolean);
    return normalized.length > 0 ? normalized : defaults;
  }

  _compileGatewayServerIdRegex(value) {
    const text = String(value || '').trim();
    if (!text || text.includes('{{')) {
      return /\/api\/([^/]+)\/tools\/[^/?#]+/;
    }
    const jsCompatible = text.replace(/\(\?P<([A-Za-z_][A-Za-z0-9_]*)>/g, '(?<$1>');
    try {
      return new RegExp(jsCompatible);
    } catch (_err) {
      return /\/api\/([^/]+)\/tools\/[^/?#]+/;
    }
  }

  _isGatewayMetaToolName(name) {
    const normalized = String(name || '').trim();
    if (!normalized) return false;
    return new Set([
      this.gatewayListServers,
      this.gatewayCallTool,
      this.gatewayListTools,
      this.gatewayGetTool,
      'call_tool',
      'get_tools',
      'select_tools',
      'get_tool',
      'get_servers',
    ]).has(normalized);
  }

  async _invokeToolEndpoint(name, body = {}) {
    const response = await fetch(`${this.baseUrl}/${name}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body || {}),
      mode: 'cors',
      credentials: 'include',
    });

    let payload = null;
    let rawText = '';
    try {
      payload = await response.json();
    } catch (_jsonErr) {
      try {
        if (typeof response?.text === 'function') {
          rawText = await response.text();
          const parsed = __mcpParseJson(rawText);
          payload = parsed !== null ? parsed : { error: rawText };
        } else {
          payload = null;
        }
      } catch (_textErr) {
        payload = null;
      }
    }
    return {
      ok: Boolean(response?.ok),
      status: Number(response?.status || 0),
      payload,
      raw_text: rawText,
      tool: String(name || ''),
    };
  }

  _unwrapToolFetchResultOrThrow(result) {
    if (result?.ok) {
      if (result.payload == null) {
        throw new Error('Tool response was not valid JSON');
      }
      return result.payload;
    }
    const payload = result?.payload;
    if (payload != null) {
      const message = this._toolErrorMessage(payload);
      if (message && message !== 'Tool call failed') {
        throw new Error(message);
      }
    }
    throw new Error(`HTTP ${result?.status || 500}`);
  }

  _isUnknownToolResult(result) {
    const payload = result?.payload;
    const rawText = typeof result?.raw_text === 'string' ? result.raw_text : '';
    if (payload == null && !rawText) return false;
    const asText = [
      this._toolErrorMessage(payload),
      typeof payload?.error === 'string' ? payload.error : '',
      typeof payload?.message === 'string' ? payload.message : '',
      rawText,
    ]
      .filter(Boolean)
      .join(' ')
      .toLowerCase();
    return asText.includes('unknown tool');
  }

  _resolveGatewayRouteHint({ name, body, options }) {
    const fromName = this._extractGatewayRouteFromString(name);
    if (fromName) {
      this._rememberGatewayRoute(fromName, [name, fromName.tool_name]);
      return fromName;
    }

    const fromOptions = this._extractGatewayRouteFromCarrier(options);
    if (fromOptions) {
      this._rememberGatewayRoute(fromOptions, [name, fromOptions.tool_name]);
      return fromOptions;
    }

    const fromBody = this._extractGatewayRouteFromCarrier(body);
    if (fromBody) {
      this._rememberGatewayRoute(fromBody, [name, fromBody.tool_name]);
      return fromBody;
    }

    return this._gatewayRouteFromCache(name);
  }

  _gatewayRouteFromCache(name) {
    const key = String(name || '').trim();
    if (!key) return null;
    const cached = this.__gatewayToolRouteCache.get(key);
    return cached ? { ...cached } : null;
  }

  _extractGatewayRouteFromString(value) {
    const text = String(value || '').trim();
    if (!text) return null;
    const match = text.match(/\/api\/([^/]+)\/tools\/([^/?#]+)/);
    if (!match) return null;
    const serverId = this._decodeUriPart(match[1]);
    const toolName = this._decodeUriPart(match[2]);
    if (!serverId || !toolName) return null;
    return {
      server_id: serverId,
      tool_name: toolName,
      via_tool: this.gatewayCallTool,
      mcp_server_id: `/api/${serverId}/tools/${toolName}`,
    };
  }

  _decodeUriPart(value) {
    const text = String(value || '').trim();
    if (!text) return '';
    try {
      return decodeURIComponent(text);
    } catch (_err) {
      return text;
    }
  }

  _extractGatewayRouteFromCarrier(carrier) {
    if (!carrier || typeof carrier !== 'object' || Array.isArray(carrier)) return null;
    const directRoute = this._extractGatewayRouteFromString(carrier.mcp_server_id || carrier.url);
    if (directRoute) {
      return {
        ...directRoute,
        via_tool: String(carrier.via_tool || directRoute.via_tool || this.gatewayCallTool),
      };
    }

    const meta = carrier.meta;
    if (meta && typeof meta === 'object' && !Array.isArray(meta)) {
      const metaRoute = this._extractGatewayRouteFromString(meta.mcp_server_id || meta.url);
      if (metaRoute) {
        return {
          ...metaRoute,
          via_tool: String(meta.via_tool || carrier.via_tool || this.gatewayCallTool),
        };
      }
    }

    const gateway = carrier.gateway || carrier.gatewayHint;
    if (gateway && typeof gateway === 'object' && !Array.isArray(gateway)) {
      const gatewayRoute = this._extractGatewayRouteFromCarrier(gateway);
      if (gatewayRoute) return gatewayRoute;
    }

    const serverId = String(
      carrier.server_id || carrier.serverId || carrier.mcp_server || ''
    ).trim();
    const toolName = String(carrier.tool_name || carrier.toolName || '').trim();
    if (serverId && toolName) {
      return {
        server_id: serverId,
        tool_name: toolName,
        via_tool: String(carrier.via_tool || this.gatewayCallTool),
        mcp_server_id: `/api/${serverId}/tools/${toolName}`,
      };
    }
    return null;
  }

  _rememberGatewayRoute(route, aliases = []) {
    if (!route || !route.server_id || !route.tool_name) return null;
    const normalized = {
      server_id: String(route.server_id),
      tool_name: String(route.tool_name),
      via_tool: String(route.via_tool || this.gatewayCallTool),
      mcp_server_id: String(
        route.mcp_server_id || `/api/${route.server_id}/tools/${route.tool_name}`
      ),
    };
    const keys = new Set([
      normalized.tool_name,
      normalized.mcp_server_id,
      ...aliases.map((entry) => String(entry || '').trim()),
    ]);
    for (const key of keys) {
      if (!key) continue;
      this.__gatewayToolRouteCache.set(key, normalized);
    }
    return normalized;
  }

  _stripGatewayMetadata(body) {
    if (!body || typeof body !== 'object' || Array.isArray(body)) {
      return body ?? {};
    }
    const cleaned = { ...body };
    delete cleaned.mcp_server_id;
    delete cleaned.url;
    delete cleaned.server_id;
    delete cleaned.tool_name;
    delete cleaned.via_tool;
    delete cleaned.gateway;
    delete cleaned.gatewayHint;
    if (cleaned.meta && typeof cleaned.meta === 'object' && !Array.isArray(cleaned.meta)) {
      const meta = { ...cleaned.meta };
      delete meta.mcp_server_id;
      delete meta.url;
      delete meta.via_tool;
      cleaned.meta = meta;
    }
    return cleaned;
  }

  async _callThroughGateway(route, body) {
    const normalizedRoute = this._rememberGatewayRoute(route, [route.tool_name]);
    const viaTool = normalizedRoute?.via_tool || this.gatewayCallTool;
    const gatewayBody = {
      server_id: normalizedRoute.server_id,
      tool_name: normalizedRoute.tool_name,
      input_data: this._stripGatewayMetadata(body || {}),
    };
    const result = await this._invokeToolEndpoint(viaTool, gatewayBody);
    const payload = this._unwrapToolFetchResultOrThrow(result);
    return { payload, route: normalizedRoute };
  }

  _extractGatewayToolEntries(payload) {
    const candidates = [];
    if (payload != null) candidates.push(payload);
    const structured = payload?.structuredContent;
    if (structured != null) candidates.push(structured);
    const content = payload?.content;
    if (typeof content === 'string') {
      const parsed = __mcpParseJson(content);
      if (parsed != null) candidates.push(parsed);
    }
    if (Array.isArray(content)) {
      for (const entry of content) {
        if (typeof entry === 'string') {
          const parsed = __mcpParseJson(entry);
          if (parsed != null) candidates.push(parsed);
          continue;
        }
        if (entry && typeof entry === 'object') {
          const text = typeof entry.text === 'string' ? entry.text : null;
          if (text) {
            const parsed = __mcpParseJson(text);
            if (parsed != null) candidates.push(parsed);
          }
        }
      }
    }

    const entries = [];
    for (const candidate of candidates) {
      entries.push(...this._extractGatewayToolEntriesFromCandidate(candidate));
    }
    return entries;
  }

  _extractGatewayToolEntriesFromCandidate(candidate) {
    if (Array.isArray(candidate)) {
      return candidate.filter((item) => item && typeof item === 'object' && item.name);
    }
    if (!candidate || typeof candidate !== 'object') return [];
    if (candidate.name) return [candidate];
    for (const key of ['result', 'tools', 'data', 'items']) {
      const nested = candidate[key];
      if (Array.isArray(nested)) {
        return nested.filter((item) => item && typeof item === 'object' && item.name);
      }
      if (nested && typeof nested === 'object' && nested.name) {
        return [nested];
      }
    }
    return [];
  }

  _routeFromGatewayToolEntry(entry) {
    if (!entry || typeof entry !== 'object') return null;
    const fromSelf = this._extractGatewayRouteFromCarrier(entry);
    if (fromSelf) {
      return this._rememberGatewayRoute(fromSelf, [entry.name]);
    }
    const fromMeta = this._extractGatewayRouteFromCarrier(entry.meta);
    if (fromMeta) {
      return this._rememberGatewayRoute(fromMeta, [entry.name]);
    }

    const serverId = String(entry.server_id || entry.server || '').trim();
    const toolName = String(entry.name || entry.tool_name || '').trim();
    if (serverId && toolName) {
      return this._rememberGatewayRoute({
        server_id: serverId,
        tool_name: toolName,
        via_tool: this.gatewayCallTool,
        mcp_server_id: `/api/${serverId}/tools/${toolName}`,
      }, [entry.name]);
    }
    return null;
  }

  _valueByDottedPath(data, path) {
    let current = data;
    for (const part of String(path || '').split('.')) {
      if (!part) continue;
      if (!current || typeof current !== 'object' || Array.isArray(current)) {
        return null;
      }
      if (!Object.prototype.hasOwnProperty.call(current, part)) {
        return null;
      }
      current = current[part];
    }
    return current;
  }

  _extractServerIdFromUrlLike(value) {
    const text = String(value || '').trim();
    if (!text) return '';
    const directRoute = this._extractGatewayRouteFromString(text);
    if (directRoute?.server_id) return directRoute.server_id;
    const regex = this.gatewayServerIdRegex;
    if (regex instanceof RegExp) {
      const match = text.match(regex);
      if (match) {
        if (typeof match.groups?.server_id === 'string' && match.groups.server_id.trim()) {
          return this._decodeUriPart(match.groups.server_id.trim());
        }
        const fallbackGroup = match[1];
        if (typeof fallbackGroup === 'string' && fallbackGroup.trim()) {
          return this._decodeUriPart(fallbackGroup.trim());
        }
      }
    }
    if (!text.includes('/') && !text.includes(' ') && text.length <= 160) {
      return text;
    }
    return '';
  }

  _extractServerIdFromCarrier(carrier) {
    if (!carrier || typeof carrier !== 'object' || Array.isArray(carrier)) return '';
    const route = this._extractGatewayRouteFromCarrier(carrier);
    if (route?.server_id) return route.server_id;
    for (const path of this.gatewayServerIdFields || []) {
      const candidate = this._valueByDottedPath(carrier, path);
      if (typeof candidate === 'string') {
        const serverId = this._extractServerIdFromUrlLike(candidate);
        if (serverId) return serverId;
      }
    }
    return '';
  }

  _extractGatewayServerEntries(payload) {
    const entries = [];
    const queue = [];
    if (payload != null) queue.push(payload);
    if (payload?.structuredContent != null) queue.push(payload.structuredContent);
    if (typeof payload?.content === 'string') {
      const parsed = __mcpParseJson(payload.content);
      if (parsed != null) queue.push(parsed);
    }
    if (Array.isArray(payload?.content)) {
      for (const item of payload.content) {
        if (typeof item === 'string') {
          const parsed = __mcpParseJson(item);
          if (parsed != null) queue.push(parsed);
          continue;
        }
        if (item && typeof item === 'object' && typeof item.text === 'string') {
          const parsed = __mcpParseJson(item.text);
          if (parsed != null) queue.push(parsed);
        }
      }
    }
    for (const candidate of queue) {
      if (Array.isArray(candidate)) {
        entries.push(...candidate.filter((item) => item && typeof item === 'object'));
        continue;
      }
      if (!candidate || typeof candidate !== 'object') continue;
      if (this._extractServerIdFromCarrier(candidate)) entries.push(candidate);
      for (const key of ['servers', 'result', 'data', 'items']) {
        const nested = candidate[key];
        if (Array.isArray(nested)) {
          entries.push(...nested.filter((item) => item && typeof item === 'object'));
        } else if (nested && typeof nested === 'object') {
          entries.push(nested);
        }
      }
    }
    return entries;
  }

  _extractGatewayServerIds(payload) {
    const out = new Set();
    const serverEntries = this._extractGatewayServerEntries(payload);
    for (const entry of serverEntries) {
      const serverId = this._extractServerIdFromCarrier(entry);
      if (serverId) out.add(serverId);
    }
    const toolEntries = this._extractGatewayToolEntries(payload);
    for (const entry of toolEntries) {
      const route = this._routeFromGatewayToolEntry(entry);
      if (route?.server_id) out.add(route.server_id);
    }
    return out;
  }

  _buildGatewayPrompt(toolName, body = {}, options = {}) {
    const candidates = [
      body?.user_query,
      options?.user_query,
      body?.query,
      options?.query,
      body?.prompt,
      options?.prompt,
      toolName,
    ]
      .map((item) => String(item || '').trim())
      .filter(Boolean);
    const prompt = candidates[0] || '';
    if (!prompt) return '';
    if (!Number.isFinite(this.gatewayPromptArgMaxChars) || this.gatewayPromptArgMaxChars <= 0) {
      return prompt;
    }
    return prompt.slice(0, this.gatewayPromptArgMaxChars);
  }

  _resolveGatewayArgTemplateValue({ value, context }) {
    if (typeof value === 'string') {
      const placeholders = {
        '${prompt}': context.prompt,
        '${server_id}': context.server_id,
        '${tool_name}': context.tool_name,
      };
      if (Object.prototype.hasOwnProperty.call(placeholders, value)) {
        const resolved = placeholders[value];
        return (resolved == null) ? [false, null] : [true, resolved];
      }
      let resolvedValue = value;
      for (const [placeholder, replacement] of Object.entries(placeholders)) {
        if (!resolvedValue.includes(placeholder)) continue;
        resolvedValue = resolvedValue.replaceAll(
          placeholder,
          replacement == null ? '' : String(replacement)
        );
      }
      return [true, resolvedValue];
    }
    if (Array.isArray(value)) {
      const out = [];
      for (const item of value) {
        const [include, resolved] = this._resolveGatewayArgTemplateValue({ value: item, context });
        if (include) out.push(resolved);
      }
      return [true, out];
    }
    if (value && typeof value === 'object') {
      const out = {};
      for (const [key, child] of Object.entries(value)) {
        const [include, resolved] = this._resolveGatewayArgTemplateValue({ value: child, context });
        if (include) out[String(key)] = resolved;
      }
      return [true, out];
    }
    return [true, value];
  }

  _buildGatewayCallArgs(role, defaults = {}, context = {}) {
    const args = { ...(defaults || {}) };
    const template = this.gatewayRoleArgs?.[role];
    if (!template || typeof template !== 'object' || Array.isArray(template)) {
      return args;
    }
    for (const [key, templateValue] of Object.entries(template)) {
      const [include, resolved] = this._resolveGatewayArgTemplateValue({
        value: templateValue,
        context,
      });
      if (include) {
        args[String(key)] = resolved;
      } else {
        delete args[String(key)];
      }
    }
    return args;
  }

  _dedupeObjectList(items) {
    const out = [];
    const seen = new Set();
    for (const item of items) {
      const key = __mcpStringify(item || {});
      if (seen.has(key)) continue;
      seen.add(key);
      out.push(item || {});
    }
    return out;
  }

  _extractMissingFieldsFromResult(result) {
    const text = [
      this._toolErrorMessage(result?.payload),
      typeof result?.payload?.error === 'string' ? result.payload.error : '',
      typeof result?.payload?.message === 'string' ? result.payload.message : '',
      typeof result?.raw_text === 'string' ? result.raw_text : '',
    ]
      .filter(Boolean)
      .join('\n');
    const fields = new Set();
    const linePattern = /(?:^|\n)\s*([A-Za-z_][A-Za-z0-9_]*)\s*\n\s*Field required/gi;
    let lineMatch = linePattern.exec(text);
    while (lineMatch) {
      fields.add(String(lineMatch[1] || '').trim());
      lineMatch = linePattern.exec(text);
    }
    const inlinePattern = /missing(?: required)?(?: field| argument)?[:\s]+([A-Za-z_][A-Za-z0-9_]*)/gi;
    let inlineMatch = inlinePattern.exec(text);
    while (inlineMatch) {
      fields.add(String(inlineMatch[1] || '').trim());
      inlineMatch = inlinePattern.exec(text);
    }
    return fields;
  }

  _buildGatewayDiscoveryArgs({
    role,
    context,
    defaults = {},
    includeLegacy = false,
    missingFields = new Set(),
  }) {
    const base = this._buildGatewayCallArgs(role, defaults, context);
    const variants = [base];
    const prompt = String(context?.prompt || '').trim();
    const serverId = String(context?.server_id || '').trim();
    const toolName = String(context?.tool_name || '').trim();

    if (missingFields.has('server_id') && serverId) {
      variants.push({ ...base, server_id: serverId });
    }
    if (missingFields.has('tool_name') && toolName) {
      variants.push({ ...base, tool_name: toolName });
    }
    if (missingFields.has('user_query') && prompt) {
      variants.push({ ...base, user_query: prompt });
    }
    if (missingFields.has('prompt') && prompt) {
      variants.push({ ...base, prompt });
    }
    if (missingFields.has('query') && prompt) {
      variants.push({ ...base, query: prompt });
    }

    if (role === 'list_tools' && serverId) {
      variants.push({ ...base, server_id: serverId });
    }
    if (role === 'get_tool' && serverId && toolName) {
      variants.push({ ...base, server_id: serverId, tool_name: toolName });
    }

    if (includeLegacy && role === 'list_tools' && prompt) {
      variants.push({ ...base, prompt });
      variants.push({ ...base, query: prompt });
    }

    return this._dedupeObjectList(variants);
  }

  _discoveryToolCandidates(primary, fallbacks = []) {
    return [primary, ...fallbacks]
      .map((item) => String(item || '').trim())
      .filter((item, index, arr) => item && arr.indexOf(item) === index);
  }

  _buildGatewayDiscoveryContext(toolName, body = {}, options = {}) {
    const initialServerId = this._extractServerIdFromCarrier(body)
      || this._extractServerIdFromCarrier(options)
      || '';
    return {
      prompt: this._buildGatewayPrompt(toolName, body, options),
      server_id: initialServerId,
      tool_name: String(toolName || '').trim(),
    };
  }

  _ingestGatewayDiscoveryPayload(payload) {
    const serverIds = this._extractGatewayServerIds(payload);
    const routeFromPayload = this._extractGatewayRouteFromCarrier(payload);
    if (routeFromPayload) {
      const remembered = this._rememberGatewayRoute(routeFromPayload, [routeFromPayload.tool_name]);
      if (remembered?.server_id) serverIds.add(remembered.server_id);
    }
    const routeFromStructured = this._extractGatewayRouteFromCarrier(payload?.structuredContent);
    if (routeFromStructured) {
      const remembered = this._rememberGatewayRoute(routeFromStructured, [routeFromStructured.tool_name]);
      if (remembered?.server_id) serverIds.add(remembered.server_id);
    }
    const entries = this._extractGatewayToolEntries(payload);
    for (const entry of entries) {
      const route = this._routeFromGatewayToolEntry(entry);
      if (route?.server_id) serverIds.add(route.server_id);
    }
    return { serverIds };
  }

  async _runGatewayDiscoveryTool({
    toolName,
    role,
    context,
    normalizedTool,
    defaults = {},
    includeLegacy = false,
  }) {
    const seen = new Set();
    const collectedMissing = new Set();
    const collectedServerIds = new Set();
    let queue = this._buildGatewayDiscoveryArgs({
      role,
      context,
      defaults,
      includeLegacy,
      missingFields: new Set(),
    });
    if (queue.length === 0) queue = [{}];

    while (queue.length > 0) {
      const args = queue.shift();
      const key = `${toolName}:${__mcpStringify(args || {})}`;
      if (seen.has(key)) continue;
      seen.add(key);

      let result = null;
      try {
        result = await this._invokeToolEndpoint(toolName, args || {});
      } catch (_err) {
        continue;
      }

      const payload = result?.payload;
      const { serverIds } = this._ingestGatewayDiscoveryPayload(payload);
      for (const serverId of serverIds) {
        collectedServerIds.add(serverId);
      }

      const cached = this._gatewayRouteFromCache(normalizedTool);
      if (cached) {
        return {
          route: cached,
          missingFields: collectedMissing,
          serverIds: collectedServerIds,
        };
      }

      const missing = this._extractMissingFieldsFromResult(result);
      if (missing.size > 0) {
        for (const field of missing) collectedMissing.add(field);
        const additions = this._buildGatewayDiscoveryArgs({
          role,
          context,
          defaults,
          includeLegacy,
          missingFields: missing,
        });
        for (const extra of additions) queue.push(extra);
      }
    }

    return {
      route: this._gatewayRouteFromCache(normalizedTool),
      missingFields: collectedMissing,
      serverIds: collectedServerIds,
    };
  }

  async _discoverGatewayRoute(toolName, body, options = {}) {
    const normalizedTool = String(toolName || '').trim();
    if (!normalizedTool) return null;

    const cachedRoute = this._gatewayRouteFromCache(normalizedTool);
    if (cachedRoute) return cachedRoute;

    const context = this._buildGatewayDiscoveryContext(normalizedTool, body, options);
    const knownServerIds = new Set();
    if (context.server_id) knownServerIds.add(context.server_id);

    const listToolsCandidates = this._discoveryToolCandidates(
      this.gatewayListTools,
      ['get_tools', 'select_tools']
    );
    const listServersCandidates = this._discoveryToolCandidates(
      this.gatewayListServers,
      ['get_servers']
    );
    const getToolCandidates = this._discoveryToolCandidates(
      this.gatewayGetTool,
      ['get_tool']
    );

    let listToolsMissing = new Set();
    for (const listTool of listToolsCandidates) {
      const result = await this._runGatewayDiscoveryTool({
        toolName: listTool,
        role: 'list_tools',
        context,
        normalizedTool,
      });
      if (result.route) return result.route;
      for (const serverId of result.serverIds) knownServerIds.add(serverId);
      for (const missingField of result.missingFields) listToolsMissing.add(missingField);
    }

    if (knownServerIds.size === 0 || listToolsMissing.has('server_id')) {
      for (const listServersTool of listServersCandidates) {
        const result = await this._runGatewayDiscoveryTool({
          toolName: listServersTool,
          role: 'list_servers',
          context,
          normalizedTool,
        });
        if (result.route) return result.route;
        for (const serverId of result.serverIds) knownServerIds.add(serverId);
      }
    }

    if (knownServerIds.size > 0) {
      for (const serverId of knownServerIds) {
        const serverContext = { ...context, server_id: serverId };
        for (const listTool of listToolsCandidates) {
          const result = await this._runGatewayDiscoveryTool({
            toolName: listTool,
            role: 'list_tools',
            context: serverContext,
            normalizedTool,
            defaults: { server_id: serverId },
          });
          if (result.route) return result.route;
        }
      }
    }

    if (knownServerIds.size > 0) {
      for (const serverId of knownServerIds) {
        const getToolContext = {
          ...context,
          server_id: serverId,
          tool_name: normalizedTool,
        };
        for (const getToolName of getToolCandidates) {
          const result = await this._runGatewayDiscoveryTool({
            toolName: getToolName,
            role: 'get_tool',
            context: getToolContext,
            normalizedTool,
            defaults: {
              server_id: serverId,
              tool_name: normalizedTool,
            },
          });
          if (result.route) return result.route;
        }
      }
    }

    for (const listTool of listToolsCandidates) {
      const result = await this._runGatewayDiscoveryTool({
        toolName: listTool,
        role: 'list_tools',
        context,
        normalizedTool,
        includeLegacy: true,
      });
      if (result.route) return result.route;
      for (const serverId of result.serverIds) knownServerIds.add(serverId);
    }

    if (knownServerIds.size > 0) {
      for (const serverId of knownServerIds) {
        const serverContext = { ...context, server_id: serverId };
        for (const listTool of listToolsCandidates) {
          const result = await this._runGatewayDiscoveryTool({
            toolName: listTool,
            role: 'list_tools',
            context: serverContext,
            normalizedTool,
            defaults: { server_id: serverId },
            includeLegacy: true,
          });
          if (result.route) return result.route;
        }
      }
    }

    return this._gatewayRouteFromCache(normalizedTool);
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
