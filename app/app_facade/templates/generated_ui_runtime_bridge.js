(() => {
  if (typeof window === 'undefined' || window.parent === window) return;
  const MAX_LEN = 2000;
  const MAX_ACTION_ENTRIES = 50;
  const trim = (value) => {
    if (value == null) return '';
    const text = String(value);
    return text.length > MAX_LEN ? text.slice(0, MAX_LEN) + '...[truncated]' : text;
  };
  const post = (kind, payload) => {
    try {
      window.parent.postMessage({ source: '{{RUNTIME_BRIDGE_MARKER}}', kind, payload }, window.location.origin);
    } catch (_err) {
    }
  };
  window.addEventListener('error', (event) => {
    post('window_error', {
      message: trim(event?.message),
      stack: trim(event?.error?.stack),
      filename: trim(event?.filename),
      line: event?.lineno || null,
      column: event?.colno || null,
    });
  });
  window.addEventListener('unhandledrejection', (event) => {
    const reason = event?.reason;
    post('unhandled_rejection', {
      message: trim(reason?.message || reason),
      stack: trim(reason?.stack),
    });
  });
  const originalConsoleError = console.error?.bind(console);
  const originalConsoleWarn = console.warn?.bind(console);
  console.error = (...args) => {
    post('console_error', {
      message: trim(args.map((value) => {
        if (typeof value === 'string') return value;
        try { return JSON.stringify(value); } catch (_err) { return String(value); }
      }).join(' ')),
    });
    if (originalConsoleError) originalConsoleError(...args);
  };
  console.warn = (...args) => {
    post('console_warning', {
      message: trim(args.map((value) => {
        if (typeof value === 'string') return value;
        try { return JSON.stringify(value); } catch (_err) { return String(value); }
      }).join(' ')),
    });
    if (originalConsoleWarn) originalConsoleWarn(...args);
  };

  const sanitizeEntry = (entry) => {
    const object = entry && typeof entry === 'object' ? entry : {};
    return {
      cursor: Number(object.cursor) || 0,
      timestamp: Number(object.timestamp) || Date.now(),
      tool: trim(object.tool),
      request_body: object.request_body ?? {},
      request_options: object.request_options ?? {},
      response_payload: object.response_payload ?? null,
      error: trim(object.error),
      mocked: Boolean(object.mocked),
    };
  };

  window.addEventListener('message', (event) => {
    if (event.origin !== window.location.origin) return;
    const data = event.data || {};
    if (data.source !== '{{RUNTIME_BRIDGE_MARKER}}') return;
    if (data.kind !== 'action') return;

    const payload = data.payload || {};
    const action = String(payload.action || '');
    const requestId = trim(payload.request_id || '');

    if (action === 'collect_service_exchanges') {
      const sinceCursor = Number(payload.since_cursor) || 0;
      const limit = Math.max(1, Math.min(MAX_ACTION_ENTRIES, Number(payload.limit) || 20));
      let result = { cursor: sinceCursor, entries: [] };
      try {
        const collector = globalThis.__generatedUiCollectServiceExchanges;
        if (typeof collector === 'function') {
          const response = collector(sinceCursor, limit) || {};
          const entries = Array.isArray(response.entries)
            ? response.entries.slice(0, limit).map(sanitizeEntry)
            : [];
          result = {
            cursor: Number(response.cursor) || sinceCursor,
            entries,
          };
        }
      } catch (_err) {
      }
      post('action_response', {
        request_id: requestId,
        action,
        ...result,
      });
      return;
    }

    if (action === 'clear_service_exchanges') {
      let result = { cursor: 0, cleared: false };
      try {
        const clearer = globalThis.__generatedUiClearServiceExchanges;
        if (typeof clearer === 'function') {
          const response = clearer() || {};
          result = {
            cursor: Number(response.cursor) || 0,
            cleared: Boolean(response.cleared),
          };
        }
      } catch (_err) {
      }
      post('action_response', {
        request_id: requestId,
        action,
        ...result,
      });
    }
  });
})();
