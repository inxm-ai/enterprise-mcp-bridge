import { trace, context, propagation } from 'https://esm.sh/@opentelemetry/api@1.9.0';
import {
  WebTracerProvider,
  BatchSpanProcessor
} from "https://esm.sh/@opentelemetry/sdk-trace-web@2.0.1"
import { resourceFromAttributes } from "https://esm.sh/@opentelemetry/resources@2.0.1";
import {
  OTLPTraceExporter
} from "https://esm.sh/@opentelemetry/exporter-trace-otlp-http@0.203.0";
import { ZoneContextManager } from 'https://esm.sh/@opentelemetry/context-zone@2.0.1';

import {ATTR_SERVICE_NAME } from "https://esm.sh/@opentelemetry/semantic-conventions@1.36.0";
import { W3CTraceContextPropagator } from 'https://esm.sh/@opentelemetry/core@2.0.1';


const exporter = new OTLPTraceExporter({
  url: 'https://inxm.local/ops/otel-collector/v1/traces',
  headers: {}
});

const provider = new WebTracerProvider({
  resource: resourceFromAttributes({
    [ATTR_SERVICE_NAME]: 'app-web',
  }),
  spanProcessors: [
    new BatchSpanProcessor(exporter)
  ]
});
trace.setGlobalTracerProvider(provider);
propagation.setGlobalPropagator(new W3CTraceContextPropagator());


provider.register({
    contextManager: new ZoneContextManager(),
});

const tracer = provider.getTracer('fetch-tracer');


/**
 * Wraps the native fetch function with OpenTelemetry tracing.
 *
 * @param {RequestInfo} input The resource to fetch.
 * @param {RequestInit} [init] An options object.
 * @returns {Promise<Response>} A promise that resolves to the Response object.
 */
export async function otelFetch(input, init = {}) {
  const span = tracer.startSpan('http-client-fetch', {
    attributes: {
      'http.url': typeof input === 'string' ? input : input.url,
      'http.method': init.method || 'GET',
      'otel.scope.name': 'fetch-tracer',
      'otel.scope.version': '1.0.0',
    },
    kind: 2 // Client kind
  });
  
  const headers = new Headers(init.headers);
  const activeContext = context.active();
  const spanContext = span.spanContext();

  if (activeContext) {
    propagation.inject(trace.setSpan(activeContext, span), headers);
  } else {
    console.warn('No active context found. Headers will not include traceparent.');
  }

  // Convert Headers object to a plain object using Object.fromEntries
  const plainHeaders = Object.entries(headers).reduce((acc, [key, value]) => {
    acc[key] = value;
    return acc;
  }, {});

  init.headers = {...init.headers, ...plainHeaders};

  console.log(init.headers)

  try {
    const response = await fetch(input, init);
    if (response.status === 401) {
      span.setAttribute('error', true);
      span.setStatus({
        code: 2, // SpanStatusCode.ERROR
        message: 'Unauthorized - User may be logged out'
      });
      window.location.href = '/oauth2/sign_out';
    }

    span.setAttribute('http.status_code', response.status);
    span.setAttribute('http.status_text', response.statusText);
    span.setStatus({
      code: 1, // SpanStatusCode.OK
      message: 'OK'
    });
    return response;
  } catch (error) {
    span.setAttribute('error', true);
    span.setStatus({
      code: 2, // SpanStatusCode.ERROR
      message: error.message
    });
    throw error;
  } finally {
    span.end();
  }
}

export { trace, context, propagation };
