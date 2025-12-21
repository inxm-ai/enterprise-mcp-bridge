export async function otelFetch(input, init = {}) {
  // Ignore AbortSignal to avoid client-side timeouts.
  const { signal: _signal, ...fetchInit } = init;
  const response = await fetch(input, {
    ...fetchInit,
    headers: {
      'X-Requested-With': 'XMLHttpRequest',
      ...(fetchInit.headers || {})
    }
  });

  if (response.status === 401) {
    window.location.href = '/oauth2/sign_out';
  }

  return response;
}
