export async function otelFetch(input, init = {}) {
  const response = await fetch(input, {
    ...init,
    headers: {
      'X-Requested-With': 'XMLHttpRequest',
      ...(init.headers || {})
    }
  });

  if (response.status === 401) {
    window.location.href = '/oauth2/sign_out';
  }

  return response;
}
