const API_URL = import.meta.env.VITE_API_URL ?? "http://127.0.0.1:8000";

export async function getJson(path) {
  const response = await fetch(`${API_URL}${path}`);
  return parseResponse(response);
}

export async function postJson(path, body = {}) {
  const response = await fetch(`${API_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  return parseResponse(response);
}

async function parseResponse(response) {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(payload.detail ?? `HTTP ${response.status}`);
  }
  return payload;
}
