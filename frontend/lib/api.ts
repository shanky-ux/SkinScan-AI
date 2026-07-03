import { apiBaseUrl } from "./data";
import type { ClassItem, PredictResponse } from "./types";

async function parseError(response: Response) {
  const payload = await response.json().catch(() => null);
  return payload?.detail ?? payload?.message ?? "Request failed";
}

export async function fetchClasses(): Promise<ClassItem[]> {
  const response = await fetch(`${apiBaseUrl}/api/classes`, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(await parseError(response));
  }
  return response.json();
}

export async function predictImage(file: File): Promise<PredictResponse> {
  const formData = new FormData();
  formData.append("file", file);

  console.log("[API] Uploading to:", `${apiBaseUrl}/api/predict`, "file:", file.name, file.size, file.type);

  const response = await fetch(`${apiBaseUrl}/api/predict`, {
    method: "POST",
    body: formData,
  });

  console.log("[API] Response status:", response.status, "ok:", response.ok);

  if (!response.ok) {
    const errText = await response.text();
    console.log("[API] Error body:", errText);
    throw new Error(await parseError(response));
  }

  const json = await response.json();
  console.log("[API] Response json:", json);
  return json;
}