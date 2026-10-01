/** Loads julia1-rs's wasm module and a Julia-1 checkpoint fetched straight from the Hugging Face Hub (or read
 * from a local folder), entirely client-side. Files are cached with the Cache API keyed by their HF URL. */

import { createSignal } from "solid-js";
import {
  fsAccessSupported,
  forgetRememberedDirectory,
  loadRememberedDirectory,
  pickDirectory,
  queryReadPermission,
  readRelativeFile,
  rememberDirectory,
  requestReadPermission,
} from "./localFs";

export { fsAccessSupported };

export const MODEL = {
  label: "Julia-1",
  hfRepo: "SupersonicLabs/Julia-1",
  approxSizeMb: 577,
  description: "144M-parameter decision model on a ModernBERT (mmBERT-small) backbone. FP32, runs on one CPU thread in the browser.",
};

/** What a checkpoint directory holds, as the runtime reads it. */
export const FILES = [
  "julia_config.json",
  "encoder/config.json",
  "tokenizer/tokenizer.json",
  "tokenizer/tokenizer_config.json",
  "model.safetensors",
] as const;

export const hfFileUrl = (file: string) => `https://huggingface.co/${MODEL.hfRepo}/resolve/main/${file}`;

export interface WasmJulia {
  predict(stateJson: string, questionsJson: string): string;
  free(): void;
}

type WasmExports = {
  default: (module_or_path?: unknown) => Promise<unknown>;
  WasmJulia: {
    load(
      weights: Uint8Array,
      tokenizerJson: Uint8Array,
      tokenizerConfig: string,
      juliaConfig: string,
      encoderConfig: string,
      headLength: number,
    ): WasmJulia;
  };
};

let wasmReady: Promise<WasmExports> | null = null;
function loadWasm(): Promise<WasmExports> {
  wasmReady ??= (async () => {
    const mod = (await import("../wasm-pkg/julia1.js")) as unknown as WasmExports;
    await mod.default();
    return mod;
  })();
  return wasmReady;
}

export interface Progress {
  file: string;
  index: number;
  count: number;
  loaded: number;
  total: number | null;
}

const CACHE_NAME = "julia1-rs-checkpoint-v1";

async function fetchFile(url: string, onProgress: (loaded: number, total: number | null) => void): Promise<Uint8Array> {
  const cache = await caches.open(CACHE_NAME);
  const cached = await cache.match(url);
  if (cached) {
    const buf = await cached.arrayBuffer();
    onProgress(buf.byteLength, buf.byteLength);
    return new Uint8Array(buf);
  }
  const resp = await fetch(url);
  if (!resp.ok || !resp.body) throw new Error(`fetching ${url}: HTTP ${resp.status}`);
  const header = resp.headers.get("content-length");
  const total = header ? Number(header) : null;
  const reader = resp.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const c of chunks) {
    bytes.set(c, offset);
    offset += c.byteLength;
  }
  try {
    await cache.put(url, new Response(bytes, { headers: { "content-type": "application/octet-stream" } }));
  } catch {
    /* private mode / quota: the model still loads */
  }
  return bytes;
}

export async function clearCache(): Promise<void> {
  await caches.delete(CACHE_NAME);
}

export async function isCached(): Promise<boolean> {
  if (typeof caches === "undefined") return false;
  try {
    return (await (await caches.open(CACHE_NAME)).match(hfFileUrl("model.safetensors"))) !== undefined;
  } catch {
    return false;
  }
}

export type Source = "hf" | "local";
const [current, setCurrent] = createSignal<{ model: WasmJulia; source: Source; ms: number } | null>(null);
export const loadedModel = current;

/** The question head's token budget: the Python runtime's default for the typed API. */
const HEAD_LENGTH = 512;

async function build(files: Record<string, Uint8Array>, source: Source): Promise<void> {
  const wasm = await loadWasm();
  const text = (name: string) => new TextDecoder().decode(files[name]);
  const t = performance.now();
  // Let the "building" message paint before the synchronous load blocks the thread.
  await new Promise((r) => setTimeout(r, 30));
  const model = wasm.WasmJulia.load(
    files["model.safetensors"],
    files["tokenizer/tokenizer.json"],
    text("tokenizer/tokenizer_config.json"),
    text("julia_config.json"),
    text("encoder/config.json"),
    HEAD_LENGTH,
  );
  current()?.model.free();
  setCurrent({ model, source, ms: performance.now() - t });
}

export async function loadFromHuggingFace(onProgress: (p: Progress) => void, onPhase: (s: string) => void) {
  onPhase("Downloading");
  const files: Record<string, Uint8Array> = {};
  for (let i = 0; i < FILES.length; i++) {
    const file = FILES[i];
    files[file] = await fetchFile(hfFileUrl(file), (loaded, total) =>
      onProgress({ file, index: i, count: FILES.length, loaded, total }),
    );
  }
  onPhase("Building the model");
  await build(files, "hf");
}

/* ---- local folder ---- */

export interface LocalFolder {
  handle: FileSystemDirectoryHandle;
  name: string;
  /** The directory that actually holds `julia_config.json` (the picked one, or its `Julia-1` child). */
  root: FileSystemDirectoryHandle | null;
}

async function hasCheckpoint(d: FileSystemDirectoryHandle): Promise<boolean> {
  try {
    await d.getFileHandle("julia_config.json");
    await d.getFileHandle("model.safetensors");
    return true;
  } catch {
    return false;
  }
}

async function findRoot(dir: FileSystemDirectoryHandle): Promise<FileSystemDirectoryHandle | null> {
  if (await hasCheckpoint(dir)) return dir;
  try {
    const child = await dir.getDirectoryHandle("Julia-1");
    if (await hasCheckpoint(child)) return child;
  } catch {
    /* no such child */
  }
  return null;
}

const [localFolder, setLocalFolder] = createSignal<LocalFolder | null>(null);
const [pendingFolder, setPendingFolder] = createSignal<FileSystemDirectoryHandle | null>(null);
export { localFolder, pendingFolder };

async function adopt(handle: FileSystemDirectoryHandle) {
  setLocalFolder({ handle, name: handle.name, root: await findRoot(handle) });
  setPendingFolder(null);
}

export async function chooseLocalFolder(): Promise<LocalFolder | null> {
  const handle = await pickDirectory();
  if (!handle) return null;
  await rememberDirectory(handle);
  await adopt(handle);
  return localFolder();
}

let resumed = false;
export function resumeLocalFolder(): void {
  if (resumed || localFolder()) return;
  resumed = true;
  void (async () => {
    const handle = await loadRememberedDirectory();
    if (!handle) return;
    if ((await queryReadPermission(handle)) === "granted") await adopt(handle);
    else setPendingFolder(handle);
  })();
}

export async function grantPendingFolder(): Promise<boolean> {
  const handle = pendingFolder();
  if (!handle || (await requestReadPermission(handle)) !== "granted") return false;
  await adopt(handle);
  return true;
}

export async function forgetLocalFolder(): Promise<void> {
  await forgetRememberedDirectory();
  setLocalFolder(null);
  setPendingFolder(null);
}

export async function loadFromLocal(folder: LocalFolder, onPhase: (s: string) => void) {
  if (!folder.root) throw new Error(`No Julia-1 checkpoint found in ${folder.name}/`);
  onPhase("Reading the files");
  const files: Record<string, Uint8Array> = {};
  for (const name of FILES) {
    files[name] = new Uint8Array(await (await readRelativeFile(folder.root, name)).arrayBuffer());
  }
  onPhase("Building the model");
  await build(files, "local");
}

export function unloadModel(): void {
  current()?.model.free();
  setCurrent(null);
}

/* ---- answers ---- */

export interface Answer {
  type: "choice" | "score" | "noul";
  probabilities: Record<string, number>;
  choice?: string;
  score?: number;
  noul?: number;
  max_probability?: number;
}

export function predict(stateJson: string, questionsJson: string): Record<string, Answer> {
  const m = current();
  if (!m) throw new Error("no model loaded");
  return (JSON.parse(m.model.predict(stateJson, questionsJson)) as { answers: Record<string, Answer> }).answers;
}
