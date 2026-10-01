/* tslint:disable */
/* eslint-disable */

export class WasmJulia {
    private constructor();
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Loads a checkpoint from its file contents. `head_length` is the token budget for the question head
     * (the Python runtime's default for the typed API is 512).
     */
    static load(weights: Uint8Array, tokenizer_json: Uint8Array, tokenizer_config: string, julia_config: string, encoder_config: string, head_length: number): WasmJulia;
    /**
     * Answers named typed questions about a state: `state_json` is a JSON string or object, `questions_json`
     * maps IDs to `{type, instructions, criteria}` (`choice` / `score` / `noul`). Returns the same JSON the
     * `julia1 predict` command prints for one row.
     */
    predict(state_json: string, questions_json: string): string;
}

/**
 * Readable panic messages in the browser console.
 */
export function init_panic_hook(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmjulia_free: (a: number, b: number) => void;
    readonly init_panic_hook: () => void;
    readonly wasmjulia_load: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number, i: number, j: number, k: number) => [number, number, number];
    readonly wasmjulia_predict: (a: number, b: number, c: number, d: number, e: number) => [number, number, number, number];
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_exn_store: (a: number) => void;
    readonly __externref_table_alloc: () => number;
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
