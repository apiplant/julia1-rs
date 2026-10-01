import { Show, createEffect, createSignal, onSettled } from "solid-js";
import { Badge, Button } from "../ui";
import {
  type Progress,
  MODEL,
  chooseLocalFolder,
  clearCache,
  forgetLocalFolder,
  fsAccessSupported,
  grantPendingFolder,
  isCached,
  loadFromHuggingFace,
  loadFromLocal,
  loadedModel,
  localFolder,
  pendingFolder,
  resumeLocalFolder,
  unloadModel,
} from "../../lib/julia";

function fmtBytes(n: number): string {
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / (1024 * 1024)).toFixed(1)} MB`;
}

/** Loads Julia-1 either from Hugging Face (cached in the browser afterwards) or from a folder on disk. */
export function ModelPicker() {
  const [busy, setBusy] = createSignal(false);
  const [phase, setPhase] = createSignal("");
  const [progress, setProgress] = createSignal<Progress | null>(null);
  const [error, setError] = createSignal<string | null>(null);
  const [cached, setCached] = createSignal(false);

  onSettled(() => {
    void isCached().then(setCached);
    resumeLocalFolder();
  });

  async function run(task: () => Promise<void>) {
    setBusy(true);
    setError(null);
    setProgress(null);
    try {
      await task();
      setCached(await isCached());
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
      setPhase("");
      setProgress(null);
    }
  }

  const fromHf = () => run(() => loadFromHuggingFace(setProgress, setPhase));
  const fromLocal = () => {
    const f = localFolder();
    if (f) void run(() => loadFromLocal(f, setPhase));
  };

  async function pick() {
    setError(null);
    try {
      const f = await chooseLocalFolder();
      if (f && !f.root) setError(`No Julia-1 checkpoint (julia_config.json + model.safetensors) found in ${f.name}/.`);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }

  async function grant() {
    setError(null);
    if (!(await grantPendingFolder())) setError("Access wasn't granted.");
  }

  // A remembered, still-readable folder is already on disk: use it without being asked.
  let auto = false;
  createEffect(
    () => localFolder(),
    (f) => {
      if (f?.root && !auto && !loadedModel() && !busy()) {
        auto = true;
        void run(() => loadFromLocal(f, setPhase));
      }
    },
  );

  const pct = () => {
    const p = progress();
    return p?.total ? Math.round((p.loaded / p.total) * 100) : null;
  };

  return (
    <div class="rounded-xl border border-line bg-surface p-4">
      <div class="flex flex-wrap items-center gap-2">
        <p class="text-sm font-medium text-ink">Model</p>
        <Show when={loadedModel()} fallback={<Badge>Not loaded</Badge>}>
          {(m) => (
            <Badge tone="accent">
              Loaded from {m().source === "hf" ? "Hugging Face" : "local folder"} in {(m().ms / 1000).toFixed(1)} s
            </Badge>
          )}
        </Show>
        <Show when={loadedModel()}>
          <button type="button" onClick={unloadModel} class="ml-auto text-xs text-muted underline decoration-dotted hover:text-ink">
            unload
          </button>
        </Show>
      </div>

      <div class="mt-4 grid gap-4 md:grid-cols-2">
        <div class="md:border-r md:border-line md:pr-4">
          <p class="text-sm font-medium text-ink">Hugging Face</p>
          <p class="mt-2 text-sm text-muted">
            {MODEL.description} ~{MODEL.approxSizeMb} MB download.{cached() ? " Already cached in this browser." : ""}
          </p>
          <p class="mt-1 font-mono text-xs text-faint">{MODEL.hfRepo}</p>
          <div class="mt-4 flex items-center gap-3">
            <Button variant="primary" disabled={busy()} onClick={fromHf}>
              {busy() ? "Loading…" : loadedModel()?.source === "hf" ? "Reload" : cached() ? "Load" : "Download & load"}
            </Button>
            <Show when={cached()}>
              <button
                type="button"
                disabled={busy()}
                onClick={async () => {
                  await clearCache();
                  setCached(false);
                }}
                class="text-xs text-muted underline decoration-dotted hover:text-ink disabled:opacity-40"
              >
                clear downloaded cache
              </button>
            </Show>
          </div>
        </div>

        <div>
          <p class="text-sm font-medium text-ink">Local folder</p>
          <Show
            when={fsAccessSupported()}
            fallback={
              <p class="mt-2 text-sm text-danger">
                This browser doesn't support the File System Access API needed to read a local folder. Try Chrome or Edge.
              </p>
            }
          >
            <Show
              when={localFolder()}
              fallback={
                <div class="mt-2">
                  <Show
                    when={pendingFolder()}
                    fallback={
                      <p class="text-sm text-muted">
                        Pick a Julia-1 checkpoint directory (it holds <code class="font-mono text-xs">model.safetensors</code>,{" "}
                        <code class="font-mono text-xs">julia_config.json</code>, <code class="font-mono text-xs">encoder/</code> and{" "}
                        <code class="font-mono text-xs">tokenizer/</code>), or a folder that contains one named{" "}
                        <code class="font-mono text-xs">Julia-1</code>.
                      </p>
                    }
                  >
                    {(h) => (
                      <>
                        <p class="mb-3 rounded-lg border border-accent-line bg-accent-soft px-3 py-2 text-sm text-ink">
                          Resuming <span class="font-mono text-xs">{h().name}/</span> from last time: grant access to reload it.
                        </p>
                        <Button variant="primary" disabled={busy()} onClick={grant}>
                          Grant access
                        </Button>
                      </>
                    )}
                  </Show>
                  <Button class="mt-3" variant={pendingFolder() ? "secondary" : "primary"} disabled={busy()} onClick={pick}>
                    {pendingFolder() ? "Choose a different folder…" : "Choose folder…"}
                  </Button>
                </div>
              }
            >
              {(f) => (
                <div class="mt-2">
                  <p class="font-mono text-xs text-faint">{f().name}/</p>
                  <p class="mt-2 text-sm text-muted">
                    <Show when={f().root} fallback={<span class="text-danger">No Julia-1 checkpoint found in this folder.</span>}>
                      Checkpoint found.
                    </Show>
                  </p>
                  <div class="mt-4 flex items-center gap-3">
                    <Button variant="primary" disabled={busy() || !f().root} onClick={fromLocal}>
                      {busy() ? "Loading…" : loadedModel()?.source === "local" ? "Reload" : "Load"}
                    </Button>
                    <button type="button" disabled={busy()} onClick={pick} class="text-xs text-muted underline decoration-dotted hover:text-ink">
                      change folder
                    </button>
                    <button type="button" disabled={busy()} onClick={() => void forgetLocalFolder()} class="text-xs text-muted underline decoration-dotted hover:text-ink">
                      forget
                    </button>
                  </div>
                </div>
              )}
            </Show>
          </Show>
        </div>
      </div>

      <Show when={busy()}>
        <div class="mt-4">
          <div class="flex justify-between text-xs text-faint">
            <span>
              {phase()}
              {progress() ? ` · ${progress()!.file} (${progress()!.index + 1}/${progress()!.count})` : ""}…
            </span>
            <Show when={progress()}>
              {(p) => (
                <span>
                  {fmtBytes(p().loaded)}
                  {p().total ? ` / ${fmtBytes(p().total!)}` : ""}
                </span>
              )}
            </Show>
          </div>
          <div class="mt-1 h-1.5 w-full overflow-hidden rounded-full bg-surface-3">
            <div class={`h-full rounded-full bg-accent transition-all duration-150 ${pct() === null ? "animate-pulse" : ""}`} style={{ width: `${pct() ?? 100}%` }} />
          </div>
        </div>
      </Show>

      <Show when={error()}>
        <p class="mt-3 text-sm text-danger">{error()}</p>
      </Show>
    </div>
  );
}
