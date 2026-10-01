import { For, Show, createSignal } from "solid-js";
import { ModelPicker } from "./ModelPicker";
import { JsonEditor } from "./JsonEditor";
import { JsonView } from "./JsonView";
import { Badge, Button } from "../ui";
import { PRESETS, type Preset } from "../../lib/presets";
import { type Answer, loadedModel, predict } from "../../lib/julia";

function pct(p: number): string {
  return `${(p * 100).toFixed(1)}%`;
}

/** One answered question: the verdict, then every option's probability as a bar. */
function AnswerCard(props: { id: string; answer: Answer; question: unknown }) {
  const a = () => props.answer;
  const entries = () => Object.entries(a().probabilities);
  const sorted = () => (a().type === "score" ? entries() : [...entries()].sort((x, y) => y[1] - x[1]));
  const top = () => Math.max(...entries().map(([, p]) => p));
  const instructions = () => {
    const q = props.question as { instructions?: unknown } | undefined;
    const i = q?.instructions;
    if (typeof i === "string") return i;
    if (i && typeof i === "object" && "question" in i) return String((i as { question: unknown }).question);
    return "";
  };
  const label = (key: string) => {
    const criteria = (props.question as { criteria?: unknown } | undefined)?.criteria;
    return a().type === "score" && Array.isArray(criteria) ? `${key} · ${String(criteria[Number(key)] ?? "")}` : key;
  };
  const verdict = () => {
    const x = a();
    if (x.type === "choice") return x.choice ?? "";
    if (x.type === "noul") return (x.noul ?? 0) >= 0.5 ? "yes" : "no";
    return `score ${(x.score ?? 0).toFixed(2)}`;
  };
  return (
    <div class="rounded-xl border border-line bg-surface p-4">
      <div class="flex flex-wrap items-center gap-2">
        <h3 class="font-mono text-sm font-semibold text-ink">{props.id}</h3>
        <Badge>{a().type}</Badge>
        <Badge tone="accent">{verdict()}</Badge>
        <Show when={a().type === "noul"}>
          <span class="text-xs text-faint">P(true) {pct(a().noul ?? 0)}</span>
        </Show>
      </div>
      <Show when={instructions()}>
        <p class="mt-1 text-xs leading-relaxed text-faint">{instructions()}</p>
      </Show>
      <div class="mt-3 space-y-1.5">
        <For each={sorted()}>
          {([key, p]) => (
            <div class="grid grid-cols-[minmax(0,14rem)_1fr_3.5rem] items-center gap-2 text-xs">
              <span class="truncate text-muted" title={key}>
                {label(key)}
              </span>
              <div class="h-2 overflow-hidden rounded-full bg-surface-3">
                <div class={`h-full rounded-full ${p === top() ? "bg-accent" : "bg-line-strong"}`} style={{ width: `${p * 100}%` }} />
              </div>
              <span class="text-right font-mono text-faint">{pct(p)}</span>
            </div>
          )}
        </For>
      </div>
    </div>
  );
}

export function Playground() {
  const [stateText, setStateText] = createSignal("");
  const [questionsText, setQuestionsText] = createSignal("");
  const [running, setRunning] = createSignal(false);
  const [error, setError] = createSignal<string | null>(null);
  const [answers, setAnswers] = createSignal<Record<string, Answer> | null>(null);
  const [asked, setAsked] = createSignal<Record<string, unknown>>({});
  const [ms, setMs] = createSignal(0);
  const [raw, setRaw] = createSignal(false);

  function loadPreset(p: Preset) {
    setStateText(JSON.stringify(p.state, null, 2));
    setQuestionsText(JSON.stringify(p.questions, null, 2));
    setAnswers(null);
    setError(null);
  }

  function clear() {
    setStateText("");
    setQuestionsText("");
    setAnswers(null);
    setError(null);
  }

  async function run() {
    setRunning(true);
    setError(null);
    // Let the button repaint before the synchronous forward pass blocks the thread.
    await new Promise((r) => setTimeout(r, 30));
    try {
      JSON.parse(stateText());
      const questions = JSON.parse(questionsText()) as Record<string, unknown>;
      const t = performance.now();
      const result = predict(stateText(), questionsText());
      setMs(performance.now() - t);
      setAsked(questions);
      setAnswers(result);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setRunning(false);
    }
  }

  return (
    <div class="mx-auto w-full max-w-5xl px-5 py-10">
      <p class="font-mono text-xs text-accent">100% client-side · runs in your browser via WebAssembly</p>
      <h1 class="mt-2 text-3xl font-semibold tracking-tight text-ink sm:text-4xl">Julia-1 playground</h1>
      <p class="mt-2 max-w-2xl leading-relaxed text-muted">
        Give Julia-1 a state (text or JSON) and typed questions, <span class="font-mono text-sm">choice</span>,{" "}
        <span class="font-mono text-sm">score</span> or <span class="font-mono text-sm">noul</span> (yes/no), and every
        question is answered in one bidirectional pass: no generation, nothing to parse. Powered by{" "}
        <a href="https://github.com/apiplant/julia1-rs" class="text-accent hover:text-accent-dim" target="_blank" rel="noreferrer noopener">
          julia1-rs
        </a>
        , the Rust runtime.
      </p>

      <div class="mt-8">
        <ModelPicker />
      </div>

      <hr class="my-6 border-line" />

      <div class="flex flex-wrap gap-2">
        <For each={PRESETS}>
          {(p) => (
            <Button variant="secondary" onClick={() => loadPreset(p)}>
              {p.label}
            </Button>
          )}
        </For>
        <Button variant="secondary" onClick={clear}>
          Clear
        </Button>
      </div>

      <div class="mt-4 grid gap-6 lg:grid-cols-2">
        <label class="block text-sm text-muted">
          State <span class="text-faint">(a JSON string, object or array)</span>
          <JsonEditor value={stateText()} onInput={setStateText} rows={14} />
        </label>
        <label class="block text-sm text-muted">
          Questions <span class="text-faint">(id → choice / score / noul)</span>
          <JsonEditor value={questionsText()} onInput={setQuestionsText} rows={14} />
        </label>
      </div>

      <div class="mt-4 flex flex-wrap items-center gap-3">
        <Button variant="primary" disabled={!loadedModel() || running() || !stateText().trim() || !questionsText().trim()} onClick={run}>
          {running() ? "Asking…" : "Ask"}
        </Button>
        <Show when={!loadedModel()}>
          <span class="text-xs text-faint">Load the model first.</span>
        </Show>
      </div>

      <Show when={error()}>
        <p class="mt-4 text-sm text-danger">{error()}</p>
      </Show>

      <Show when={answers()}>
        {(result) => (
          <div class="mt-6">
            <div class="flex flex-wrap items-center gap-3">
              <h2 class="text-base font-semibold text-ink">Answers</h2>
              <Badge tone="accent">{ms().toFixed(0)} ms</Badge>
              <button type="button" onClick={() => setRaw(!raw())} class="ml-auto text-xs text-muted underline decoration-dotted hover:text-ink">
                {raw() ? "show cards" : "show JSON"}
              </button>
            </div>
            <Show
              when={!raw()}
              fallback={
                <div class="mt-3">
                  <JsonView value={{ answers: result() }} />
                </div>
              }
            >
              <div class="mt-3 grid gap-3 md:grid-cols-2">
                <For each={Object.entries(result())}>
                  {([id, answer]) => <AnswerCard id={id} answer={answer} question={asked()[id]} />}
                </For>
              </div>
            </Show>
          </div>
        )}
      </Show>
    </div>
  );
}
