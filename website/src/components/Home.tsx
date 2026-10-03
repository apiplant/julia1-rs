import { For } from "solid-js";
import { Badge, LinkButton, Mono } from "./ui";
import { CopyBlock } from "./Code";
import { Pre } from "./docs/Prose";
import { highlight } from "../lib/highlight";
import { GITHUB_URL } from "../lib/links";
import { CUDA_PLATFORM, PLATFORMS, LATEST_RELEASE_URL, assetName, downloadUrl } from "../lib/release";

/* ------------------------------------------------------------------ */
/* A fake terminal panel showing a real julia1 invocation and output.  */
/* ------------------------------------------------------------------ */

function TerminalDemo() {
  const command = "julia1 predict < ticket.jsonl";
  const input = `{"state": "I was charged twice for the same order.",
 "questions": {"team": {"type": "choice",
   "instructions": "Which team should handle this request?",
   "criteria": {"billing": "Billing and payment disputes",
                "shipping": "Shipping and delivery",
                "access": "Account access and login"}}}}`;
  const output = `{"answers": {"team": {
    "type": "choice",
    "choice": "billing",
    "max_probability": 0.8545,
    "probabilities": {"billing": 0.8545, "shipping": 0.1428, "access": 0.0028}}}}`;

  return (
    <div class="min-w-0 overflow-hidden rounded-xl border border-line shadow-2xl">
      <div class="flex items-center gap-2 border-b border-line bg-surface px-4 py-2.5">
        <span class="h-2.5 w-2.5 rounded-full bg-danger" />
        <span class="h-2.5 w-2.5 rounded-full bg-warn" />
        <span class="h-2.5 w-2.5 rounded-full bg-success" />
        <span class="ml-2 font-mono text-xs text-faint">julia1</span>
      </div>
      <pre class="overflow-x-auto bg-code-bg px-4 py-4 font-mono text-[0.78rem] leading-relaxed">
        <code class="language-json">
          <span class="select-none text-faint"># ticket.jsonl (one request per line)</span>
          {"\n"}
          <span innerHTML={highlight(input, "json")} />
        </code>
        {"\n\n"}
        <code class="language-bash">
          <span class="select-none text-faint">$ </span>
          <span innerHTML={highlight(command, "bash")} />
        </code>
        {"\n"}
        <code class="language-json">
          <span innerHTML={highlight(output, "json")} />
        </code>
      </pre>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Headline numbers (all from the README's measured tables).           */
/* ------------------------------------------------------------------ */

const STATS = [
  { value: "0.9 ms", label: "per request on an RTX 4090", note: "6.85× faster than the Python runtime" },
  { value: "27.6 ms", label: "per request on 4 CPU threads", note: "1.34× faster, AVX-512 GEMM" },
  { value: "2000/2000", label: "answers match Python on CPU", note: "identical token ids on every row" },
  { value: "7.5k tokens", label: "in 20.5 ms on CUDA", note: "5.56× faster on long requests" },
];

function Stats() {
  return (
    <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
      <For each={STATS}>
        {(s) => (
          <div class="rounded-xl border border-line bg-surface p-5">
            <p class="font-mono text-2xl font-semibold tracking-tight text-accent">{s.value}</p>
            <p class="mt-2 text-sm font-medium text-ink">{s.label}</p>
            <p class="mt-1 text-xs leading-relaxed text-faint">{s.note}</p>
          </div>
        )}
      </For>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The three question types.                                           */
/* ------------------------------------------------------------------ */

const QUESTION_TYPES = [
  {
    type: "choice",
    title: "Pick one",
    body: "Route a ticket, label an email, choose an action. Each option gets a probability; they sum to 1.",
    code: `"team": {
  "type": "choice",
  "instructions": "Who handles this?",
  "criteria": {
    "billing": "Payments, refunds",
    "access": "Login problems"
  }
}`,
  },
  {
    type: "score",
    title: "Rate on a rubric",
    body: "An ordered rubric, scored as the expected index, so 1.4 means between the second and third level.",
    code: `"severity": {
  "type": "score",
  "instructions": "How bad is it?",
  "criteria": ["low", "medium", "high"]
}`,
  },
  {
    type: "noul",
    title: "Yes or no",
    body: "A single probability that the statement is true. Criteria for true and false are optional.",
    code: `"approved": {
  "type": "noul",
  "instructions": "Is this a refund?"
}`,
  },
];

function QuestionTypes() {
  return (
    <div class="grid gap-4 lg:grid-cols-3">
      <For each={QUESTION_TYPES}>
        {(q) => (
          <div class="flex min-w-0 flex-col rounded-xl border border-line bg-surface p-5">
            <div class="flex items-center gap-2">
              <Badge tone="accent">{q.type}</Badge>
              <h3 class="text-[0.9375rem] font-semibold tracking-tight text-ink">{q.title}</h3>
            </div>
            <p class="mt-2 text-sm leading-relaxed text-muted">{q.body}</p>
            <pre class="mt-4 overflow-x-auto rounded-lg border border-line bg-code-bg px-3 py-3 font-mono text-[0.75rem] leading-relaxed">
              <code innerHTML={highlight(q.code, "json")} />
            </pre>
          </div>
        )}
      </For>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Features.                                                          */
/* ------------------------------------------------------------------ */

const FEATURES = [
  {
    title: "One pass, no generation",
    body: "The whole request, state and every question, goes through a bidirectional encoder once. There is no text to sample, parse or hallucinate.",
  },
  {
    title: "AVX-512 CPU backend",
    body: "FP32 GEMM over pre-packed weights with fused bias, ReLU, residual and GEGLU epilogues, reaching about 92–95% of single-core peak. A portable fallback covers other CPUs.",
  },
  {
    title: "CUDA with BF16 tensor cores",
    body: "cuBLAS GEMMs with FP32 accumulation and a FlashAttention-2-style kernel with in-kernel RoPE and sliding window. Opt in with the cuda feature.",
  },
  {
    title: "No padding",
    body: "Sequences are packed back to back, so every GEMM and attention tile runs on real tokens only. Sliding-window layers compute just their ±64 band.",
  },
  {
    title: "Matches the Python runtime",
    body: "Token ids are identical on every test row, including JSON states, which required reproducing Python's json.dumps byte for byte. Same argmax on all 2,000 test questions.",
  },
  {
    title: "Strict encoding",
    body: "Marker injection and any truncation are rejected up front instead of silently changing the answer, with the same validation errors as the original.",
  },
  {
    title: "Long and wide requests",
    body: "Contexts up to 8,192 tokens, and a hierarchical router for questions with more than 20 options.",
  },
  {
    title: "Runs in your browser",
    body: "The same engine compiles to WebAssembly. The demo loads the checkpoint into your tab and answers locally: nothing is uploaded.",
  },
  {
    title: "HTTP server built in",
    body: "julia1 serve exposes the named-question API with a serial execution queue, 429 backpressure and graceful shutdown.",
  },
];

function Features() {
  return (
    <div class="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
      <For each={FEATURES}>
        {(f) => (
          <div class="rounded-xl border border-line bg-surface p-5">
            <h3 class="text-[0.9375rem] font-semibold tracking-tight text-ink">{f.title}</h3>
            <p class="mt-2 text-sm leading-relaxed text-muted">{f.body}</p>
          </div>
        )}
      </For>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Benchmarks.                                                        */
/* ------------------------------------------------------------------ */

const BENCH = [
  { config: "CUDA", metric: "single request", python: "6.4 ms", rust: "0.9 ms", speedup: "6.85×" },
  { config: "CUDA", metric: "2,000 rows, batch 16", python: "1,434 rows/s", rust: "3,202 rows/s", speedup: "2.23×" },
  { config: "CUDA", metric: "7,556-token request", python: "114 ms", rust: "20.5 ms", speedup: "5.56×" },
  { config: "CPU · 4 threads", metric: "single request", python: "37.0 ms", rust: "27.6 ms", speedup: "1.34×" },
  { config: "CPU · 4 threads", metric: "2,000 rows, batch 16", python: "14.6 rows/s", rust: "19.7 rows/s", speedup: "1.35×" },
  { config: "CPU · 4 threads", metric: "7,556-token request", python: "6,742 ms", rust: "3,042 ms", speedup: "2.22×" },
  { config: "CPU · 16 threads", metric: "single request", python: "26.3 ms", rust: "19.4 ms", speedup: "1.35×" },
  { config: "CPU · 16 threads", metric: "7,556-token request", python: "2,852 ms", rust: "1,080 ms", speedup: "2.64×" },
];

function BenchTable() {
  return (
    <div class="mt-8 overflow-hidden overflow-x-auto rounded-xl border border-line">
      <table class="w-full min-w-[34rem] border-collapse text-left text-sm">
        <thead>
          <tr class="bg-surface text-xs uppercase tracking-[0.1em] text-faint">
            <th class="border-b border-line px-4 py-3 font-semibold">Config</th>
            <th class="border-b border-line px-4 py-3 font-semibold">Metric</th>
            <th class="border-b border-line px-4 py-3 text-right font-semibold">Python</th>
            <th class="border-b border-line px-4 py-3 text-right font-semibold">julia1-rs</th>
            <th class="border-b border-line px-4 py-3 text-right font-semibold">Speedup</th>
          </tr>
        </thead>
        <tbody>
          <For each={BENCH}>
            {(row, i) => (
              <tr class={i() % 2 === 0 ? "bg-canvas" : "bg-surface"}>
                <td class="whitespace-nowrap border-b border-line px-4 py-2.5 text-muted">{row.config}</td>
                <td class="border-b border-line px-4 py-2.5 text-muted">{row.metric}</td>
                <td class="border-b border-line px-4 py-2.5 text-right font-mono text-xs text-faint">{row.python}</td>
                <td class="border-b border-line px-4 py-2.5 text-right font-mono text-xs text-ink">{row.rust}</td>
                <td class="border-b border-line px-4 py-2.5 text-right font-mono text-xs font-semibold text-accent">
                  {row.speedup}
                </td>
              </tr>
            )}
          </For>
        </tbody>
      </table>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Install: numbered step cards, the apiplant layout.                  */
/* ------------------------------------------------------------------ */

const stepCard =
  "grid min-w-0 gap-5 rounded-2xl border bg-surface p-5 sm:p-6 lg:grid-cols-[minmax(14rem,0.7fr)_minmax(0,1.3fr)] lg:items-start";

function InstallSteps() {
  const homebrewCommands = `brew tap apiplant/tap
brew install apiplant/tap/julia1-rs`;
  const pacmanCommands = `curl -sSfL https://apiplant.github.io/pacman/apiplant.gpg -o /tmp/apiplant.gpg
keyid=$(gpg --show-keys --with-colons /tmp/apiplant.gpg | awk -F: '/^pub:/ { print $5; exit }') && sudo pacman-key --add /tmp/apiplant.gpg && sudo pacman-key --finger "$keyid" && sudo pacman-key --lsign-key "$keyid"
printf '\\n[apiplant]\\nSigLevel = Required DatabaseOptional\\nServer = https://apiplant.github.io/pacman/$arch\\n' | sudo tee -a /etc/pacman.conf > /dev/null
sudo pacman -Sy julia1-rs`;
  const aptCommands = `curl -sSfL https://apt.apiplant.com/apiplant-archive-keyring.gpg | sudo tee /usr/share/keyrings/apiplant.gpg > /dev/null
echo "deb [signed-by=/usr/share/keyrings/apiplant.gpg] https://apt.apiplant.com stable main" | sudo tee /etc/apt/sources.list.d/apiplant.list > /dev/null
sudo apt update && sudo apt install julia1-rs`;
  const cargoCommands = `cargo add julia1           # as a library dependency
cargo install julia1       # the julia1 binary
cargo install julia1 --features cuda   # with CUDA (needs nvcc)`;

  return (
    <div class="mt-8 space-y-4 sm:mt-10">
      <div class={`${stepCard} border-accent-line`}>
        <div>
          <div class="flex items-center gap-2">
            <span class="font-mono text-xs text-accent">01</span>
            <Badge tone="accent">Recommended</Badge>
          </div>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Use a package manager</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">
            macOS (Apple Silicon), Arch Linux and Debian/Ubuntu are all published to the apiplant
            shared repositories. The CUDA build for Linux x86_64 is a separate package,{" "}
            <Mono>julia1-rs-cuda</Mono>, that conflicts with <Mono>julia1-rs</Mono>.
          </p>
        </div>

        <div class="min-w-0 space-y-5">
          <div>
            <p class="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-faint">Homebrew</p>
            <CopyBlock command={homebrewCommands} />
          </div>

          <div>
            <p class="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-faint">
              Arch Linux / pacman
            </p>
            <CopyBlock command={pacmanCommands} />
          </div>

          <div>
            <p class="mb-2 text-xs font-semibold uppercase tracking-[0.14em] text-faint">
              Debian / Ubuntu
            </p>
            <CopyBlock command={aptCommands} />
          </div>
        </div>
      </div>

      <div class={`${stepCard} border-line`}>
        <div>
          <span class="font-mono text-xs text-accent">02</span>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Download the archive</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">
            One archive per platform, holding the <Mono>julia1</Mono> binary and the README. No
            installation needed; unpack and run. The first run downloads Julia-1 (about 577 MB)
            into <Mono>~/.cache/julia1-rs</Mono>.
          </p>
        </div>

        <ul class="min-w-0 space-y-1 border-t border-line pt-4 lg:border-t-0 lg:pt-0">
          <For each={[...PLATFORMS, CUDA_PLATFORM]}>
            {(platform) => (
              <li class="min-w-0">
                <a
                  href={downloadUrl(platform)}
                  title={assetName(platform)}
                  class="flex min-w-0 items-baseline justify-between gap-3 rounded-md py-1 text-muted transition-colors hover:text-ink"
                >
                  <span class="shrink-0 text-sm">{platform.label}</span>
                  <span class="min-w-0 truncate font-mono text-xs text-accent">{assetName(platform)}</span>
                </a>
              </li>
            )}
          </For>
        </ul>

        <a
          href={LATEST_RELEASE_URL}
          target="_blank"
          rel="noreferrer noopener"
          class="text-sm font-medium text-accent hover:text-accent-dim lg:col-start-2"
        >
          All releases and checksums
        </a>
      </div>

      <div class={`${stepCard} border-line`}>
        <div>
          <span class="font-mono text-xs text-faint">03</span>
          <h3 class="mt-3 text-base font-semibold tracking-tight text-ink">Cargo</h3>
          <p class="mt-2 text-sm leading-relaxed text-muted">
            As a library dependency, or to build the CLI from source via crates.io. CUDA is never
            on by default.
          </p>
        </div>

        <div class="min-w-0">
          <CopyBlock command={cargoCommands} />
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Page.                                                              */
/* ------------------------------------------------------------------ */

export function Home() {
  return (
    <div class="mx-auto w-full max-w-6xl px-5">
      {/* Hero */}
      <section class="grid items-center gap-10 py-16 sm:py-20 lg:grid-cols-2 lg:gap-12">
        <div class="min-w-0">
          <div class="flex flex-wrap items-center gap-2">
            <Badge tone="accent">v{__VERSION__}</Badge>
            <Badge>Rust</Badge>
            <Badge>CPU + CUDA</Badge>
            <Badge>Apache-2.0</Badge>
          </div>
          <h1 class="mt-5 text-4xl font-semibold tracking-tight text-ink sm:text-5xl">
            Julia-1 decisions, <span class="text-accent">in one forward pass</span>.
          </h1>
          <p class="mt-4 max-w-lg text-lg leading-relaxed text-muted">
            Give it a state, text or JSON, and typed <Mono>choice</Mono>/<Mono>score</Mono>/
            <Mono>noul</Mono> questions. Get calibrated probabilities back from a single pass of
            the model: nothing to generate, nothing to parse. Up to 6.85× faster than the Python
            runtime.
          </p>
          <div class="mt-7">
            <LinkButton
              href="/demo"
              variant="primary"
              class="!px-5 !py-4 !text-base shadow-lg shadow-accent/20 sm:!px-8 sm:!text-lg"
            >
              ▶ Try it now, in your browser
            </LinkButton>
          </div>
          <div class="mt-4 flex flex-wrap gap-3">
            <LinkButton href={GITHUB_URL} size="lg">
              View on GitHub
            </LinkButton>
            <LinkButton href="/#install" size="lg">
              Install
            </LinkButton>
            <LinkButton href="/#serve" size="lg">
              HTTP server
            </LinkButton>
          </div>
          <p class="mt-5 text-sm text-faint">
            A Rust port of the inference runtime for{" "}
            <a
              href="https://huggingface.co/SupersonicLabs/Julia-1"
              target="_blank"
              rel="noreferrer noopener"
              class="text-accent hover:text-accent-dim"
            >
              Supersonic Labs' Julia-1
            </a>
            , reading the unmodified checkpoint.
          </p>
        </div>
        <TerminalDemo />
      </section>

      {/* Numbers */}
      <section class="pb-16">
        <Stats />
      </section>

      {/* Question types */}
      <section id="questions" class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink">Three kinds of question</h2>
        <p class="mt-2 max-w-2xl text-muted">
          Ask as many as you like about the same state. They are answered together, keyed by the
          ids you choose.
        </p>
        <div class="mt-8">
          <QuestionTypes />
        </div>
      </section>

      {/* Features */}
      <section id="features" class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink">Built for speed, checked for parity</h2>
        <p class="mt-2 max-w-2xl text-muted">
          Hand-written CPU and CUDA backends, validated against the Python reference down to the
          token ids.
        </p>
        <div class="mt-8">
          <Features />
        </div>
      </section>

      {/* Benchmarks */}
      <section id="benchmarks" class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink">Benchmarks</h2>
        <p class="mt-2 max-w-2xl text-muted">
          Python (torch 2.12 + transformers 5.0, the unmodified{" "}
          <span class="rounded bg-surface-2 px-1.5 py-0.5 font-mono text-[0.8125rem] text-ink">FastEngine</span>) and Rust,
          run back to back on the same idle machine with the same rows: an AMD Ryzen 9 7950X and an
          RTX 4090. Thread counts are equal on both sides.
        </p>
        <BenchTable />
        <p class="mt-3 text-xs text-faint">
          Single request is about 160 tokens including tokenization; batch is 2,000 typed rows at
          roughly 293 tokens each. The full table and method are in the{" "}
          <a href={`${GITHUB_URL}#benchmarks`} target="_blank" rel="noreferrer noopener" class="text-accent hover:text-accent-dim">
            README
          </a>
          .
        </p>
      </section>

      {/* Install */}
      <section id="install" class="pb-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">Install</h2>
        <p class="mt-3 max-w-2xl leading-relaxed text-muted">
          Use Homebrew, pacman or apt when your platform has it. Otherwise take the prebuilt
          archive, or pull the crate from crates.io.
        </p>
        <InstallSteps />
      </section>

      {/* Serve */}
      <section id="serve" class="border-t border-line pb-16 pt-16">
        <div class="flex flex-wrap items-center justify-between gap-2">
          <h2 class="text-2xl font-semibold tracking-tight text-ink">Serve it over HTTP</h2>
          <LinkButton href={`${GITHUB_URL}#serve-over-http`} size="sm">
            Server docs →
          </LinkButton>
        </div>
        <p class="mt-3 max-w-2xl leading-relaxed text-muted">
          <Mono>julia1 serve</Mono> loads the checkpoint once and answers named-question requests.
          Requests run one at a time behind a bounded queue, so a burst gets a clean{" "}
          <Mono>429</Mono> with <Mono>Retry-After</Mono> instead of piling up. Ctrl-C finishes the
          in-flight request before exiting.
        </p>
        <div class="mt-6">
          <CopyBlock command={"julia1 serve                                   # 127.0.0.1:8000\njulia1 serve --device cuda --port 9000 --host 0.0.0.0"} />
        </div>
        <div class="mt-6 grid gap-4 lg:grid-cols-2">
          <Pre caption="POST /v1/classifier" lang="json">{`{
  "state": "I was charged twice for March.",
  "questions": {
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "Billing and payment disputes",
        "shipping": "Shipping and delivery"
      }
    }
  }
}`}</Pre>
          <Pre caption="200 response" lang="json">{`{
  "model": "julia-1",
  "answers": {
    "team": {
      "type": "choice",
      "choice": "billing",
      "max_probability": 0.7507,
      "probabilities": { "billing": 0.7507, "shipping": 0.2493 }
    }
  },
  "usage": { "input_tokens": 30, "output_tokens": 0 }
}`}</Pre>
        </div>
        <ul class="mt-6 grid gap-x-8 gap-y-2 text-sm text-muted sm:grid-cols-2">
          <li>
            <Mono>POST /v1/classifier</Mono> and its alias <Mono>/v1/systemone</Mono>
          </li>
          <li>
            <Mono>GET /health</Mono> reports readiness without running the model
          </li>
          <li>
            <Mono>422</Mono> for invalid questions, strict-encoding rejections or a body over 1 MiB
          </li>
          <li>
            <Mono>500</Mono> if a forward fails; the server keeps serving
          </li>
        </ul>
      </section>

      {/* Library */}
      <section id="library" class="border-t border-line pb-16 pt-16">
        <h2 class="text-2xl font-semibold tracking-tight text-ink">As a library</h2>
        <p class="mt-3 max-w-2xl leading-relaxed text-muted">
          One <Mono>Engine</Mono>, one <Mono>predict_typed</Mono> call. Any number of questions,
          scored against one state together. The engine can download Julia-1 for you on first use.
        </p>
        <div class="mt-6">
          <CopyBlock command="cargo add julia1" />
        </div>
        <Pre caption="src/main.rs" lang="rust">{`use julia1::{Engine, EngineOptions, State};

// Downloads Julia-1 into ~/.cache/julia1-rs on first use (about 577 MB);
// or Engine::load("path/to/Julia-1", options) for a local copy.
let engine = Engine::from_pretrained(EngineOptions {
    strict_encoding: true,
    head_length: 512,
    ..Default::default()
})?;
let questions = serde_json::json!({"team": {"type": "choice",
    "instructions": "Which team should handle this request?",
    "criteria": {"billing": "Billing and payment disputes",
                 "shipping": "Shipping and delivery"}}});
let answers = engine.predict_typed(&State::from("I was charged twice."), &questions)?;
println!("{}", answers[0].1.choice().unwrap());`}</Pre>
      </section>

      {/* Closing call to action */}
      <section class="border-t border-line pb-20 pt-16 text-center">
        <h2 class="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">See it answer, locally</h2>
        <p class="mx-auto mt-3 max-w-xl leading-relaxed text-muted">
          The demo runs the same engine as WebAssembly in your browser. Load the model once, then
          ask your own questions about your own text.
        </p>
        <div class="mt-7 flex justify-center">
          <LinkButton href="/demo" variant="primary" size="lg">
            ▶ Open the demo
          </LinkButton>
        </div>
      </section>
    </div>
  );
}
