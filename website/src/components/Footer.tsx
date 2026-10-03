import { For } from "solid-js";
import { ChipMark } from "./ui";
import { COMPANY_URL, GITHUB_URL } from "../lib/links";

const COLUMNS = [
  {
    title: "Product",
    links: [
      { label: "Demo", href: "/demo" },
      { label: "Features", href: "/#features" },
      { label: "Benchmarks", href: "/#benchmarks" },
      { label: "Install", href: "/#install" },
    ],
  },
  {
    title: "Use it",
    links: [
      { label: "HTTP server", href: "/#serve" },
      { label: "Rust library", href: "/#library" },
      { label: "CLI on GitHub", href: `${GITHUB_URL}#use` },
    ],
  },
  {
    title: "Source",
    links: [
      { label: "GitHub", href: GITHUB_URL },
      { label: "Issues", href: `${GITHUB_URL}/issues` },
      { label: "Releases", href: `${GITHUB_URL}/releases` },
      { label: "crates.io", href: "https://crates.io/crates/julia1" },
      { label: "Julia-1 model", href: "https://huggingface.co/SupersonicLabs/Julia-1" },
    ],
  },
];

function FooterLink(props: { href: string; label: string }) {
  return props.href.startsWith("/") ? (
    <a href={props.href} class="text-muted transition-colors hover:text-ink">
      {props.label}
    </a>
  ) : (
    <a
      href={props.href}
      target="_blank"
      rel="noreferrer noopener"
      class="text-muted transition-colors hover:text-ink"
    >
      {props.label}
    </a>
  );
}

export function Footer() {
  return (
    <footer class="border-t border-line">
      <div class="mx-auto grid w-full max-w-6xl gap-10 px-5 py-12 sm:grid-cols-2 lg:grid-cols-4">
        <div>
          <div class="flex items-center gap-2">
            <ChipMark class="h-7 w-7" />
            <span class="text-[0.9375rem] font-semibold tracking-tight text-ink">
              julia1<span class="text-accent">-rs</span>
            </span>
          </div>
          <p class="mt-3 max-w-xs text-sm leading-relaxed text-faint">
            Rust runtime for Supersonic Labs' Julia-1 decision model: typed choice/score/noul
            questions answered in one forward pass, on CPU or CUDA, in the browser or over HTTP.
          </p>
        </div>

        <For each={COLUMNS}>
          {(column) => (
            <div>
              <h2 class="text-[0.6875rem] font-semibold uppercase tracking-[0.12em] text-faint">
                {column.title}
              </h2>
              <ul class="mt-3 grid gap-2 text-sm">
                <For each={column.links}>
                  {(link) => (
                    <li>
                      <FooterLink href={link.href} label={link.label} />
                    </li>
                  )}
                </For>
              </ul>
            </div>
          )}
        </For>
      </div>

      <div class="border-t border-line">
        <div class="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-2 px-5 py-5 text-xs text-faint">
          <span>
            © {new Date().getFullYear()} julia1-rs · Apache-2.0 · Julia-1 is by{" "}
            <FooterLink href="https://huggingface.co/SupersonicLabs/Julia-1" label="Supersonic Labs" />
          </span>
          <span>
            <FooterLink href={COMPANY_URL} label="API Plant" />
          </span>
        </div>
      </div>
    </footer>
  );
}
