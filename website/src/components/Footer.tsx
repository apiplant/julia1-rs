import { GITHUB_URL } from "../lib/links";

export function Footer() {
  return (
    <footer class="border-t border-line">
      <div class="mx-auto flex w-full max-w-6xl flex-wrap items-center justify-between gap-2 px-5 py-5 text-xs text-faint">
        <span>
          © {new Date().getFullYear()} julia1-rs · Apache-2.0 · Julia-1 is by{" "}
          <a href="https://huggingface.co/SupersonicLabs/Julia-1" class="hover:text-ink" target="_blank" rel="noreferrer noopener">
            Supersonic Labs
          </a>
        </span>
        <span>
          Install the CLI and library: <a href={`${GITHUB_URL}#install`} class="text-muted hover:text-ink" target="_blank" rel="noreferrer noopener">github.com/apiplant/julia1-rs</a>
        </span>
      </div>
    </footer>
  );
}
