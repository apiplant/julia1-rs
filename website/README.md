# julia1-rs website

The website for [julia1-rs](https://github.com/apiplant/julia1-rs). Solid 2 RC + Tailwind v4 + Vite, static build, deployed to Cloudflare.

```bash
pnpm install
pnpm dev       # local dev server
pnpm build     # -> dist/
pnpm check     # types only
pnpm build:wasm  # regenerate src/wasm-pkg/ (needs Rust + wasm-pack)
```

A single page: the playground runs julia1-rs itself in the browser, so the site depends on `src/wasm-pkg/`
(the `wasm-pack` output for the crate one directory up), which is **committed** so a deploy needs only Node.
