import { type ParentProps } from "solid-js";
import { createRouter } from "@solidjs/router";
import { Header } from "./components/Header";
import { Footer } from "./components/Footer";
import { Playground } from "./components/demo/Playground";

function Shell(props: ParentProps) {
  return (
    <div class="flex min-h-screen flex-col">
      <Header />
      <main class="flex-1">{props.children}</main>
      <Footer />
    </div>
  );
}

// A single page: every path shows the playground.
const Router = createRouter({
  routes: [{ path: "*all", component: Playground }],
});

export function App() {
  return <Router>{(props) => <Shell>{props.children}</Shell>}</Router>;
}
