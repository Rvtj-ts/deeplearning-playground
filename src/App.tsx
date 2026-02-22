import { Component, type ComponentType, type ErrorInfo, type ReactNode, Suspense, lazy, useEffect, useState } from "react";
import { CNNViz } from "./components/CNNViz";
import { GradientDescentViz } from "./components/GradientDescentViz";
import { LLMFlowViz } from "./components/LLMFlowViz";
import { LLMViz } from "./components/LLMViz";
import { RNNViz } from "./components/RNNViz";
import { SVDViz } from "./components/SVDViz";

function lazyWithRetry<T extends { default: ComponentType<any> }>(
  importer: () => Promise<T>,
) {
  return lazy(async () => {
    try {
      return await importer();
    } catch {
      await new Promise((resolve) => window.setTimeout(resolve, 350));
      return importer();
    }
  });
}

const PCAViz = lazyWithRetry(() =>
  import("./components/PCAViz").then((module) => ({ default: module.PCAViz })),
);

class LazyChunkBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean }
> {
  state = { hasError: false };

  static getDerivedStateFromError() {
    return { hasError: true };
  }

  componentDidCatch(_error: Error, _errorInfo: ErrorInfo) {}

  render() {
    if (this.state.hasError) {
      return (
        <div className="formula-block">
          Could not load this module chunk.
          <br />
          Please refresh once after deploy.
        </div>
      );
    }
    return this.props.children;
  }
}

const concepts = [
  { id: "svd", label: "SVD" },
  { id: "pca", label: "PCA" },
  { id: "gd", label: "Gradient Descent" },
  { id: "cnn", label: "CNN" },
  { id: "rnn", label: "RNN" },
  { id: "llm", label: "LLM" },
  { id: "llmflow", label: "LLM Flow" },
] as const;

type ConceptId = (typeof concepts)[number]["id"];

const DEFAULT_TAB: ConceptId = "svd";

function isConceptId(value: string | null): value is ConceptId {
  if (!value) return false;
  return concepts.some((c) => c.id === value);
}

function getTabFromUrl(): ConceptId {
  if (typeof window === "undefined") return DEFAULT_TAB;
  const value = new URLSearchParams(window.location.search).get("tab");
  return isConceptId(value) ? value : DEFAULT_TAB;
}

function updateUrlTab(tab: ConceptId, replace: boolean) {
  const url = new URL(window.location.href);
  url.searchParams.set("tab", tab);
  const next = `${url.pathname}${url.search}${url.hash}`;
  if (replace) {
    window.history.replaceState(null, "", next);
  } else {
    window.history.pushState(null, "", next);
  }
}

export default function App() {
  const [active, setActive] = useState<ConceptId>(() => getTabFromUrl());

  useEffect(() => {
    const raw = new URLSearchParams(window.location.search).get("tab");
    if (!isConceptId(raw)) {
      updateUrlTab(active, true);
    }
  }, [active]);

  useEffect(() => {
    const onPopState = () => {
      setActive(getTabFromUrl());
    };
    window.addEventListener("popstate", onPopState);
    return () => window.removeEventListener("popstate", onPopState);
  }, []);

  const changeTab = (tab: ConceptId) => {
    if (tab === active) return;
    setActive(tab);
    updateUrlTab(tab, false);
  };

  return (
    <div className="page">
      <header className="hero">
        <h1>Deep Learning Concept Visualizer</h1>
        <p>
          Interactive intuition builders for your presentation: matrix
          factorization, dimensionality reduction on handwritten digit data, and
          optimization, plus CNN, RNN, and LLM attention.
        </p>
      </header>

      <nav className="tabs" aria-label="Concepts">
        {concepts.map((concept) => (
          <button
            key={concept.id}
            className={concept.id === active ? "tab tab-active" : "tab"}
            onClick={() => changeTab(concept.id)}
          >
            {concept.label}
          </button>
        ))}
      </nav>

      <main className="panel">
        {active === "svd" && <SVDViz />}
        {active === "pca" && (
          <LazyChunkBoundary>
            <Suspense fallback={<p className="subtext">Loading PCA module...</p>}>
              <PCAViz />
            </Suspense>
          </LazyChunkBoundary>
        )}
        {active === "gd" && <GradientDescentViz />}
        {active === "cnn" && <CNNViz />}
        {active === "rnn" && <RNNViz />}
        {active === "llm" && <LLMViz />}
        {active === "llmflow" && <LLMFlowViz />}
      </main>
    </div>
  );
}
