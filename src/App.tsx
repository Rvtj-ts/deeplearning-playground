import { useState } from "react";
import { CNNViz } from "./components/CNNViz";
import { GradientDescentViz } from "./components/GradientDescentViz";
import { LLMFlowViz } from "./components/LLMFlowViz";
import { LLMViz } from "./components/LLMViz";
import { PCAViz } from "./components/PCAViz";
import { RNNViz } from "./components/RNNViz";
import { SVDViz } from "./components/SVDViz";

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

export default function App() {
  const [active, setActive] = useState<ConceptId>("svd");

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
            onClick={() => setActive(concept.id)}
          >
            {concept.label}
          </button>
        ))}
      </nav>

      <main className="panel">
        {active === "svd" && <SVDViz />}
        {active === "pca" && <PCAViz />}
        {active === "gd" && <GradientDescentViz />}
        {active === "cnn" && <CNNViz />}
        {active === "rnn" && <RNNViz />}
        {active === "llm" && <LLMViz />}
        {active === "llmflow" && <LLMFlowViz />}
      </main>
    </div>
  );
}
