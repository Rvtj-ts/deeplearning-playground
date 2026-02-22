import { useState } from "react";
import { CNNViz } from "./components/CNNViz";
import { GradientDescentViz } from "./components/GradientDescentViz";
import { PCAViz } from "./components/PCAViz";
import { SVDViz } from "./components/SVDViz";

const concepts = [
  { id: "svd", label: "SVD" },
  { id: "pca", label: "PCA" },
  { id: "gd", label: "Gradient Descent" },
  { id: "cnn", label: "CNN" },
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
          optimization, and CNN feature learning.
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
      </main>
    </div>
  );
}
