import { useEffect, useState } from "react";

const STEPS = [
  {
    title: "1) User Prompt",
    text: "Raw text enters the model context window.",
    detail: "Example: 'Explain PCA in simple terms'.",
    explain: "The model starts with plain text, not meaning. It must first convert text to internal tokens.",
  },
  {
    title: "2) Tokenization",
    text: "Text is split into model tokens/subwords.",
    detail: "Words become IDs the model can process numerically.",
    explain: "A word can be one token or multiple pieces. The model predicts the next token piece-by-piece.",
  },
  {
    title: "3) Embedding + Position",
    text: "Each token ID maps to a vector; position is added.",
    detail: "Now each token is a dense numeric representation.",
    explain: "Same word in different positions gets a different final representation due to position encoding.",
  },
  {
    title: "4) Transformer Blocks",
    text: "Repeated attention + feed-forward layers transform context.",
    detail: "Causal mask prevents looking at future tokens.",
    explain: "Attention decides which earlier tokens matter for each current token. Feed-forward refines that info.",
  },
  {
    title: "5) Logits",
    text: "Final hidden state projects to vocabulary scores.",
    detail: "One score per candidate next token.",
    explain: "Higher score means the model currently prefers that token more strongly.",
  },
  {
    title: "6) Sampling",
    text: "Softmax + decoding picks next token.",
    detail: "Greedy/top-k/top-p/temperature affect creativity.",
    explain: "Low temperature is safer and more deterministic; high temperature is more diverse but riskier.",
  },
  {
    title: "7) Append + Repeat",
    text: "Chosen token is appended, process runs again.",
    detail: "This autoregressive loop generates the response.",
    explain: "LLMs generate one token at a time until stop condition or max length is reached.",
  },
] as const;

export function LLMFlowViz() {
  const [active, setActive] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1100);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setActive((s) => (s + 1) % STEPS.length);
    }, speed);
    return () => window.clearInterval(timer);
  }, [playing, speed]);

  return (
    <section>
      <h2>LLM End-to-End Flow</h2>
      <p className="subtext">
        This is the full lifecycle for one generated token. During generation, steps 4 to 7 repeat many times.
      </p>

      <div className="explain-card">
        <strong>How to present this</strong>
        <span>Read left-to-right: text in, token out.</span>
        <span>Highlight that only one token is chosen each loop.</span>
        <span>Looping this process is what creates full sentences.</span>
        <span>Use the active stage panel below to explain each step in plain English.</span>
      </div>

      <div className="flow-controls">
        <button className="ghost-btn" onClick={() => setPlaying((v) => !v)}>
          {playing ? "Pause" : "Play"}
        </button>
        <button className="ghost-btn" onClick={() => setActive((s) => Math.max(0, s - 1))}>
          Prev
        </button>
        <button className="ghost-btn" onClick={() => setActive((s) => Math.min(STEPS.length - 1, s + 1))}>
          Next
        </button>
        <label>
          Speed: {speed} ms
          <input type="range" min={500} max={2000} step={50} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} />
        </label>
      </div>

      <div className="llm-flow-grid">
        {STEPS.map((step, idx) => (
          <div key={step.title} className={idx === active ? "llm-flow-step llm-flow-step-active" : "llm-flow-step"}>
            <h3>{step.title}</h3>
            <p>{step.text}</p>
            <small>{step.detail}</small>
            {idx < STEPS.length - 1 && <span className="llm-flow-arrow">{"->"}</span>}
          </div>
        ))}
      </div>

      <div className="formula-block">
        Current stage: {STEPS[active].title}
        <br />
        Why this matters: {STEPS[active].detail}
        <br />
        Plain-English explanation: {STEPS[active].explain}
      </div>

      <div className="llm-flow-detail">
        <strong>Concrete example at this step</strong>
        <span>Prompt fragment: "Researchers analyze data and write ..."</span>
        <span>
          At stage <code>{active + 1}</code>, the model is doing: {STEPS[active].text}
        </span>
        <span>
          Output of this stage feeds the next stage, and only at stage 6 one token is selected.
        </span>
      </div>
    </section>
  );
}
