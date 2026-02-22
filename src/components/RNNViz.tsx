import { useEffect, useMemo, useState } from "react";

const VOCAB = ["h", "e", "l", "o", " "] as const;
const TOKENS = ["h", "e", "l", "l", "o", " "] as const;
const HIDDEN = 8;

type VocabToken = (typeof VOCAB)[number];

function softmax(values: number[]) {
  const m = Math.max(...values);
  const exps = values.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / (s + 1e-9));
}

function tanh(v: number) {
  return Math.tanh(v);
}

function tokenVec(token: VocabToken) {
  return VOCAB.map((v) => (v === token ? 1 : 0));
}

function buildWeights() {
  const wxh = Array.from({ length: HIDDEN }, (_, i) =>
    Array.from({ length: VOCAB.length }, (_, j) => Math.sin((i + 1) * (j + 2)) * 0.7),
  );
  const whh = Array.from({ length: HIDDEN }, (_, i) =>
    Array.from({ length: HIDDEN }, (_, j) => Math.cos((i + 2) * (j + 1)) * 0.22),
  );
  const why = Array.from({ length: VOCAB.length }, (_, i) =>
    Array.from({ length: HIDDEN }, (_, j) => Math.sin((i + 3) * (j + 1)) * 0.45),
  );
  const bh = Array.from({ length: HIDDEN }, (_, i) => Math.sin(i + 0.3) * 0.1);
  const by = Array.from({ length: VOCAB.length }, (_, i) => Math.cos(i + 0.7) * 0.1);
  return { wxh, whh, why, bh, by };
}

function stepRnn(
  x: number[],
  hPrev: number[],
  w: ReturnType<typeof buildWeights>,
  recurrentGain: number,
  inputGain: number,
) {
  const h = Array.from({ length: HIDDEN }, (_, i) => {
    let sum = w.bh[i];
    for (let j = 0; j < VOCAB.length; j += 1) sum += w.wxh[i][j] * x[j] * inputGain;
    for (let j = 0; j < HIDDEN; j += 1) sum += w.whh[i][j] * hPrev[j] * recurrentGain;
    return tanh(sum);
  });

  const logits = Array.from({ length: VOCAB.length }, (_, i) => {
    let sum = w.by[i];
    for (let j = 0; j < HIDDEN; j += 1) sum += w.why[i][j] * h[j];
    return sum;
  });

  return { h, probs: softmax(logits) };
}

function runSequence(tokens: readonly VocabToken[], w: ReturnType<typeof buildWeights>, recurrentGain: number, inputGain: number) {
  const states: number[][] = [];
  const probs: number[][] = [];
  let h = Array.from({ length: HIDDEN }, () => 0);
  for (let i = 0; i < tokens.length; i += 1) {
    const x = tokenVec(tokens[i]);
    const out = stepRnn(x, h, w, recurrentGain, inputGain);
    h = out.h;
    states.push([...h]);
    probs.push(out.probs);
  }
  return { states, probs };
}

function l2(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

export function RNNViz() {
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(520);
  const [t, setT] = useState(0);
  const [recurrentGain, setRecurrentGain] = useState(1);
  const [inputGain, setInputGain] = useState(1);
  const weights = useMemo(() => buildWeights(), []);

  const timeline = useMemo(() => {
    return runSequence(TOKENS, weights, recurrentGain, inputGain);
  }, [weights, recurrentGain, inputGain]);

  const influence = useMemo(() => {
    const baseState = timeline.states[t];
    const raw = TOKENS.map((_, idx) => {
      if (idx > t) return 0;
      const modified = [...TOKENS] as VocabToken[];
      modified[idx] = " ";
      const alt = runSequence(modified, weights, recurrentGain, inputGain);
      return l2(baseState, alt.states[t]);
    });
    const max = Math.max(1e-9, ...raw);
    return raw.map((v) => v / max);
  }, [timeline, t, weights, recurrentGain, inputGain]);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setT((v) => (v + 1) % TOKENS.length);
    }, speed);
    return () => window.clearInterval(timer);
  }, [playing, speed]);

  const state = timeline.states[t];
  const probs = timeline.probs[t];
  const predIdx = probs.reduce((best, p, i, arr) => (p > arr[best] ? i : best), 0);
  const predToken = VOCAB[predIdx];
  const generated = TOKENS.slice(0, t + 1).join("");

  return (
    <section>
      <h2>RNN Sequence Visualizer</h2>
      <p className="subtext">
        This unrolled RNN reads one token per timestep, updates hidden memory, and predicts the next token. You can narrate how memory flows left to right.
      </p>

      <div className="explain-card">
        <strong>How to explain this slide</strong>
        <span>1) Each cell gets current token and previous hidden state.</span>
        <span>2) Hidden state carries context across time.</span>
        <span>3) Output probabilities estimate the next token.</span>
        <span>4) Memory influence bars show how strongly each past token still affects the current state.</span>
      </div>

      <div className="rnn-layout">
        <div className="rnn-track">
          {TOKENS.map((token, idx) => (
            <div key={idx} className={idx === t ? "rnn-step rnn-step-active" : "rnn-step"}>
              <div className="rnn-token">x{idx}: "{token === " " ? "space" : token}"</div>
              <div className="rnn-cell">h{idx + 1}</div>
            </div>
          ))}
        </div>

        <div className="rnn-panels">
          <div className="formula-block">
            Current timestep: {t + 1}/{TOKENS.length}
            <br />
            Sequence seen: "{generated.replace(" ", "_ ")}"
            <br />
            Predicted next token: "{predToken === " " ? "space" : predToken}" ({(probs[predIdx] * 100).toFixed(1)}%)
          </div>

          <div className="formula-block">
            Hidden state h{t + 1}
            <div className="rnn-hidden-row">
              {state.map((v, idx) => (
                <div
                  key={idx}
                  className="rnn-hidden-unit"
                  style={{
                    height: `${10 + Math.abs(v) * 48}px`,
                    background: v >= 0 ? "#8de6b2" : "#90c8ff",
                  }}
                />
              ))}
            </div>
          </div>

          <div className="formula-block">
            Memory influence on h{t + 1}
            <div className="rnn-memory-row">
              {TOKENS.map((token, idx) => {
                const value = influence[idx];
                return (
                  <div key={`mem-${idx}`} className="rnn-memory-item">
                    <div
                      className={idx <= t ? "rnn-memory-bar" : "rnn-memory-bar rnn-memory-future"}
                      style={{ height: `${8 + value * 54}px` }}
                    />
                    <span>{token === " " ? "_" : token}</span>
                    <small>t{idx + 1}</small>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="formula-block">
            Next-token probabilities
            <div className="rnn-prob-list">
              {VOCAB.map((tok, idx) => (
                <div key={tok} className="rnn-prob-item">
                  <span>{tok === " " ? "space" : tok}</span>
                  <div className="rnn-prob-bar-wrap">
                    <div className="rnn-prob-bar" style={{ width: `${probs[idx] * 100}%` }} />
                  </div>
                  <span>{(probs[idx] * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="controls">
          <div className="preset-row">
            <button className="ghost-btn" onClick={() => setPlaying((v) => !v)}>
              {playing ? "Pause" : "Play"}
            </button>
            <button className="ghost-btn" onClick={() => setT(0)}>
              Reset
            </button>
            <button
              className="ghost-btn"
              onClick={() => {
                setPlaying(false);
                setT((v) => Math.max(0, v - 1));
              }}
            >
              Prev Step
            </button>
            <button
              className="ghost-btn"
              onClick={() => {
                setPlaying(false);
                setT((v) => Math.min(TOKENS.length - 1, v + 1));
              }}
            >
              Next Step
            </button>
          </div>

          <label>
            Speed: {speed} ms
            <input type="range" min={180} max={1200} step={20} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} />
          </label>

          <label>
            Timestep: {t + 1}
            <input type="range" min={0} max={TOKENS.length - 1} step={1} value={t} onChange={(e) => {
              setPlaying(false);
              setT(Number(e.target.value));
            }} />
          </label>

          <label>
            Recurrent memory strength: {recurrentGain.toFixed(2)}
            <input
              type="range"
              min={0.2}
              max={1.45}
              step={0.01}
              value={recurrentGain}
              onChange={(e) => setRecurrentGain(Number(e.target.value))}
            />
          </label>

          <label>
            Input injection strength: {inputGain.toFixed(2)}
            <input
              type="range"
              min={0.4}
              max={1.6}
              step={0.01}
              value={inputGain}
              onChange={(e) => setInputGain(Number(e.target.value))}
            />
          </label>

          <div className="formula-block">
            Model sketch: h_t = tanh(Wxh x_t + Whh h_(t-1) + b)
            <br />
            y_t = softmax(Why h_t + b_y)
          </div>
        </div>
      </div>
    </section>
  );
}
