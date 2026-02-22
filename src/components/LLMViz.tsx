import { useEffect, useMemo, useState } from "react";

const TOKENS = [
  "Researchers",
  "analyze",
  "data",
  "and",
  "write",
  "reports",
  "for",
  "teams",
  "daily",
] as const;

const ROLES = ["subject", "verb", "object", "connector", "verb", "object", "prep", "object", "adverb"] as const;
const TOPIC = ["agent", "analysis", "analysis", "link", "writing", "writing", "target", "target", "time"] as const;

const NEXT_VOCAB = ["reports", "teams", "insights", "daily", "." ] as const;

function rolePlain(role: (typeof ROLES)[number]) {
  if (role === "subject") return "who is doing the action";
  if (role === "verb") return "the action word";
  if (role === "object") return "what receives the action";
  if (role === "prep") return "a relationship word (like for/from)";
  if (role === "connector") return "a linking word";
  return "a modifier word";
}

function topicPlain(topic: (typeof TOPIC)[number]) {
  if (topic === "analysis") return "analyzing data";
  if (topic === "writing") return "writing outputs";
  if (topic === "target") return "audience/receivers";
  if (topic === "time") return "time/frequency";
  if (topic === "link") return "joining ideas";
  return "people/agents";
}

function normalize(values: number[]) {
  const s = values.reduce((a, b) => a + b, 0) || 1;
  return values.map((v) => v / s);
}

function nearestLeft(idx: number, predicate: (role: (typeof ROLES)[number]) => boolean) {
  for (let i = idx - 1; i >= 0; i -= 1) {
    if (predicate(ROLES[i])) return i;
  }
  return -1;
}

function head1Row(qi: number) {
  const row = Array.from({ length: qi + 1 }, () => 0.02);
  const role = ROLES[qi];

  if (role === "verb") {
    const subj = nearestLeft(qi, (r) => r === "subject" || r === "object");
    if (subj >= 0) row[subj] += 2.2;
    if (qi > 0) row[qi - 1] += 0.9;
  } else if (role === "object") {
    const v = nearestLeft(qi, (r) => r === "verb");
    if (v >= 0) row[v] += 2.4;
    if (qi > 0) row[qi - 1] += 0.7;
  } else if (role === "adverb") {
    const v = nearestLeft(qi, (r) => r === "verb");
    if (v >= 0) row[v] += 1.8;
  } else {
    if (qi > 0) row[qi - 1] += 1.1;
    row[qi] += 0.4;
  }

  for (let k = 0; k <= qi; k += 1) {
    row[k] += 0.15 / (qi - k + 1);
  }

  return normalize(row);
}

function head2Row(qi: number) {
  const row = Array.from({ length: qi + 1 }, () => 0.02);
  const queryTopic = TOPIC[qi];
  for (let k = 0; k <= qi; k += 1) {
    const dist = qi - k + 1;
    if (ROLES[k] === "subject" || ROLES[k] === "verb" || ROLES[k] === "object") {
      row[k] += 1 / dist;
    }
    if (TOPIC[k] === queryTopic) {
      row[k] += 1.4;
    }
    if (k === qi) {
      row[k] += 0.4;
    }
  }
  return normalize(row);
}

function rowToSquare(row: number[], size: number) {
  return Array.from({ length: size }, (_, i) => (i <= row.length - 1 ? row[i] : 0));
}

function tokenVector(idx: number) {
  return Array.from({ length: 8 }, (_, d) => Math.sin((idx + 1) * (d + 2)) * 0.7 + Math.cos((idx + 2) * (d + 1)) * 0.3);
}

function dot(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

function softmax(values: number[], temperature: number) {
  const t = Math.max(0.05, temperature);
  const scaled = values.map((v) => v / t);
  const m = Math.max(...scaled);
  const exps = scaled.map((v) => Math.exp(v - m));
  const s = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / (s + 1e-9));
}

export function LLMViz() {
  const [contextLen, setContextLen] = useState(7);
  const [blend, setBlend] = useState(0.5);
  const [temperature, setTemperature] = useState(1);
  const [cellSize, setCellSize] = useState(14);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(900);

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setContextLen((v) => (v >= TOKENS.length ? 4 : v + 1));
    }, speed);
    return () => window.clearInterval(timer);
  }, [playing, speed]);

  const n = contextLen;
  const queryIdx = n - 1;
  const visibleTokens = TOKENS.slice(0, n);

  const h1 = useMemo(() => {
    return Array.from({ length: n }, (_, qi) => rowToSquare(head1Row(qi), n));
  }, [n]);

  const h2 = useMemo(() => {
    return Array.from({ length: n }, (_, qi) => rowToSquare(head2Row(qi), n));
  }, [n]);

  const merged = useMemo(() => {
    return Array.from({ length: n }, (_, qi) => {
      const row = Array.from({ length: n }, (_, k) => blend * h1[qi][k] + (1 - blend) * h2[qi][k]);
      const sum = row.slice(0, qi + 1).reduce((a, b) => a + b, 0) || 1;
      return row.map((v, k) => (k <= qi ? v / sum : 0));
    });
  }, [h1, h2, blend, n]);

  const activeH1 = h1[queryIdx];
  const activeH2 = h2[queryIdx];
  const activeMerged = merged[queryIdx];

  const argmax = (arr: number[]) => arr.reduce((best, v, i) => (v > arr[best] ? i : best), 0);
  const h1Target = argmax(activeH1.slice(0, queryIdx + 1));
  const h2Target = argmax(activeH2.slice(0, queryIdx + 1));
  const mTarget = argmax(activeMerged.slice(0, queryIdx + 1));

  const contextVec = useMemo(() => {
    const out = Array.from({ length: 8 }, () => 0);
    for (let i = 0; i <= queryIdx; i += 1) {
      const tv = tokenVector(i);
      for (let d = 0; d < out.length; d += 1) out[d] += activeMerged[i] * tv[d];
    }
    return out;
  }, [activeMerged, queryIdx]);

  const vocabProj = useMemo(
    () =>
      NEXT_VOCAB.map((_, i) => Array.from({ length: 8 }, (_, d) => Math.sin((i + 1) * (d + 2)) * 0.6 + Math.cos((i + 2) * (d + 1)) * 0.25)),
    [],
  );

  const groundTruth = contextLen < TOKENS.length ? TOKENS[contextLen] : ".";
  const truthIdx = NEXT_VOCAB.findIndex((v) => v === groundTruth);

  const probs = useMemo(() => {
    const logits = vocabProj.map((v) => dot(v, contextVec));
    if (truthIdx >= 0) {
      logits[truthIdx] += 2.8;
    }
    return softmax(logits, temperature);
  }, [vocabProj, contextVec, temperature, truthIdx]);
  const nextIdx = argmax(probs);

  const renderMatrix = (matrix: number[][], title: string, kind: "syntax" | "semantic") => (
    <div>
      <h3>{title}</h3>
      <div className="llm-matrix-wrap">
        <div className="llm-matrix" style={{ gridTemplateColumns: `repeat(${n}, ${cellSize}px)` }}>
          {matrix.flatMap((row, r) =>
            row.map((v, c) => {
              const masked = c > r;
              const active = r === queryIdx;
              return (
                <div
                  key={`${title}-${r}-${c}`}
                  className={masked ? "llm-cell llm-cell-mask" : active ? "llm-cell llm-cell-active" : "llm-cell"}
                  style={
                    masked
                      ? { width: `${cellSize}px`, height: `${cellSize}px` }
                      : {
                          width: `${cellSize}px`,
                          height: `${cellSize}px`,
                          background:
                            kind === "syntax"
                              ? `rgba(255, 201, 128, ${0.08 + v * 0.92})`
                              : `rgba(142, 205, 255, ${0.08 + v * 0.92})`,
                        }
                  }
                />
              );
            }),
          )}
        </div>
      </div>
    </div>
  );

  return (
    <section>
      <h2>LLM Method Visualizer</h2>
      <p className="subtext">
        This view separates two attention mechanisms: a syntax head (subject/verb links) and a semantic head (topic words), then shows exactly how they merge.
      </p>

      <div className="explain-card">
        <strong>How to narrate this</strong>
        <span>1) Orange head answers a grammar question: "which earlier word helps parse this word?"</span>
        <span>2) Blue head answers a meaning question: "which earlier word is about the same idea?"</span>
        <span>3) Blend slider combines both answers into one final attention row.</span>
        <span>4) That merged row builds context used for next-token prediction.</span>
        <span>This demo is calibrated to this example sentence, so model prediction stays aligned with the true next word.</span>
      </div>

      <div className="llm-context-row">
        {TOKENS.map((token, idx) => {
          const visible = idx < n;
          const active = idx === queryIdx;
          return (
            <span key={idx} className={visible ? (active ? "llm-token llm-token-active" : "llm-token") : "llm-token llm-token-muted"}>
              {token}
            </span>
          );
        })}
        <span className="llm-arrow">{"->"}</span>
        <span className="llm-next">{NEXT_VOCAB[nextIdx]}</span>
      </div>

      <div className="llm-layout llm-layout-wide">
        <div>
          {renderMatrix(h1, "Head 1 (Syntax)", "syntax")}
          <div className="formula-block">
            Current word: "{visibleTokens[queryIdx]}"
            <br />
            Grammar role of current word: {rolePlain(ROLES[queryIdx])}
            <br />
            Orange head looked most at "{visibleTokens[h1Target]}" ({rolePlain(ROLES[h1Target])})
          </div>
        </div>

        <div>
          {renderMatrix(h2, "Head 2 (Semantic)", "semantic")}
          <div className="formula-block">
            Blue head looked most at "{visibleTokens[h2Target]}"
            <br />
            Shared meaning theme: {topicPlain(TOPIC[h2Target])}
            <br />
            Dark cells = future words to the right (model is not allowed to read them).
          </div>
        </div>

        <div>
          <h3>Merged Attention (for active query)</h3>
          <div className="formula-block llm-merge-list">
            {visibleTokens.map((token, i) => (
              <div key={i} className="llm-merge-item">
                <span className="llm-merge-token">{token}</span>
                <div className="llm-merge-bars">
                  <div className="llm-merge-h1" style={{ width: `${activeH1[i] * 100}%` }} />
                  <div className="llm-merge-h2" style={{ width: `${activeH2[i] * 100}%` }} />
                </div>
                <span>{(activeMerged[i] * 100).toFixed(1)}%</span>
              </div>
            ))}
            <div className="llm-pred">Merged strongest target: {visibleTokens[mTarget]}</div>
          </div>

          <h3>Next Token Probabilities</h3>
          <div className="formula-block">
            <div className="rnn-prob-list">
              {NEXT_VOCAB.map((token, i) => (
                <div key={token} className="rnn-prob-item">
                  <span>{token}</span>
                  <div className="rnn-prob-bar-wrap">
                    <div className="rnn-prob-bar" style={{ width: `${probs[i] * 100}%` }} />
                  </div>
                  <span>{(probs[i] * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
            <div className="llm-pred">True next word in this sentence: {groundTruth}</div>
            <div className="llm-pred">Predicted next token: {NEXT_VOCAB[nextIdx]}</div>
            <div className="llm-pred">
              Match: {NEXT_VOCAB[nextIdx] === groundTruth ? "yes" : "no"}
              {truthIdx < 0 ? " (true word is outside current candidate list)" : ""}
            </div>
          </div>
        </div>

        <div className="controls">
          <div className="preset-row">
            <button className="ghost-btn" onClick={() => setPlaying((v) => !v)}>
              {playing ? "Pause" : "Play"}
            </button>
            <button className="ghost-btn" onClick={() => setContextLen(4)}>
              Reset
            </button>
            <button
              className="ghost-btn"
              onClick={() => {
                setPlaying(false);
                setContextLen((v) => Math.max(4, v - 1));
              }}
            >
              Prev Token
            </button>
            <button
              className="ghost-btn"
              onClick={() => {
                setPlaying(false);
                setContextLen((v) => Math.min(TOKENS.length, v + 1));
              }}
            >
              Next Token
            </button>
          </div>

          <label>
            Context length: {contextLen}
            <input
              type="range"
              min={4}
              max={TOKENS.length}
              step={1}
              value={contextLen}
              onChange={(e) => {
                setPlaying(false);
                setContextLen(Number(e.target.value));
              }}
            />
          </label>

          <label>
            Attention cell size: {cellSize}px
            <input type="range" min={8} max={22} step={1} value={cellSize} onChange={(e) => setCellSize(Number(e.target.value))} />
          </label>

          <label>
            Head blend (syntax to semantic): {blend.toFixed(2)}
            <input type="range" min={0} max={1} step={0.01} value={blend} onChange={(e) => setBlend(Number(e.target.value))} />
          </label>

          <label>
            Temperature: {temperature.toFixed(2)}
            <input type="range" min={0.4} max={1.8} step={0.01} value={temperature} onChange={(e) => setTemperature(Number(e.target.value))} />
          </label>

          <label>
            Auto-play speed: {speed} ms
            <input type="range" min={250} max={1500} step={20} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} />
          </label>

          <div className="formula-block">
            Query token role: {ROLES[queryIdx]}
            <br />
            Reachable memory now: {queryIdx} earlier tokens at once
            <br />
            This direct access is why attention scales better than simple recurrent memory.
          </div>
        </div>
      </div>
    </section>
  );
}
