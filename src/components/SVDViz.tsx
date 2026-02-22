import { useEffect, useMemo, useState } from "react";
import { SVD } from "svd-js";

const N = 14;

type Matrix = number[][];

function buildFaceMatrix() {
  const m = Array.from({ length: N }, () => Array.from({ length: N }, () => 0));

  for (let y = 2; y < N - 2; y += 1) {
    for (let x = 2; x < N - 2; x += 1) {
      const inFace = (x - 7) * (x - 7) + (y - 7) * (y - 7) <= 24;
      if (inFace) m[y][x] = 0.22;
    }
  }

  m[5][5] = 1;
  m[5][9] = 1;

  for (let x = 4; x <= 9; x += 1) {
    m[9][x] = 0.95;
  }

  for (let i = 2; i < 12; i += 1) {
    m[i][2] = Math.max(m[i][2], 0.74);
    m[i][11] = Math.max(m[i][11], 0.74);
    m[2][i] = Math.max(m[2][i], 0.74);
    m[11][i] = Math.max(m[11][i], 0.74);
  }

  return m;
}

function reconstruct(u: number[][], s: number[], v: number[][], k: number) {
  const out = Array.from({ length: N }, () => Array.from({ length: N }, () => 0));
  const use = Math.min(k, s.length);
  for (let y = 0; y < N; y += 1) {
    for (let x = 0; x < N; x += 1) {
      let sum = 0;
      for (let i = 0; i < use; i += 1) {
        sum += s[i] * u[y][i] * v[x][i];
      }
      out[y][x] = Math.max(0, Math.min(1, sum));
    }
  }
  return out;
}

function residual(a: Matrix, b: Matrix) {
  return a.map((row, y) => row.map((v, x) => Math.abs(v - b[y][x])));
}

function cumulativeEnergy(s: number[], k: number) {
  const total = s.reduce((acc, v) => acc + v * v, 0);
  const partial = s.slice(0, k).reduce((acc, v) => acc + v * v, 0);
  return total <= 1e-9 ? 0 : partial / total;
}

function componentMap(u: number[][], s: number[], v: number[][], idx: number) {
  const i = Math.max(0, Math.min(idx, s.length - 1));
  const m = Array.from({ length: N }, () => Array.from({ length: N }, () => 0));
  for (let y = 0; y < N; y += 1) {
    for (let x = 0; x < N; x += 1) {
      m[y][x] = s[i] * u[y][i] * v[x][i];
    }
  }
  return m;
}

function matrixMax(m: Matrix) {
  return Math.max(1e-6, ...m.flat());
}

function matrixAbsMax(m: Matrix) {
  return Math.max(1e-6, ...m.flat().map((v) => Math.abs(v)));
}

export function SVDViz() {
  const [k, setK] = useState(4);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(750);
  const [selectedComp, setSelectedComp] = useState(1);

  const image = useMemo(() => buildFaceMatrix(), []);
  const svd = useMemo(() => SVD(image), [image]);
  const u = svd.u as number[][];
  const v = svd.v as number[][];
  const s = svd.q;
  const maxRank = s.length;

  useEffect(() => {
    if (!playing) return;
    const timer = window.setInterval(() => {
      setK((curr) => (curr >= maxRank ? 1 : curr + 1));
    }, speed);
    return () => window.clearInterval(timer);
  }, [playing, speed, maxRank]);

  const recon = useMemo(() => reconstruct(u, s, v, k), [u, s, v, k]);
  const err = useMemo(() => residual(image, recon), [image, recon]);
  const comp = useMemo(() => componentMap(u, s, v, selectedComp - 1), [u, s, v, selectedComp]);

  const reconMax = matrixMax(recon);
  const errMax = matrixMax(err);
  const compMax = matrixAbsMax(comp);

  return (
    <section>
      <h2>SVD Insight Visualizer</h2>
      <p className="subtext">
        Real SVD decomposition of an image-like matrix. Build rank-k reconstruction progressively to show how a few singular components keep most structure.
      </p>

      <div className="explain-card">
        <strong>How to explain this slide</strong>
        <span>1) Original image is a matrix A.</span>
        <span>2) SVD splits A into ordered components by strength (singular values).</span>
        <span>3) Keep only first k components to compress data.</span>
        <span>4) Residual map shows what information gets lost.</span>
      </div>

      <div className="svd-layout">
        <div className="svd-panels">
          <div>
            <h3>Original Matrix A</h3>
            <div className="svd-grid" style={{ gridTemplateColumns: `repeat(${N}, 1fr)` }}>
              {image.flatMap((row, y) =>
                row.map((value, x) => (
                  <div key={`o-${x}-${y}`} className="svd-cell" style={{ background: `rgba(149, 215, 255, ${0.06 + value * 0.94})` }} />
                )),
              )}
            </div>
          </div>

          <div>
            <h3>Rank-{k} Reconstruction</h3>
            <div className="svd-grid" style={{ gridTemplateColumns: `repeat(${N}, 1fr)` }}>
              {recon.flatMap((row, y) =>
                row.map((value, x) => (
                  <div
                    key={`r-${x}-${y}`}
                    className="svd-cell"
                    style={{ background: `rgba(255, 208, 132, ${0.06 + (value / reconMax) * 0.94})` }}
                  />
                )),
              )}
            </div>
          </div>

          <div>
            <h3>Residual |A - A_k|</h3>
            <div className="svd-grid" style={{ gridTemplateColumns: `repeat(${N}, 1fr)` }}>
              {err.flatMap((row, y) =>
                row.map((value, x) => (
                  <div
                    key={`e-${x}-${y}`}
                    className="svd-cell"
                    style={{ background: `rgba(255, 136, 136, ${0.05 + (value / errMax) * 0.95})` }}
                  />
                )),
              )}
            </div>
          </div>

          <div>
            <h3>Component #{selectedComp}: sigma * u_i * v_i^T</h3>
            <div className="svd-grid" style={{ gridTemplateColumns: `repeat(${N}, 1fr)` }}>
              {comp.flatMap((row, y) =>
                row.map((value, x) => {
                  const t = Math.abs(value) / compMax;
                  const color = value >= 0 ? `rgba(255, 196, 110, ${0.1 + t * 0.9})` : `rgba(124, 194, 255, ${0.1 + t * 0.9})`;
                  return <div key={`c-${x}-${y}`} className="svd-cell" style={{ background: color }} />;
                }),
              )}
            </div>
          </div>
        </div>

        <div className="controls">
          <div className="preset-row">
            <button className="ghost-btn" onClick={() => setPlaying((v) => !v)}>
              {playing ? "Pause" : "Play"}
            </button>
            <button className="ghost-btn" onClick={() => setK(1)}>
              Reset
            </button>
            <button
              className="ghost-btn"
              onClick={() => {
                setPlaying(false);
                setK((v) => Math.max(1, v - 1));
              }}
            >
              Prev k
            </button>
            <button
              className="ghost-btn"
              onClick={() => {
                setPlaying(false);
                setK((v) => Math.min(maxRank, v + 1));
              }}
            >
              Next k
            </button>
          </div>

          <label>
            Rank k: {k}
            <input
              type="range"
              min={1}
              max={maxRank}
              step={1}
              value={k}
              onChange={(e) => {
                setPlaying(false);
                setK(Number(e.target.value));
              }}
            />
          </label>

          <label>
            Component to inspect: {selectedComp}
            <input
              type="range"
              min={1}
              max={maxRank}
              step={1}
              value={selectedComp}
              onChange={(e) => setSelectedComp(Number(e.target.value))}
            />
          </label>

          <label>
            Auto-play speed: {speed} ms
            <input type="range" min={250} max={1500} step={20} value={speed} onChange={(e) => setSpeed(Number(e.target.value))} />
          </label>

          <div className="formula-block">
            Compression ratio: {((k / (N * N)) * 100).toFixed(2)}% of raw entries (conceptual)
            <br />
            Energy kept by first {k}: {(cumulativeEnergy(s, k) * 100).toFixed(1)}%
            <br />
            sigma_{selectedComp}: {s[selectedComp - 1].toFixed(3)}
          </div>

          <div className="formula-block">
            Singular value spectrum
            <div className="svd-bars">
              {s.map((sv, i) => {
                const pct = (sv / (s[0] + 1e-9)) * 100;
                return (
                  <div key={i} className="svd-bar-row">
                    <span>#{i + 1}</span>
                    <div className="svd-bar-wrap">
                      <div className={i < k ? "svd-bar svd-bar-active" : "svd-bar"} style={{ width: `${pct}%` }} />
                    </div>
                    <span>{sv.toFixed(2)}</span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
