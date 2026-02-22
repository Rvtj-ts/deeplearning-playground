import { useEffect, useMemo, useState } from "react";

const IMAGE_SIZE = 14;
const KERNEL_SIZE = 3;

type Matrix = number[][];

type KernelDef = {
  name: string;
  description: string;
  matrix: Matrix;
  color: string;
};

const KERNELS: KernelDef[] = [
  {
    name: "Eye Spot Detector",
    description: "Looks for compact bright spots (eye-like points).",
    matrix: [
      [0, -1, 0],
      [-1, 4, -1],
      [0, -1, 0],
    ],
    color: "#9fd2ff",
  },
  {
    name: "Mouth Line Detector",
    description: "Responds to horizontal bright lines (mouth-like strokes).",
    matrix: [
      [-1, -1, -1],
      [2, 2, 2],
      [-1, -1, -1],
    ],
    color: "#ffd79a",
  },
  {
    name: "Face Edge Detector",
    description: "Responds to vertical edge transitions around the face boundary.",
    matrix: [
      [-1, 0, 1],
      [-1, 0, 1],
      [-1, 0, 1],
    ],
    color: "#b0edc8",
  },
];

function buildSampleImage() {
  const m = Array.from({ length: IMAGE_SIZE }, () => Array.from({ length: IMAGE_SIZE }, () => 0));

  for (let y = 2; y < IMAGE_SIZE - 2; y += 1) {
    for (let x = 2; x < IMAGE_SIZE - 2; x += 1) {
      const inFace = (x - 7) * (x - 7) + (y - 7) * (y - 7) <= 24;
      if (inFace) m[y][x] = 0.24;
    }
  }

  m[5][5] = 1;
  m[5][9] = 1;

  for (let x = 4; x <= 9; x += 1) {
    m[9][x] = 0.95;
  }

  for (let i = 2; i < 12; i += 1) {
    m[i][2] = Math.max(m[i][2], 0.75);
    m[i][11] = Math.max(m[i][11], 0.75);
    m[2][i] = Math.max(m[2][i], 0.75);
    m[11][i] = Math.max(m[11][i], 0.75);
  }

  return m;
}

function conv2dSame(image: Matrix, kernel: Matrix) {
  const outSize = image.length;
  const pad = Math.floor(kernel.length / 2);
  const out = Array.from({ length: outSize }, () => Array.from({ length: outSize }, () => 0));

  for (let y = 0; y < outSize; y += 1) {
    for (let x = 0; x < outSize; x += 1) {
      let sum = 0;
      for (let ky = 0; ky < kernel.length; ky += 1) {
        for (let kx = 0; kx < kernel[0].length; kx += 1) {
          const iy = y + ky - pad;
          const ix = x + kx - pad;
          const pixel = iy >= 0 && iy < image.length && ix >= 0 && ix < image[0].length ? image[iy][ix] : 0;
          sum += pixel * kernel[ky][kx];
        }
      }
      out[y][x] = Math.max(0, sum);
    }
  }
  return out;
}

function maxPool2x2(feature: Matrix) {
  const outSize = Math.floor(feature.length / 2);
  const out = Array.from({ length: outSize }, () => Array.from({ length: outSize }, () => 0));
  for (let y = 0; y < outSize; y += 1) {
    for (let x = 0; x < outSize; x += 1) {
      const y0 = y * 2;
      const x0 = x * 2;
      out[y][x] = Math.max(feature[y0][x0], feature[y0][x0 + 1], feature[y0 + 1][x0], feature[y0 + 1][x0 + 1]);
    }
  }
  return out;
}

function flatten(m: Matrix) {
  return m.flat();
}

function gridMax(m: Matrix) {
  return Math.max(1e-6, ...m.flat());
}

function randUnit(seed: number) {
  const x = Math.sin(seed * 12.9898) * 43758.5453;
  return x - Math.floor(x);
}

function strongestPoints(map: Matrix, count = 3) {
  const points: Array<{ x: number; y: number; v: number }> = [];
  for (let y = 0; y < map.length; y += 1) {
    for (let x = 0; x < map[0].length; x += 1) {
      points.push({ x, y, v: map[y][x] });
    }
  }
  points.sort((a, b) => b.v - a.v);

  const out: Array<{ x: number; y: number; v: number }> = [];
  for (const p of points) {
    if (p.v <= 0) continue;
    const farEnough = out.every((q) => Math.abs(q.x - p.x) + Math.abs(q.y - p.y) >= 3);
    if (farEnough) out.push(p);
    if (out.length >= count) break;
  }
  return out;
}

export function CNNViz() {
  const [activeKernel, setActiveKernel] = useState(0);
  const [dropRate, setDropRate] = useState(0.35);
  const [scanPlaying, setScanPlaying] = useState(true);
  const [scanSpeed, setScanSpeed] = useState(160);
  const [scanIndex, setScanIndex] = useState(0);
  const [iterPlaying, setIterPlaying] = useState(true);
  const [iteration, setIteration] = useState(1);

  const image = useMemo(() => buildSampleImage(), []);
  const activeKernelDef = KERNELS[activeKernel];

  const featureMaps = useMemo(() => KERNELS.map((k) => conv2dSame(image, k.matrix)), [image]);
  const pooledMaps = useMemo(() => featureMaps.map((m) => maxPool2x2(m)), [featureMaps]);

  const convMap = featureMaps[activeKernel];
  const pooled = pooledMaps[activeKernel];
  const vector = useMemo(() => flatten(pooled), [pooled]);

  const scanPositions = useMemo(() => {
    const positions: Array<{ x: number; y: number }> = [];
    for (let y = 0; y < IMAGE_SIZE - KERNEL_SIZE + 1; y += 1) {
      for (let x = 0; x < IMAGE_SIZE - KERNEL_SIZE + 1; x += 1) {
        positions.push({ x, y });
      }
    }
    return positions;
  }, []);

  useEffect(() => {
    if (!scanPlaying) return;
    const t = window.setInterval(() => {
      setScanIndex((i) => (i + 1) % scanPositions.length);
    }, scanSpeed);
    return () => window.clearInterval(t);
  }, [scanPlaying, scanSpeed, scanPositions.length]);

  useEffect(() => {
    if (!iterPlaying) return;
    const t = window.setInterval(() => {
      setIteration((i) => (i >= 120 ? 1 : i + 1));
    }, 330);
    return () => window.clearInterval(t);
  }, [iterPlaying]);

  const scanPos = scanPositions[scanIndex];
  const patch = useMemo(() => {
    const out = Array.from({ length: KERNEL_SIZE }, () => Array.from({ length: KERNEL_SIZE }, () => 0));
    for (let y = 0; y < KERNEL_SIZE; y += 1) {
      for (let x = 0; x < KERNEL_SIZE; x += 1) {
        out[y][x] = image[scanPos.y + y][scanPos.x + x];
      }
    }
    return out;
  }, [image, scanPos]);

  const patchActivation = useMemo(() => {
    let sum = 0;
    for (let y = 0; y < KERNEL_SIZE; y += 1) {
      for (let x = 0; x < KERNEL_SIZE; x += 1) {
        sum += patch[y][x] * activeKernelDef.matrix[y][x];
      }
    }
    return Math.max(0, sum);
  }, [patch, activeKernelDef]);

  const droppedMask = useMemo(() => {
    return vector.map((_, idx) => randUnit(iteration * 1000 + idx * 37) > dropRate);
  }, [vector, iteration, dropRate]);

  const keptCount = droppedMask.filter(Boolean).length;
  const convMax = gridMax(convMap);
  const poolMax = gridMax(pooled);
  const topPoints = useMemo(() => strongestPoints(convMap, 3), [convMap]);

  return (
    <section>
      <h2>CNN Feature Flow Visualizer</h2>
      <p className="subtext">
        Instead of "raw pixels", each conv map is a feature detector. Use the detector selector to show eyes, mouth line, and face-edge evidence separately.
      </p>

      <div className="explain-card">
        <strong>How to explain this slide</strong>
        <span>1) Choose a detector (eye, mouth, or edge).</span>
        <span>2) Bright cells in the conv map mean this feature is present there.</span>
        <span>3) "Top activations" are the model's strongest evidence locations.</span>
        <span>4) Pooling and dropout keep signal while improving robustness.</span>
      </div>

      <div className="cnn-kernel-row">
        {KERNELS.map((k, idx) => (
          <button key={k.name} className={idx === activeKernel ? "tab tab-active" : "tab"} onClick={() => setActiveKernel(idx)}>
            {k.name}
          </button>
        ))}
      </div>

      <div className="cnn-layout">
        <div className="cnn-main">
          <div>
            <h3>Input Image + Sliding Kernel</h3>
            <div className="cnn-grid" style={{ gridTemplateColumns: `repeat(${IMAGE_SIZE}, 1fr)` }}>
              {image.flatMap((row, y) =>
                row.map((value, x) => {
                  const inWindow = x >= scanPos.x && x < scanPos.x + KERNEL_SIZE && y >= scanPos.y && y < scanPos.y + KERNEL_SIZE;
                  const isTop = topPoints.some((p) => p.x === x && p.y === y);
                  return (
                    <div
                      key={`${x}-${y}`}
                      className={inWindow ? "cnn-cell cnn-window" : isTop ? "cnn-cell cnn-cell-focus" : "cnn-cell"}
                      style={{ background: `rgba(148, 218, 255, ${0.08 + value * 0.92})` }}
                    />
                  );
                }),
              )}
            </div>
          </div>

          <div>
            <h3>Conv Feature Map ({IMAGE_SIZE}x{IMAGE_SIZE})</h3>
            <div className="cnn-grid" style={{ gridTemplateColumns: `repeat(${convMap.length}, 1fr)` }}>
              {convMap.flatMap((row, y) =>
                row.map((value, x) => {
                  const isFocus = x === scanPos.x + 1 && y === scanPos.y + 1;
                  const isTop = topPoints.some((p) => p.x === x && p.y === y);
                  return (
                    <div
                      key={`conv-${x}-${y}`}
                      className={isFocus || isTop ? "cnn-cell cnn-cell-focus" : "cnn-cell"}
                      style={{ background: `color-mix(in srgb, ${activeKernelDef.color} ${(6 + (value / convMax) * 94).toFixed(1)}%, #0f2430)` }}
                    />
                  );
                }),
              )}
            </div>
          </div>

          <div>
            <h3>MaxPool to Flatten</h3>
            <div className="cnn-grid" style={{ gridTemplateColumns: `repeat(${pooled.length}, 1fr)` }}>
              {pooled.flatMap((row, y) =>
                row.map((value, x) => (
                  <div
                    key={`pool-${x}-${y}`}
                    className="cnn-cell"
                    style={{ background: `rgba(172, 243, 191, ${0.06 + (value / poolMax) * 0.94})` }}
                  />
                )),
              )}
            </div>
            <div className="cnn-compress">
              Per channel: {IMAGE_SIZE}x{IMAGE_SIZE} to {pooled.length}x{pooled.length} to {vector.length}
              <br />
              Three detectors to final features from three channels.
            </div>
          </div>
        </div>

        <div className="controls">
          <div className="formula-block">
            <strong>{activeKernelDef.name}</strong>
            <br />
            {activeKernelDef.description}
            <div className="cnn-kernel-mini">
              {activeKernelDef.matrix.flatMap((row, y) =>
                row.map((v, x) => (
                  <div key={`k-${x}-${y}`} className="cnn-kernel-weight">
                    {v}
                  </div>
                )),
              )}
            </div>
          </div>

          <div className="formula-block">
            Top activations (where feature is strongest)
            {topPoints.map((p, i) => (
              <div key={i}>
                #{i + 1}: ({p.x}, {p.y}) strength {p.v.toFixed(2)}
              </div>
            ))}
          </div>

          <div className="preset-row">
            <button className="ghost-btn" onClick={() => setScanPlaying((v) => !v)}>
              {scanPlaying ? "Pause Scan" : "Play Scan"}
            </button>
            <button className="ghost-btn" onClick={() => setIterPlaying((v) => !v)}>
              {iterPlaying ? "Pause Training" : "Play Training"}
            </button>
            <button
              className="ghost-btn"
              onClick={() => {
                setScanPlaying(false);
                setScanIndex((i) => (i - 1 + scanPositions.length) % scanPositions.length);
              }}
            >
              Prev Scan
            </button>
            <button
              className="ghost-btn"
              onClick={() => {
                setScanPlaying(false);
                setScanIndex((i) => (i + 1) % scanPositions.length);
              }}
            >
              Next Scan
            </button>
          </div>

          <label>
            Scan speed: {scanSpeed} ms
            <input type="range" min={60} max={400} step={10} value={scanSpeed} onChange={(e) => setScanSpeed(Number(e.target.value))} />
          </label>

          <label>
            Scan index: {scanIndex + 1}/{scanPositions.length}
            <input
              type="range"
              min={0}
              max={scanPositions.length - 1}
              step={1}
              value={scanIndex}
              onChange={(e) => {
                setScanPlaying(false);
                setScanIndex(Number(e.target.value));
              }}
            />
          </label>

          <label>
            Training iteration: {iteration}
            <input type="range" min={1} max={120} step={1} value={iteration} onChange={(e) => setIteration(Number(e.target.value))} />
          </label>

          <label>
            Dropout rate: {(dropRate * 100).toFixed(0)}%
            <input type="range" min={0.1} max={0.75} step={0.01} value={dropRate} onChange={(e) => setDropRate(Number(e.target.value))} />
          </label>

          <div className="formula-block">
            Current scan patch activation: {patchActivation.toFixed(3)}
            <br />
            Kernel position: ({scanPos.x}, {scanPos.y})
            <br />
            Mapped feature cell: ({scanPos.x + 1}, {scanPos.y + 1})
            <br />
            Kept units this iteration: {keptCount}/{vector.length}
          </div>

          <div className="formula-block">
            Dropout mask sample
            <div className="cnn-dropout-row">
              {vector.slice(0, 42).map((v, i) => {
                const on = droppedMask[i];
                return (
                  <div
                    key={i}
                    className={on ? "cnn-dropunit" : "cnn-dropunit cnn-dropunit-off"}
                    style={{ height: `${7 + v * 44}px` }}
                  />
                );
              })}
            </div>
            <div className="cnn-dropout-labels">
              <span>active units</span>
              <span>dropped units</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
