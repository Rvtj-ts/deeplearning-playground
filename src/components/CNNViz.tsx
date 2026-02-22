import { useEffect, useMemo, useState } from "react";

const IMAGE_SIZE = 14;
const KERNEL_SIZE = 3;

type Matrix = number[][];

function buildSampleImage() {
  const m = Array.from({ length: IMAGE_SIZE }, () => Array.from({ length: IMAGE_SIZE }, () => 0));

  for (let y = 2; y < IMAGE_SIZE - 2; y += 1) {
    for (let x = 2; x < IMAGE_SIZE - 2; x += 1) {
      const inFace = (x - 7) * (x - 7) + (y - 7) * (y - 7) <= 24;
      if (inFace) m[y][x] = 0.2;
    }
  }

  m[5][5] = 1;
  m[5][9] = 1;
  m[9][4] = 0.9;
  m[9][5] = 0.9;
  m[9][6] = 0.9;
  m[9][7] = 0.9;
  m[9][8] = 0.9;
  m[9][9] = 0.9;

  for (let i = 2; i < 12; i += 1) {
    m[i][2] = Math.max(m[i][2], 0.75);
    m[i][11] = Math.max(m[i][11], 0.75);
    m[2][i] = Math.max(m[2][i], 0.75);
    m[11][i] = Math.max(m[11][i], 0.75);
  }

  return m;
}

function conv2d(image: Matrix, kernel: Matrix) {
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
      out[y][x] = Math.abs(sum);
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

export function CNNViz() {
  const [dropRate, setDropRate] = useState(0.35);
  const [scanPlaying, setScanPlaying] = useState(true);
  const [scanSpeed, setScanSpeed] = useState(160);
  const [scanIndex, setScanIndex] = useState(0);
  const [iterPlaying, setIterPlaying] = useState(true);
  const [iteration, setIteration] = useState(1);

  const image = useMemo(() => buildSampleImage(), []);
  const kernel = useMemo<Matrix>(() => [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], []);
  const convMap = useMemo(() => conv2d(image, kernel), [image, kernel]);
  const pooled = useMemo(() => maxPool2x2(convMap), [convMap]);
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
        sum += patch[y][x] * kernel[y][x];
      }
    }
    return Math.abs(sum);
  }, [patch, kernel]);

  const droppedMask = useMemo(() => {
    return vector.map((_, idx) => randUnit(iteration * 1000 + idx * 37) > dropRate);
  }, [vector, iteration, dropRate]);

  const keptCount = droppedMask.filter(Boolean).length;
  const displayUnits = 42;
  const convMax = gridMax(convMap);
  const poolMax = gridMax(pooled);

  return (
    <section>
      <h2>CNN Feature Flow Visualizer</h2>
      <p className="subtext">
        Watch a convolution kernel slide across an image, then see spatial compression (conv to pool to flatten) and dropout masks across training iterations. The feature map brightness shows edge-response strength, not raw pixels.
      </p>

      <div className="explain-card">
        <strong>How to explain this slide</strong>
        <span>1) Kernel scans local patches and computes a weighted sum.</span>
        <span>2) Each feature-map cell stores pattern strength at that location (not pixel value).</span>
        <span>3) Pooling keeps strongest local responses, shrinking spatial size.</span>
        <span>4) Dropout randomly disables units per iteration to improve generalization.</span>
      </div>

      <div className="cnn-layout">
        <div className="cnn-main">
          <div>
            <h3>Input Image + Sliding Kernel</h3>
            <div
              className="cnn-grid"
              style={{ gridTemplateColumns: `repeat(${IMAGE_SIZE}, 1fr)` }}
            >
              {image.flatMap((row, y) =>
                row.map((value, x) => {
                  const inWindow =
                    x >= scanPos.x &&
                    x < scanPos.x + KERNEL_SIZE &&
                    y >= scanPos.y &&
                    y < scanPos.y + KERNEL_SIZE;
                  return (
                    <div
                      key={`${x}-${y}`}
                      className={inWindow ? "cnn-cell cnn-window" : "cnn-cell"}
                      style={{ background: `rgba(148, 218, 255, ${0.08 + value * 0.92})` }}
                    />
                  );
                }),
              )}
            </div>
          </div>

          <div>
            <h3>Conv Feature Map (14x14, absolute response)</h3>
            <div className="cnn-grid" style={{ gridTemplateColumns: `repeat(${convMap.length}, 1fr)` }}>
              {convMap.flatMap((row, y) =>
                row.map((value, x) => {
                  const isFocus = x === scanPos.x + 1 && y === scanPos.y + 1;
                  return (
                    <div
                      key={`conv-${x}-${y}`}
                      className={isFocus ? "cnn-cell cnn-cell-focus" : "cnn-cell"}
                      style={{ background: `rgba(255, 204, 128, ${0.06 + (value / convMax) * 0.94})` }}
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
              {IMAGE_SIZE}x{IMAGE_SIZE} to {convMap.length}x{convMap.length} to {pooled.length}x{pooled.length} to {vector.length}
            </div>
          </div>
        </div>

        <div className="controls">
          <div className="preset-row">
            <button className="ghost-btn" onClick={() => setScanPlaying((v) => !v)}>
              {scanPlaying ? "Pause Scan" : "Play Scan"}
            </button>
            <button className="ghost-btn" onClick={() => setIterPlaying((v) => !v)}>
              {iterPlaying ? "Pause Training" : "Play Training"}
            </button>
          </div>

          <label>
            Scan speed: {scanSpeed} ms
            <input type="range" min={60} max={400} step={10} value={scanSpeed} onChange={(e) => setScanSpeed(Number(e.target.value))} />
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
            Presenter notes
            <br />
            - Bright in conv map means strong edge-like pattern match.
            <br />
            - Dim means weak/no match.
            <br />
            - Spatial info shrinks, but key features remain.
          </div>

          <div className="formula-block">
            Dropout mask sample
            <div className="cnn-dropout-row">
              {vector.slice(0, displayUnits).map((v, i) => {
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
