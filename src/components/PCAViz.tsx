import { useEffect, useMemo, useState } from "react";

const SCATTER_W = 520;
const SCATTER_H = 360;

type PresetKey = "6" | "12" | "14" | "18" | "30";

type PcaArtifact = {
  meta: {
    dim: number;
    imageSide: number;
    trainSize: number;
    testSize: number;
    presetComponents: number[];
    knnNeighbors: number;
  };
  testLabels: number[];
  meanVector: number[];
  testVectors: number[][];
  reconstructions: Record<PresetKey, number[][]>;
  scatter: Array<{ x: number; y: number; label: number }>;
  eigendigits: number[][];
  pcStd: number[];
  explained: Record<PresetKey, number>;
  explainedCumulative: number[];
  knnAccuracy: Record<PresetKey, number>;
};

let pcaDataCache: PcaArtifact | null = null;
let pcaDataPromise: Promise<PcaArtifact> | null = null;

function loadPcaArtifact() {
  if (pcaDataCache) {
    return Promise.resolve(pcaDataCache);
  }
  if (pcaDataPromise) {
    return pcaDataPromise;
  }

  pcaDataPromise = fetch("/data/pca-presets.json")
    .then((response) => {
      if (!response.ok) {
        throw new Error(`Failed to load PCA presets (${response.status})`);
      }
      return response.json() as Promise<PcaArtifact>;
    })
    .then((artifact) => {
      pcaDataCache = artifact;
      return artifact;
    })
    .catch((error) => {
      pcaDataPromise = null;
      throw error;
    });

  return pcaDataPromise;
}

const presets: Array<{ key: PresetKey; label: string }> = [
  { key: "6", label: "6 PCs" },
  { key: "12", label: "12 PCs" },
  { key: "14", label: "14 PCs" },
  { key: "18", label: "18 PCs" },
  { key: "30", label: "All (30 PCs)" },
];

const colorScale = [
  "#66d9ef",
  "#ffce7a",
  "#9be07d",
  "#ff8d8d",
  "#bfa8ff",
  "#ffd66b",
  "#7ed5b4",
  "#f4a0d7",
  "#8ab4ff",
  "#f8bf86",
];

function renderImage(vec: number[], className: string) {
  return (
    <div className="digit28-grid" aria-hidden="true">
      {vec.map((v, i) => (
        <div key={i} className={className} style={{ opacity: 0.05 + v * 0.95 }} />
      ))}
    </div>
  );
}

function renderEigen(comp: number[]) {
  const maxAbs = Math.max(1e-9, ...comp.map((v) => Math.abs(v)));
  return (
    <div className="digit28-grid eigen-grid" aria-hidden="true">
      {comp.map((v, i) => {
        const strength = Math.min(1, Math.abs(v) / maxAbs);
        const color = v >= 0 ? `rgba(255, 186, 118, ${0.2 + strength * 0.8})` : `rgba(118, 196, 255, ${0.2 + strength * 0.8})`;
        return (
          <div
            key={i}
            className="eigen-cell"
            style={{
              background: color,
            }}
          />
        );
      })}
    </div>
  );
}

function clamp01(v: number) {
  return Math.max(0, Math.min(1, v));
}

function addScaled(base: number[], direction: number[], scale: number) {
  return base.map((v, i) => clamp01(v + direction[i] * scale));
}

export function PCAViz() {
  const [data, setData] = useState<PcaArtifact | null>(() => pcaDataCache);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [preset, setPreset] = useState<PresetKey>("12");
  const [sampleIndex, setSampleIndex] = useState(0);
  const [selectedPc, setSelectedPc] = useState(0);

  useEffect(() => {
    let active = true;

    loadPcaArtifact()
      .then((artifact) => {
        if (active) {
          setData(artifact);
        }
      })
      .catch((error: unknown) => {
        if (active) {
          setLoadError(error instanceof Error ? error.message : "Failed to load PCA presets");
        }
      });

    return () => {
      active = false;
    };
  }, []);

  const xVals = useMemo(() => (data ? data.scatter.map((p) => p.x) : []), [data]);
  const yVals = useMemo(() => (data ? data.scatter.map((p) => p.y) : []), [data]);

  if (loadError) {
    return (
      <section>
        <h2>PCA on Handwritten Digits</h2>
        <p className="subtext">Could not load PCA presets: {loadError}</p>
      </section>
    );
  }

  if (!data) {
    return (
      <section>
        <h2>PCA on Handwritten Digits</h2>
        <p className="subtext">Loading PCA presets...</p>
      </section>
    );
  }

  const sampleCount = data.testVectors.length;
  const selectedIndex = Math.min(sampleIndex, sampleCount - 1);

  const original = data.testVectors[selectedIndex];
  const reconstructed = data.reconstructions[preset][selectedIndex];
  const label = data.testLabels[selectedIndex];

  const xMin = Math.min(...xVals);
  const xMax = Math.max(...xVals);
  const yMin = Math.min(...yVals);
  const yMax = Math.max(...yVals);
  const pc = data.eigendigits[selectedPc];
  const sigma = data.pcStd[selectedPc] ?? 0.1;
  const plusImage = addScaled(data.meanVector, pc, sigma * 2.2);
  const minusImage = addScaled(data.meanVector, pc, -sigma * 2.2);

  return (
    <section>
      <h2>PCA on Handwritten Digits</h2>
      <p className="subtext">
        This view uses precomputed PCA presets so the demo stays fast: 784D images compressed to 6, 12, 14, 18, or 30 principal components.
      </p>

      <div className="digit-layout">
        <div>
          <svg viewBox={`0 0 ${SCATTER_W} ${SCATTER_H}`} className="viz-canvas">
            <rect width={SCATTER_W} height={SCATTER_H} fill="#0c1c27" />
            <line x1={20} y1={SCATTER_H / 2} x2={SCATTER_W - 20} y2={SCATTER_H / 2} stroke="#36586b" />
            <line x1={SCATTER_W / 2} y1={20} x2={SCATTER_W / 2} y2={SCATTER_H - 20} stroke="#36586b" />

            {data.scatter.map((p, idx) => {
              const x = 30 + ((p.x - xMin) / (xMax - xMin + 1e-9)) * (SCATTER_W - 60);
              const y = 30 + ((p.y - yMin) / (yMax - yMin + 1e-9)) * (SCATTER_H - 60);
              return (
                <circle
                  key={idx}
                  cx={x}
                  cy={SCATTER_H - y}
                  r={2.8}
                  fill={colorScale[p.label]}
                  opacity={0.78}
                />
              );
            })}
          </svg>

          <div className="digit-legend">
            {colorScale.map((c, i) => (
              <span key={i}>
                <i style={{ background: c }} /> {i}
              </span>
            ))}
          </div>

          <div className="eigen-section">
            <h3>Top Eigendigits</h3>
            <p className="eigen-help">Blue pixels pull intensity down, amber pixels push it up when moving along that principal component.</p>
            <div className="eigen-list">
              {data.eigendigits.slice(0, 12).map((comp, idx) => (
                <button
                  key={idx}
                  className={idx === selectedPc ? "eigen-card eigen-card-active" : "eigen-card"}
                  onClick={() => setSelectedPc(idx)}
                  type="button"
                >
                  <strong>PC{idx + 1}</strong>
                  {renderEigen(comp)}
                  <small>{(data.explainedCumulative[idx] * 100).toFixed(1)}% cumulative</small>
                </button>
              ))}
            </div>

            <div className="eigen-shift">
              <div>
                <h4>Mean - 2.2 sigma * PC{selectedPc + 1}</h4>
                {renderImage(minusImage, "digit-cell")}
              </div>
              <div>
                <h4>Dataset Mean</h4>
                {renderImage(data.meanVector, "digit-cell")}
              </div>
              <div>
                <h4>Mean + 2.2 sigma * PC{selectedPc + 1}</h4>
                {renderImage(plusImage, "digit-cell")}
              </div>
            </div>

            <div className="eigen-polarity">
              <span>- direction</span>
              <div className="eigen-polarity-bar" aria-hidden="true" />
              <span>+ direction</span>
            </div>
          </div>
        </div>

        <div className="controls">
          <div className="digit-compare">
            <div>
              <h3>Original</h3>
              {renderImage(original, "digit-cell")}
            </div>
            <div>
              <h3>Reconstruction</h3>
              {renderImage(reconstructed, "digit-cell recon")}
            </div>
          </div>

          <div className="preset-row">
            {presets.map((item) => (
              <button
                key={item.key}
                className={item.key === preset ? "tab tab-active" : "tab"}
                onClick={() => setPreset(item.key)}
              >
                {item.label}
              </button>
            ))}
          </div>

          <label>
            Test Sample Index: {selectedIndex} (digit {label})
            <input
              type="range"
              min={0}
              max={sampleCount - 1}
              step={1}
              value={selectedIndex}
              onChange={(e) => setSampleIndex(Number(e.target.value))}
            />
          </label>

          <div className="formula-block">
            Train/Test: {data.meta.trainSize}/{data.meta.testSize}
            <br />
            Compression: {data.meta.dim} to {preset} ({((Number(preset) / data.meta.dim) * 100).toFixed(2)}%)
            <br />
            Explained variance: {(data.explained[preset] * 100).toFixed(1)}%
            <br />
            k-NN ({data.meta.knnNeighbors}) accuracy in PCA space: {(data.knnAccuracy[preset] * 100).toFixed(1)}%
          </div>
        </div>
      </div>
    </section>
  );
}
