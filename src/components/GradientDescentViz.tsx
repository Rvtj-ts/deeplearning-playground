import { useEffect, useMemo, useState } from "react";

type Vec2 = { x: number; y: number };
type Vec3 = { x: number; y: number; z: number };

const WIDTH = 560;
const HEIGHT = 430;
const DOMAIN = 4;
const CHART_W = 260;
const CHART_H = 92;

function loss(x: number, y: number) {
  return 0.8 * x * x + 2.4 * y * y + 0.6 * x * y + 1.5 * Math.sin(0.8 * x) * Math.cos(0.8 * y);
}

function grad(x: number, y: number): Vec2 {
  return {
    x: 1.6 * x + 0.6 * y + 1.2 * Math.cos(0.8 * x) * Math.cos(0.8 * y),
    y: 4.8 * y + 0.6 * x - 1.2 * Math.sin(0.8 * x) * Math.sin(0.8 * y),
  };
}

function runDescent(start: Vec2, lr: number, steps: number): Vec3[] {
  const path: Vec3[] = [{ x: start.x, y: start.y, z: loss(start.x, start.y) }];
  let p = { ...start };
  for (let i = 0; i < steps; i += 1) {
    const g = grad(p.x, p.y);
    p = { x: p.x - lr * g.x, y: p.y - lr * g.y };
    path.push({ x: p.x, y: p.y, z: loss(p.x, p.y) });
  }
  return path;
}

function project(point: Vec3, yaw: number, pitch: number, zMin: number, zMax: number) {
  const yRad = (yaw * Math.PI) / 180;
  const pRad = (pitch * Math.PI) / 180;
  const zNorm = ((point.z - zMin) / (zMax - zMin + 1e-9)) * 3.2 - 1.6;

  const xYaw = point.x * Math.cos(yRad) - point.y * Math.sin(yRad);
  const yYaw = point.x * Math.sin(yRad) + point.y * Math.cos(yRad);

  const yPitch = yYaw * Math.cos(pRad) - zNorm * Math.sin(pRad);
  const zPitch = yYaw * Math.sin(pRad) + zNorm * Math.cos(pRad);

  return {
    sx: WIDTH / 2 + xYaw * 42,
    sy: HEIGHT * 0.62 - yPitch * 42,
    depth: zPitch,
  };
}

export function GradientDescentViz() {
  const [startX, setStartX] = useState(3);
  const [startY, setStartY] = useState(-2.2);
  const [lr, setLr] = useState(0.08);
  const [steps, setSteps] = useState(28);
  const [yaw, setYaw] = useState(-32);
  const [pitch, setPitch] = useState(58);
  const [isPlaying, setIsPlaying] = useState(true);
  const [speedMs, setSpeedMs] = useState(180);
  const [visibleStep, setVisibleStep] = useState(0);

  const path = useMemo(() => runDescent({ x: startX, y: startY }, lr, steps), [startX, startY, lr, steps]);

  const surface = useMemo(() => {
    const n = 24;
    const points: Vec3[][] = [];
    let zMin = Number.POSITIVE_INFINITY;
    let zMax = Number.NEGATIVE_INFINITY;

    for (let iy = 0; iy < n; iy += 1) {
      const row: Vec3[] = [];
      const y = -DOMAIN + (2 * DOMAIN * iy) / (n - 1);
      for (let ix = 0; ix < n; ix += 1) {
        const x = -DOMAIN + (2 * DOMAIN * ix) / (n - 1);
        const z = loss(x, y);
        row.push({ x, y, z });
        zMin = Math.min(zMin, z);
        zMax = Math.max(zMax, z);
      }
      points.push(row);
    }

    return { points, zMin, zMax };
  }, []);

  const meshPaths = useMemo(() => {
    const lines: Array<{ d: string; depth: number }> = [];
    const n = surface.points.length;

    for (let r = 0; r < n; r += 1) {
      const projected = surface.points[r].map((p) => project(p, yaw, pitch, surface.zMin, surface.zMax));
      const d = projected.map((p, i) => `${i === 0 ? "M" : "L"} ${p.sx} ${p.sy}`).join(" ");
      const depth = projected.reduce((acc, p) => acc + p.depth, 0) / projected.length;
      lines.push({ d, depth });
    }

    for (let c = 0; c < n; c += 1) {
      const col = surface.points.map((row) => row[c]);
      const projected = col.map((p) => project(p, yaw, pitch, surface.zMin, surface.zMax));
      const d = projected.map((p, i) => `${i === 0 ? "M" : "L"} ${p.sx} ${p.sy}`).join(" ");
      const depth = projected.reduce((acc, p) => acc + p.depth, 0) / projected.length;
      lines.push({ d, depth });
    }

    lines.sort((a, b) => a.depth - b.depth);
    return lines;
  }, [surface, yaw, pitch]);

  useEffect(() => {
    setVisibleStep(0);
  }, [startX, startY, lr, steps]);

  useEffect(() => {
    if (!isPlaying) return;
    const timer = window.setInterval(() => {
      setVisibleStep((s) => {
        if (s >= path.length - 1) return 0;
        return s + 1;
      });
    }, speedMs);
    return () => window.clearInterval(timer);
  }, [isPlaying, speedMs, path.length]);

  const activePath = path.slice(0, visibleStep + 1);
  const deltas = useMemo(() => {
    return path.map((p, i) => {
      if (i === 0) return 0;
      return p.z - path[i - 1].z;
    });
  }, [path]);

  const activePathD = activePath
    .map((p, i) => {
      const q = project(p, yaw, pitch, surface.zMin, surface.zMax);
      return `${i === 0 ? "M" : "L"} ${q.sx} ${q.sy}`;
    })
    .join(" ");

  const current = path[visibleStep];
  const currentProjected = project(current, yaw, pitch, surface.zMin, surface.zMax);
  const final = path[path.length - 1];
  const currentDelta = deltas[visibleStep] ?? 0;

  const zValues = path.map((p) => p.z);
  const zMin = Math.min(...zValues);
  const zMax = Math.max(...zValues);
  const lossChartD = path
    .map((p, i) => {
      const x = (i / Math.max(1, path.length - 1)) * CHART_W;
      const y = CHART_H - ((p.z - zMin) / (zMax - zMin + 1e-9)) * CHART_H;
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");
  const markerX = (visibleStep / Math.max(1, path.length - 1)) * CHART_W;
  const markerY =
    CHART_H -
    ((path[visibleStep].z - zMin) / (zMax - zMin + 1e-9)) * CHART_H;

  return (
    <section>
      <h2>Gradient Descent Visualizer</h2>
      <p className="subtext">
        Animated 3D loss surface view: watch the optimizer descend the landscape step-by-step, then adjust camera angle to explain trajectory behavior.
      </p>

      <div className="viz-layout">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="viz-canvas">
          <rect width={WIDTH} height={HEIGHT} fill="#0b1720" />

          {meshPaths.map((line, idx) => (
            <path key={idx} d={line.d} fill="none" stroke="#2c4f64" strokeWidth="1" opacity="0.88" />
          ))}

          <path d={activePathD} stroke="#ffe39a" strokeWidth="3" fill="none" />
          {activePath.map((p, idx) => {
            const q = project(p, yaw, pitch, surface.zMin, surface.zMax);
            const delta = deltas[idx] ?? 0;
            const fill = idx === 0 ? "#fff4cb" : delta <= 0 ? "#7de6a9" : "#ff8f8f";
            return <circle key={idx} cx={q.sx} cy={q.sy} r={idx === visibleStep ? 5 : 2.9} fill={fill} />;
          })}

          <circle cx={currentProjected.sx} cy={currentProjected.sy} r={8} fill="none" stroke="#ffd27f" strokeWidth="2" opacity="0.8" />
        </svg>

        <div className="controls">
          <div className="preset-row">
            <button className="ghost-btn" onClick={() => setIsPlaying((v) => !v)}>
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button className="ghost-btn" onClick={() => setVisibleStep(0)}>
              Reset
            </button>
            <button
              className="ghost-btn"
              onClick={() => {
                setIsPlaying(false);
                setVisibleStep((s) => Math.max(0, s - 1));
              }}
            >
              Prev Step
            </button>
            <button
              className="ghost-btn"
              onClick={() => {
                setIsPlaying(false);
                setVisibleStep((s) => Math.min(path.length - 1, s + 1));
              }}
            >
              Next Step
            </button>
          </div>

          <label>
            Animation Speed: {speedMs} ms/step
            <input type="range" min={60} max={450} step={10} value={speedMs} onChange={(e) => setSpeedMs(Number(e.target.value))} />
          </label>

          <label>
            Start x: {startX.toFixed(2)}
            <input type="range" min={-3.8} max={3.8} step={0.1} value={startX} onChange={(e) => setStartX(Number(e.target.value))} />
          </label>

          <label>
            Start y: {startY.toFixed(2)}
            <input type="range" min={-3.8} max={3.8} step={0.1} value={startY} onChange={(e) => setStartY(Number(e.target.value))} />
          </label>

          <label>
            Learning rate: {lr.toFixed(3)}
            <input type="range" min={0.01} max={0.22} step={0.005} value={lr} onChange={(e) => setLr(Number(e.target.value))} />
          </label>

          <label>
            Steps: {steps}
            <input type="range" min={8} max={70} step={1} value={steps} onChange={(e) => setSteps(Number(e.target.value))} />
          </label>

          <label>
            Camera Yaw: {yaw} deg
            <input type="range" min={-80} max={80} step={1} value={yaw} onChange={(e) => setYaw(Number(e.target.value))} />
          </label>

          <label>
            Camera Pitch: {pitch} deg
            <input type="range" min={35} max={80} step={1} value={pitch} onChange={(e) => setPitch(Number(e.target.value))} />
          </label>

          <label>
            Visible Step: {visibleStep}/{path.length - 1}
            <input
              type="range"
              min={0}
              max={path.length - 1}
              step={1}
              value={visibleStep}
              onChange={(e) => {
                setIsPlaying(false);
                setVisibleStep(Number(e.target.value));
              }}
            />
          </label>

          <div className="formula-block">
            Current loss: {current.z.toFixed(4)}
            <br />
            Delta loss (this step): {currentDelta >= 0 ? "+" : ""}
            {currentDelta.toFixed(4)}
            <br />
            Final loss: {final.z.toFixed(4)}
            <br />
            Position: ({current.x.toFixed(2)}, {current.y.toFixed(2)})
          </div>

          <div className="formula-block loss-chart-block">
            Loss by step (green = down, red = up)
            <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="loss-chart">
              <rect x="0" y="0" width={CHART_W} height={CHART_H} fill="#0a1a24" />
              <path d={lossChartD} fill="none" stroke="#93d8ff" strokeWidth="2" />
              {path.map((p, i) => {
                const x = (i / Math.max(1, path.length - 1)) * CHART_W;
                const y = CHART_H - ((p.z - zMin) / (zMax - zMin + 1e-9)) * CHART_H;
                const delta = deltas[i] ?? 0;
                const fill = i === 0 ? "#fff4cb" : delta <= 0 ? "#7de6a9" : "#ff8f8f";
                return <circle key={i} cx={x} cy={y} r={i === visibleStep ? 3.4 : 2.1} fill={fill} />;
              })}
              <line x1={markerX} y1={0} x2={markerX} y2={CHART_H} stroke="#ffd27f" strokeDasharray="3 3" />
              <circle cx={markerX} cy={markerY} r={4.1} fill="none" stroke="#ffd27f" strokeWidth="1.5" />
            </svg>
          </div>
        </div>
      </div>
    </section>
  );
}
