import { useEffect, useMemo, useState } from "react";

type Vec2 = { x: number; y: number };
type Vec3 = { x: number; y: number; z: number };
type Projected = { sx: number; sy: number; depth: number };

const WIDTH = 620;
const HEIGHT = 460;
const DOMAIN = 1.0;
const GRID_N = 22;
const SCALE = 170;
const CHART_W = 260;
const CHART_H = 92;

// Asymmetric bowl: curvature in y is larger than in x so SGD oscillates
// in the steep direction while Adam's per-parameter adaptive step size
// glides smoothly to the minimum. At (±1, ±1) the loss equals 2 so the
// overall shape is visually close to a paraboloid z = x² + y².
const LOSS_A = 0.6;
const LOSS_B = 1.4;

function loss(x: number, y: number) {
  return LOSS_A * x * x + LOSS_B * y * y;
}

function grad(x: number, y: number): Vec2 {
  return { x: 2 * LOSS_A * x, y: 2 * LOSS_B * y };
}

function runSGD(start: Vec2, lr: number, steps: number): Vec3[] {
  const path: Vec3[] = [{ x: start.x, y: start.y, z: loss(start.x, start.y) }];
  let p = { ...start };
  for (let i = 0; i < steps; i += 1) {
    const g = grad(p.x, p.y);
    p = { x: p.x - lr * g.x, y: p.y - lr * g.y };
    path.push({ x: p.x, y: p.y, z: loss(p.x, p.y) });
  }
  return path;
}

function runAdam(
  start: Vec2,
  lr: number,
  steps: number,
  beta1: number,
  beta2: number,
): Vec3[] {
  const eps = 1e-8;
  const path: Vec3[] = [{ x: start.x, y: start.y, z: loss(start.x, start.y) }];
  let p = { ...start };
  let mx = 0;
  let my = 0;
  let vx = 0;
  let vy = 0;
  for (let t = 1; t <= steps; t += 1) {
    const g = grad(p.x, p.y);
    mx = beta1 * mx + (1 - beta1) * g.x;
    my = beta1 * my + (1 - beta1) * g.y;
    vx = beta2 * vx + (1 - beta2) * g.x * g.x;
    vy = beta2 * vy + (1 - beta2) * g.y * g.y;
    const b1c = 1 - Math.pow(beta1, t);
    const b2c = 1 - Math.pow(beta2, t);
    const mxHat = mx / b1c;
    const myHat = my / b1c;
    const vxHat = vx / b2c;
    const vyHat = vy / b2c;
    p = {
      x: p.x - (lr * mxHat) / (Math.sqrt(vxHat) + eps),
      y: p.y - (lr * myHat) / (Math.sqrt(vyHat) + eps),
    };
    path.push({ x: p.x, y: p.y, z: loss(p.x, p.y) });
  }
  return path;
}

// Orthographic projection with yaw (around z) then pitch (around x).
// Pitch is negative so the camera tilts downward, matching the reference
// view in which the bowl opens upward toward the viewer.
function project(point: Vec3, yaw: number, pitch: number): Projected {
  const yr = (yaw * Math.PI) / 180;
  const pr = (pitch * Math.PI) / 180;
  // Compress z so the bowl's vertical extent is comparable to x/y.
  const zv = point.z * 0.55;

  const xR = point.x * Math.cos(yr) - point.y * Math.sin(yr);
  const yR = point.x * Math.sin(yr) + point.y * Math.cos(yr);

  const yT = yR * Math.cos(pr) - zv * Math.sin(pr);
  const zT = yR * Math.sin(pr) + zv * Math.cos(pr);

  return {
    sx: WIDTH * 0.5 + xR * SCALE,
    sy: HEIGHT * 0.6 - yT * SCALE,
    depth: zT,
  };
}

// Coolwarm-like colormap: deep blue → white → deep red.
function colorFromZ(z: number, zMin: number, zMax: number): string {
  const t = Math.max(0, Math.min(1, (z - zMin) / (zMax - zMin + 1e-9)));
  const c1 = [59, 76, 192];
  const c2 = [221, 221, 221];
  const c3 = [180, 4, 38];
  let rgb: number[];
  if (t < 0.5) {
    const u = t * 2;
    rgb = c1.map((a, i) => Math.round(a + (c2[i] - a) * u));
  } else {
    const u = (t - 0.5) * 2;
    rgb = c2.map((a, i) => Math.round(a + (c3[i] - a) * u));
  }
  return `rgb(${rgb[0]},${rgb[1]},${rgb[2]})`;
}

export function GradientDescentViz() {
  const [startX, setStartX] = useState(-0.85);
  const [startY, setStartY] = useState(0.85);
  const [lr, setLr] = useState(0.66);
  const [adamLr, setAdamLr] = useState(0.12);
  const [steps, setSteps] = useState(40);
  const [beta1, setBeta1] = useState(0.9);
  const [beta2, setBeta2] = useState(0.999);
  const [yaw, setYaw] = useState(-38);
  const [pitch, setPitch] = useState(-55);
  const [showSGD, setShowSGD] = useState(true);
  const [showAdam, setShowAdam] = useState(true);
  const [isPlaying, setIsPlaying] = useState(true);
  const [speedMs, setSpeedMs] = useState(180);
  const [visibleStep, setVisibleStep] = useState(0);

  const sgdPath = useMemo(
    () => runSGD({ x: startX, y: startY }, lr, steps),
    [startX, startY, lr, steps],
  );
  const adamPath = useMemo(
    () => runAdam({ x: startX, y: startY }, adamLr, steps, beta1, beta2),
    [startX, startY, adamLr, steps, beta1, beta2],
  );

  const surface = useMemo(() => {
    const pts: Vec3[][] = [];
    let zMin = Number.POSITIVE_INFINITY;
    let zMax = Number.NEGATIVE_INFINITY;
    for (let iy = 0; iy < GRID_N; iy += 1) {
      const row: Vec3[] = [];
      const y = -DOMAIN + (2 * DOMAIN * iy) / (GRID_N - 1);
      for (let ix = 0; ix < GRID_N; ix += 1) {
        const x = -DOMAIN + (2 * DOMAIN * ix) / (GRID_N - 1);
        const z = loss(x, y);
        row.push({ x, y, z });
        if (z < zMin) zMin = z;
        if (z > zMax) zMax = z;
      }
      pts.push(row);
    }
    return { pts, zMin, zMax };
  }, []);

  const quads = useMemo(() => {
    const list: { d: string; fill: string; depth: number }[] = [];
    const N = surface.pts.length;
    for (let iy = 0; iy < N - 1; iy += 1) {
      for (let ix = 0; ix < N - 1; ix += 1) {
        const p00 = surface.pts[iy][ix];
        const p10 = surface.pts[iy][ix + 1];
        const p11 = surface.pts[iy + 1][ix + 1];
        const p01 = surface.pts[iy + 1][ix];
        const q00 = project(p00, yaw, pitch);
        const q10 = project(p10, yaw, pitch);
        const q11 = project(p11, yaw, pitch);
        const q01 = project(p01, yaw, pitch);
        const avgZ = (p00.z + p10.z + p11.z + p01.z) / 4;
        const avgD = (q00.depth + q10.depth + q11.depth + q01.depth) / 4;
        const fill = colorFromZ(avgZ, surface.zMin, surface.zMax);
        const d = `M ${q00.sx.toFixed(1)} ${q00.sy.toFixed(1)} L ${q10.sx.toFixed(1)} ${q10.sy.toFixed(1)} L ${q11.sx.toFixed(1)} ${q11.sy.toFixed(1)} L ${q01.sx.toFixed(1)} ${q01.sy.toFixed(1)} Z`;
        list.push({ d, fill, depth: avgD });
      }
    }
    list.sort((a, b) => a.depth - b.depth);
    return list;
  }, [surface, yaw, pitch]);

  useEffect(() => {
    setVisibleStep(0);
  }, [startX, startY, lr, adamLr, steps, beta1, beta2]);

  const maxStep = Math.max(sgdPath.length, adamPath.length) - 1;

  useEffect(() => {
    if (!isPlaying) return;
    const id = window.setInterval(() => {
      setVisibleStep((s) => (s >= maxStep ? 0 : s + 1));
    }, speedMs);
    return () => window.clearInterval(id);
  }, [isPlaying, speedMs, maxStep]);

  const sgdProj = useMemo(
    () => sgdPath.map((p) => project(p, yaw, pitch)),
    [sgdPath, yaw, pitch],
  );
  const adamProj = useMemo(
    () => adamPath.map((p) => project(p, yaw, pitch)),
    [adamPath, yaw, pitch],
  );

  const sgdVisible = sgdProj.slice(0, visibleStep + 1);
  const adamVisible = adamProj.slice(0, visibleStep + 1);

  const sgdPathD = sgdVisible
    .map((q, i) => `${i === 0 ? "M" : "L"} ${q.sx.toFixed(1)} ${q.sy.toFixed(1)}`)
    .join(" ");
  const adamPathD = adamVisible
    .map((q, i) => `${i === 0 ? "M" : "L"} ${q.sx.toFixed(1)} ${q.sy.toFixed(1)}`)
    .join(" ");

  const sgdIdx = Math.min(visibleStep, sgdPath.length - 1);
  const adamIdx = Math.min(visibleStep, adamPath.length - 1);
  const sgdCur = sgdPath[sgdIdx];
  const adamCur = adamPath[adamIdx];
  const sgdFinal = sgdPath[sgdPath.length - 1];
  const adamFinal = adamPath[adamPath.length - 1];

  const chartMax = Math.max(
    1e-9,
    ...sgdPath.map((p) => p.z),
    ...adamPath.map((p) => p.z),
  );
  const mkChartD = (path: Vec3[]) =>
    path
      .map((p, i) => {
        const x = (i / Math.max(1, path.length - 1)) * CHART_W;
        const y = CHART_H - (p.z / chartMax) * CHART_H;
        return `${i === 0 ? "M" : "L"} ${x.toFixed(1)} ${y.toFixed(1)}`;
      })
      .join(" ");
  const sgdChartD = mkChartD(sgdPath);
  const adamChartD = mkChartD(adamPath);
  const markerX = (visibleStep / Math.max(1, maxStep)) * CHART_W;

  const startProj = project(
    { x: startX, y: startY, z: loss(startX, startY) },
    yaw,
    pitch,
  );

  return (
    <section>
      <h2>Gradient Descent &amp; Adam Optimizer</h2>
      <p className="subtext">
        Watch <span style={{ color: "#ffbd4a" }}>SGD</span> and{" "}
        <span style={{ color: "#4fd0ff" }}>Adam</span> race down an asymmetric
        loss bowl. SGD uses a single learning rate and oscillates in the
        steep y direction; Adam adapts its step size per parameter and glides
        straight toward the minimum.
      </p>

      <div className="viz-layout">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="viz-canvas">
          <rect width={WIDTH} height={HEIGHT} fill="#f5f7fb" />

          {quads.map((q, i) => (
            <path
              key={i}
              d={q.d}
              fill={q.fill}
              stroke={q.fill}
              strokeWidth="0.6"
              strokeLinejoin="round"
            />
          ))}

          {showSGD && sgdVisible.length > 1 && (
            <path
              d={sgdPathD}
              fill="none"
              stroke="#e89417"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
          {showSGD &&
            sgdVisible.map((q, i) => (
              <circle
                key={`s${i}`}
                cx={q.sx}
                cy={q.sy}
                r={i === sgdIdx ? 5.2 : 2.5}
                fill="#ffbd4a"
                stroke="#3a2300"
                strokeWidth={i === sgdIdx ? 1.5 : 0.5}
              />
            ))}

          {showAdam && adamVisible.length > 1 && (
            <path
              d={adamPathD}
              fill="none"
              stroke="#1798c4"
              strokeWidth="3"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}
          {showAdam &&
            adamVisible.map((q, i) => (
              <circle
                key={`a${i}`}
                cx={q.sx}
                cy={q.sy}
                r={i === adamIdx ? 5.2 : 2.5}
                fill="#4fd0ff"
                stroke="#002230"
                strokeWidth={i === adamIdx ? 1.5 : 0.5}
              />
            ))}

          <g>
            <line
              x1={startProj.sx - 8}
              y1={startProj.sy - 8}
              x2={startProj.sx + 8}
              y2={startProj.sy + 8}
              stroke="#d62525"
              strokeWidth="2.6"
              strokeLinecap="round"
            />
            <line
              x1={startProj.sx + 8}
              y1={startProj.sy - 8}
              x2={startProj.sx - 8}
              y2={startProj.sy + 8}
              stroke="#d62525"
              strokeWidth="2.6"
              strokeLinecap="round"
            />
          </g>

          <g>
            <rect
              x={WIDTH - 162}
              y={14}
              width={148}
              height={60}
              rx={8}
              fill="rgba(255,255,255,0.88)"
              stroke="#8aa0b4"
            />
            <circle cx={WIDTH - 148} cy={32} r={5} fill="#ffbd4a" stroke="#3a2300" />
            <text x={WIDTH - 136} y={36} fontSize="11" fill="#102430">
              SGD — fixed η
            </text>
            <circle cx={WIDTH - 148} cy={56} r={5} fill="#4fd0ff" stroke="#002230" />
            <text x={WIDTH - 136} y={60} fontSize="11" fill="#102430">
              Adam — β₁, β₂
            </text>
          </g>

          <text x={16} y={26} fontSize="12" fill="#2b3a4a">
            L(x, y) = {LOSS_A.toFixed(2)} x² + {LOSS_B.toFixed(2)} y²
          </text>
        </svg>

        <div className="controls">
          <div className="preset-row">
            <button className="ghost-btn" onClick={() => setIsPlaying((v) => !v)}>
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button
              className="ghost-btn"
              onClick={() => {
                setIsPlaying(false);
                setVisibleStep(0);
              }}
            >
              Reset
            </button>
            <button
              className="ghost-btn"
              onClick={() => {
                setIsPlaying(false);
                setVisibleStep((s) => Math.max(0, s - 1));
              }}
            >
              Prev
            </button>
            <button
              className="ghost-btn"
              onClick={() => {
                setIsPlaying(false);
                setVisibleStep((s) => Math.min(maxStep, s + 1));
              }}
            >
              Next
            </button>
          </div>

          <div className="preset-row">
            <label style={{ display: "inline-flex", gap: "0.3rem", alignItems: "center", fontSize: "0.85rem" }}>
              <input
                type="checkbox"
                checked={showSGD}
                onChange={(e) => setShowSGD(e.target.checked)}
              />
              Show SGD
            </label>
            <label style={{ display: "inline-flex", gap: "0.3rem", alignItems: "center", fontSize: "0.85rem" }}>
              <input
                type="checkbox"
                checked={showAdam}
                onChange={(e) => setShowAdam(e.target.checked)}
              />
              Show Adam
            </label>
          </div>

          <label>
            Animation speed: {speedMs} ms/step
            <input
              type="range"
              min={60}
              max={500}
              step={10}
              value={speedMs}
              onChange={(e) => setSpeedMs(Number(e.target.value))}
            />
          </label>

          <label>
            Start x: {startX.toFixed(2)}
            <input
              type="range"
              min={-0.95}
              max={0.95}
              step={0.05}
              value={startX}
              onChange={(e) => setStartX(Number(e.target.value))}
            />
          </label>

          <label>
            Start y: {startY.toFixed(2)}
            <input
              type="range"
              min={-0.95}
              max={0.95}
              step={0.05}
              value={startY}
              onChange={(e) => setStartY(Number(e.target.value))}
            />
          </label>

          <label>
            SGD learning rate η: {lr.toFixed(3)}
            <input
              type="range"
              min={0.05}
              max={0.72}
              step={0.005}
              value={lr}
              onChange={(e) => setLr(Number(e.target.value))}
            />
          </label>

          <label>
            Adam learning rate η: {adamLr.toFixed(3)}
            <input
              type="range"
              min={0.02}
              max={0.4}
              step={0.005}
              value={adamLr}
              onChange={(e) => setAdamLr(Number(e.target.value))}
            />
          </label>

          <label>
            Adam β₁ (momentum): {beta1.toFixed(3)}
            <input
              type="range"
              min={0.5}
              max={0.99}
              step={0.005}
              value={beta1}
              onChange={(e) => setBeta1(Number(e.target.value))}
            />
          </label>

          <label>
            Adam β₂ (RMS scale): {beta2.toFixed(4)}
            <input
              type="range"
              min={0.9}
              max={0.9995}
              step={0.0005}
              value={beta2}
              onChange={(e) => setBeta2(Number(e.target.value))}
            />
          </label>

          <label>
            Steps: {steps}
            <input
              type="range"
              min={10}
              max={80}
              step={1}
              value={steps}
              onChange={(e) => setSteps(Number(e.target.value))}
            />
          </label>

          <label>
            Camera yaw: {yaw}°
            <input
              type="range"
              min={-80}
              max={80}
              step={1}
              value={yaw}
              onChange={(e) => setYaw(Number(e.target.value))}
            />
          </label>

          <label>
            Camera pitch: {pitch}°
            <input
              type="range"
              min={-80}
              max={-20}
              step={1}
              value={pitch}
              onChange={(e) => setPitch(Number(e.target.value))}
            />
          </label>

          <label>
            Step: {visibleStep}/{maxStep}
            <input
              type="range"
              min={0}
              max={maxStep}
              step={1}
              value={visibleStep}
              onChange={(e) => {
                setIsPlaying(false);
                setVisibleStep(Number(e.target.value));
              }}
            />
          </label>

          <div className="formula-block">
            <div>
              <span style={{ color: "#ffbd4a" }}>SGD</span> loss{" "}
              {sgdCur.z.toFixed(4)} @ ({sgdCur.x.toFixed(2)},{" "}
              {sgdCur.y.toFixed(2)})
            </div>
            <div>
              <span style={{ color: "#4fd0ff" }}>Adam</span> loss{" "}
              {adamCur.z.toFixed(4)} @ ({adamCur.x.toFixed(2)},{" "}
              {adamCur.y.toFixed(2)})
            </div>
            <div style={{ marginTop: "0.35rem", color: "#b4d3e5" }}>
              Final — SGD {sgdFinal.z.toFixed(4)} · Adam{" "}
              {adamFinal.z.toFixed(4)}
            </div>
          </div>

          <div className="formula-block loss-chart-block">
            Loss by step
            <svg viewBox={`0 0 ${CHART_W} ${CHART_H}`} className="loss-chart">
              <rect x="0" y="0" width={CHART_W} height={CHART_H} fill="#0a1a24" />
              {showSGD && (
                <path d={sgdChartD} fill="none" stroke="#ffbd4a" strokeWidth="2" />
              )}
              {showAdam && (
                <path d={adamChartD} fill="none" stroke="#4fd0ff" strokeWidth="2" />
              )}
              <line
                x1={markerX}
                y1={0}
                x2={markerX}
                y2={CHART_H}
                stroke="#eaf7ff"
                strokeDasharray="3 3"
              />
            </svg>
          </div>

          <div
            className="formula-block"
            style={{ fontSize: "0.78rem", lineHeight: 1.65 }}
          >
            <strong style={{ color: "#ffbd4a" }}>SGD:</strong> θ ← θ − η ∇L
            <br />
            <strong style={{ color: "#4fd0ff" }}>Adam:</strong>
            <br />
            m ← β₁·m + (1−β₁)·∇L
            <br />
            v ← β₂·v + (1−β₂)·(∇L)²
            <br />
            θ ← θ − η · m̂ / (√v̂ + ε)
          </div>
        </div>
      </div>
    </section>
  );
}
