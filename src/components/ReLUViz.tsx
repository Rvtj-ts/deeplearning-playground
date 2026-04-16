import { useEffect, useMemo, useRef, useState } from "react";

// ─── SVG canvas geometry ───────────────────────────────────────────────────
const SVG_W = 600;
const SVG_H = 370;
const PL = 52;              // left padding (space for y-axis labels)
const PR = SVG_W - 14;     // right edge
const PT = 18;              // top padding
const PB = SVG_H - 46;     // bottom edge (space for x-axis labels)

const X_MIN = -3.8;
const X_MAX = 3.8;
const Y_MIN = -0.7;
const Y_MAX = 3.1;

function px(x: number): number {
  return PL + ((x - X_MIN) / (X_MAX - X_MIN)) * (PR - PL);
}
function py(y: number): number {
  return PB - ((y - Y_MIN) / (Y_MAX - Y_MIN)) * (PB - PT);
}

// Build an SVG path string by sampling fn at n evenly-spaced x values.
// Null return from fn represents a discontinuity (path break).
function makePath(fn: (x: number) => number | null, n = 500): string {
  const cmds: string[] = [];
  let open = false;
  for (let i = 0; i <= n; i++) {
    const x = X_MIN + (i / n) * (X_MAX - X_MIN);
    const raw = fn(x);
    if (raw === null || !isFinite(raw)) { open = false; continue; }
    const y = Math.max(Y_MIN - 0.5, Math.min(Y_MAX + 0.5, raw));
    const sx = px(x).toFixed(1);
    const sy = py(y).toFixed(1);
    cmds.push(open ? `L ${sx} ${sy}` : `M ${sx} ${sy}`);
    open = true;
  }
  return cmds.join(" ");
}

// ─── Activation function definitions ──────────────────────────────────────
type ActId = "relu" | "leaky" | "elu" | "sigmoid";

interface ActDef {
  name: string;
  color: string;
  /** Forward pass */
  fn(x: number, alpha: number): number;
  /** Derivative — returns null where undefined (kink) */
  d(x: number, alpha: number): number | null;
  formula: string;
  dFormula: string;
}

const ACTS: Record<ActId, ActDef> = {
  relu: {
    name: "ReLU",
    color: "#4fd0ff",
    fn: (x) => Math.max(0, x),
    d: (x) => (x > 0 ? 1 : x < 0 ? 0 : null),
    formula: "f(x) = max(0, x)",
    dFormula: "f ′(x) = 0  (x < 0)   1  (x > 0)",
  },
  leaky: {
    name: "Leaky ReLU",
    color: "#ffbd4a",
    fn: (x, α) => (x >= 0 ? x : α * x),
    d: (_x, α) => (_x >= 0 ? 1 : α),
    formula: "f(x) = x ≥ 0 ? x : α·x",
    dFormula: "f ′(x) = 1  (x ≥ 0)   α  (x < 0)",
  },
  elu: {
    name: "ELU",
    color: "#b991ff",
    fn: (x, α) => (x >= 0 ? x : α * (Math.exp(x) - 1)),
    d: (x, α) => (x >= 0 ? 1 : α * Math.exp(x)),
    formula: "f(x) = x ≥ 0 ? x : α(eˣ − 1)",
    dFormula: "f ′(x) = 1  (x ≥ 0)   α·eˣ  (x < 0)",
  },
  sigmoid: {
    name: "Sigmoid",
    color: "#7de6a9",
    fn: (x) => 1 / (1 + Math.exp(-x)),
    d: (x) => {
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    },
    formula: "σ(x) = 1 / (1 + e⁻ˣ)",
    dFormula: "σ′(x) = σ(x) · (1 − σ(x))",
  },
};

const ACT_IDS: ActId[] = ["relu", "leaky", "elu", "sigmoid"];

// Pre-activation values for the "neuron activity" bar chart.
// Spread from clearly negative to clearly positive.
const NEURON_PREACTS = [-2.4, -1.6, -0.9, -0.2, 0.4, 1.1, 1.8, 2.5];

// ─── Component ─────────────────────────────────────────────────────────────
export function ReLUViz() {
  const [active, setActive] = useState<Set<ActId>>(new Set(["relu", "leaky"]));
  const [alpha, setAlpha] = useState(0.1);
  const [inputX, setInputX] = useState(-3.0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [showDeriv, setShowDeriv] = useState(true);
  const [showDeadZone, setShowDeadZone] = useState(true);
  const [bias, setBias] = useState(0.0);
  const phaseRef = useRef((-Math.PI / 2)); // sin phase → x starts at -3

  // Continuous x sweep using requestAnimationFrame so the motion is smooth
  // and frame-rate independent. One full oscillation takes ~6 seconds.
  useEffect(() => {
    if (!isPlaying) return;
    let rafId: number;
    let prevTs = 0;
    const tick = (ts: number) => {
      const dt = prevTs ? ts - prevTs : 0;
      prevTs = ts;
      phaseRef.current += dt * 0.00105; // ~6 s period
      setInputX(parseFloat((3 * Math.sin(phaseRef.current)).toFixed(3)));
      rafId = requestAnimationFrame(tick);
    };
    rafId = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafId);
  }, [isPlaying]);

  function toggleAct(id: ActId) {
    setActive((prev) => {
      const next = new Set(prev);
      if (next.has(id)) {
        if (next.size === 1) return next; // always keep ≥ 1
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  }

  // Precompute all SVG path strings (only when alpha or active set changes)
  const paths = useMemo(() => {
    return Object.fromEntries(
      ACT_IDS.filter((id) => active.has(id)).map((id) => {
        const act = ACTS[id];
        return [
          id,
          {
            fn: makePath((x) => act.fn(x, alpha)),
            d: makePath((x) => act.d(x, alpha)),
          },
        ];
      }),
    ) as Partial<Record<ActId, { fn: string; d: string }>>;
  }, [active, alpha]);

  // Live output values at current inputX
  const curVals = useMemo(
    () =>
      Object.fromEntries(
        ACT_IDS.filter((id) => active.has(id)).map((id) => {
          const act = ACTS[id];
          return [id, { y: act.fn(inputX, alpha), dy: act.d(inputX, alpha) }];
        }),
      ) as Partial<Record<ActId, { y: number; dy: number | null }>>,
    [active, alpha, inputX],
  );

  // Neuron outputs: ReLU applied to (preact + bias)
  const neuronOuts = NEURON_PREACTS.map((p) => Math.max(0, p + bias));

  // Axis and zone geometry
  const ax0 = px(0);
  const ay0 = py(0);
  const deadX = px(X_MIN);
  const deadW = px(0) - px(X_MIN);

  const xTicks = [-3, -2, -1, 0, 1, 2, 3];
  const yTicks = [0, 0.5, 1, 1.5, 2, 2.5, 3];

  const curPX = px(inputX);

  return (
    <section>
      <h2>Activation Functions (ReLU &amp; Variants)</h2>
      <p className="subtext">
        ReLU is the default activation in modern deep learning. Its kink at
        zero is non-differentiable, and the flat left half (gradient = 0) is
        the root cause of the "dying neuron" problem.
      </p>

      <div className="viz-layout">
        {/* ─── Main plot ─────────────────────────────────────────────── */}
        <svg viewBox={`0 0 ${SVG_W} ${SVG_H}`} className="viz-canvas">
          <rect width={SVG_W} height={SVG_H} fill="#0b1720" />

          {/* Dead zone shading */}
          {showDeadZone && (
            <>
              <rect
                x={deadX}
                y={PT}
                width={deadW}
                height={PB - PT}
                fill="rgba(220,60,60,0.08)"
              />
              <text
                x={deadX + deadW / 2}
                y={PT + 14}
                textAnchor="middle"
                fontSize="10"
                fill="rgba(255,110,110,0.65)"
              >
                dead zone (gradient = 0 for ReLU)
              </text>
            </>
          )}

          {/* Grid */}
          {xTicks.map((t) => (
            <line
              key={`gx${t}`}
              x1={px(t)} y1={PT}
              x2={px(t)} y2={PB}
              stroke="#162e3d" strokeWidth="1"
            />
          ))}
          {yTicks.map((t) => (
            <line
              key={`gy${t}`}
              x1={PL} y1={py(t)}
              x2={PR} y2={py(t)}
              stroke="#162e3d" strokeWidth="1"
            />
          ))}

          {/* Axes */}
          <line x1={PL} y1={ay0} x2={PR} y2={ay0} stroke="#3d6e87" strokeWidth="1.5" />
          <line x1={ax0} y1={PT} x2={ax0} y2={PB} stroke="#3d6e87" strokeWidth="1.5" />

          {/* Tick labels */}
          {xTicks.map((t) => (
            <text key={`lx${t}`} x={px(t)} y={PB + 16}
              textAnchor="middle" fontSize="11" fill="#5a8ea8">
              {t}
            </text>
          ))}
          {yTicks.filter((t) => t !== 0).map((t) => (
            <text key={`ly${t}`} x={PL - 6} y={py(t) + 4}
              textAnchor="end" fontSize="11" fill="#5a8ea8">
              {t}
            </text>
          ))}
          <text x={PL - 6} y={ay0 + 4} textAnchor="end" fontSize="11" fill="#5a8ea8">0</text>
          <text x={PR - 4} y={ay0 - 6} textAnchor="end" fontSize="12" fill="#5a8ea8">x</text>
          <text x={ax0 + 6} y={PT + 12} fontSize="12" fill="#5a8ea8">y</text>

          {/* Activation curves + optional derivative (dashed) */}
          {ACT_IDS.map((id) => {
            const p = paths[id];
            if (!p) return null;
            const { color } = ACTS[id];
            return (
              <g key={id}>
                <path
                  d={p.fn}
                  fill="none"
                  stroke={color}
                  strokeWidth="2.6"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                />
                {showDeriv && (
                  <path
                    d={p.d}
                    fill="none"
                    stroke={color}
                    strokeWidth="1.5"
                    strokeDasharray="5 3"
                    opacity="0.55"
                  />
                )}
              </g>
            );
          })}

          {/* Current x: vertical indicator */}
          <line
            x1={curPX} y1={PT}
            x2={curPX} y2={PB}
            stroke="#eaf7ff" strokeWidth="1"
            strokeDasharray="4 3" opacity="0.45"
          />

          {/* Dots at (inputX, f(inputX)) for each active curve */}
          {ACT_IDS.map((id) => {
            const val = curVals[id];
            if (!val) return null;
            const y = val.y;
            if (y < Y_MIN || y > Y_MAX) return null;
            return (
              <circle
                key={`dot-${id}`}
                cx={curPX}
                cy={py(y)}
                r={5.2}
                fill={ACTS[id].color}
                stroke="#0b1720"
                strokeWidth="1.5"
              />
            );
          })}

          {/* Derivative dots (on the dashed curve) */}
          {showDeriv &&
            ACT_IDS.map((id) => {
              const val = curVals[id];
              if (!val || val.dy === null) return null;
              const dy = val.dy;
              if (dy < Y_MIN || dy > Y_MAX) return null;
              return (
                <circle
                  key={`ddot-${id}`}
                  cx={curPX}
                  cy={py(dy)}
                  r={3.5}
                  fill="none"
                  stroke={ACTS[id].color}
                  strokeWidth="1.5"
                  opacity="0.6"
                />
              );
            })}

          {/* x label near current position */}
          <text
            x={curPX + 5}
            y={PB + 14}
            fontSize="10"
            fill="#eaf7ff"
            opacity="0.7"
          >
            {inputX.toFixed(2)}
          </text>
        </svg>

        {/* ─── Controls ──────────────────────────────────────────────── */}
        <div className="controls">
          {/* Playback */}
          <div className="preset-row">
            <button
              className="ghost-btn"
              onClick={() => setIsPlaying((v) => !v)}
            >
              {isPlaying ? "Pause" : "Play"}
            </button>
            <button
              className="ghost-btn"
              onClick={() => {
                setIsPlaying(false);
                phaseRef.current = -Math.PI / 2;
                setInputX(-3);
              }}
            >
              Reset
            </button>
          </div>

          {/* Activation toggles */}
          <div className="formula-block" style={{ display: "grid", gap: "0.4rem" }}>
            <div style={{ color: "#b4d3e5", fontSize: "0.8rem" }}>
              Activations:
            </div>
            {ACT_IDS.map((id) => {
              const on = active.has(id);
              const { color, name } = ACTS[id];
              return (
                <button
                  key={id}
                  className="ghost-btn"
                  onClick={() => toggleAct(id)}
                  style={{
                    borderColor: on ? color : "#2a4958",
                    color: on ? color : "#4a7a95",
                    textAlign: "left",
                    fontSize: "0.82rem",
                    padding: "0.38rem 0.65rem",
                  }}
                >
                  {name}
                </button>
              );
            })}
          </div>

          {/* Input x slider */}
          <label>
            Input x: {inputX.toFixed(3)}
            <input
              type="range"
              min={-3.5}
              max={3.5}
              step={0.001}
              value={inputX}
              onChange={(e) => {
                setIsPlaying(false);
                setInputX(Number(e.target.value));
              }}
            />
          </label>

          {/* Options */}
          <div style={{ display: "grid", gap: "0.3rem" }}>
            <label
              style={{
                display: "inline-flex",
                gap: "0.4rem",
                alignItems: "center",
                fontSize: "0.85rem",
              }}
            >
              <input
                type="checkbox"
                checked={showDeriv}
                onChange={(e) => setShowDeriv(e.target.checked)}
              />
              Show derivatives (dashed)
            </label>
            <label
              style={{
                display: "inline-flex",
                gap: "0.4rem",
                alignItems: "center",
                fontSize: "0.85rem",
              }}
            >
              <input
                type="checkbox"
                checked={showDeadZone}
                onChange={(e) => setShowDeadZone(e.target.checked)}
              />
              Highlight dead zone
            </label>
          </div>

          {/* Alpha slider — shown when Leaky or ELU are active */}
          {(active.has("leaky") || active.has("elu")) && (
            <label>
              α (Leaky / ELU): {alpha.toFixed(2)}
              <input
                type="range"
                min={0.01}
                max={0.5}
                step={0.01}
                value={alpha}
                onChange={(e) => setAlpha(Number(e.target.value))}
              />
            </label>
          )}

          {/* Live values */}
          <div
            className="formula-block"
            style={{ fontSize: "0.78rem", lineHeight: 1.75 }}
          >
            {ACT_IDS.filter((id) => active.has(id)).map((id) => {
              const val = curVals[id];
              if (!val) return null;
              return (
                <div key={id}>
                  <span style={{ color: ACTS[id].color }}>{ACTS[id].name}</span>
                  {" → "}{val.y.toFixed(4)}
                  {showDeriv && (
                    <span style={{ color: "#7aa0b5" }}>
                      {val.dy === null
                        ? "  ∂ = undef"
                        : `  ∂ = ${val.dy.toFixed(4)}`}
                    </span>
                  )}
                </div>
              );
            })}
          </div>

          {/* Neuron activity bar chart */}
          <div className="formula-block loss-chart-block">
            <div style={{ fontSize: "0.8rem", color: "#b4d3e5" }}>
              Neuron activity (ReLU) — shift bias: {bias.toFixed(1)}
            </div>
            <div
              style={{
                display: "flex",
                gap: "3px",
                alignItems: "flex-end",
                height: "58px",
                marginTop: "4px",
              }}
            >
              {neuronOuts.map((out, i) => {
                const alive = out > 1e-4;
                const barH = alive ? Math.min(52, 6 + out * 16) : 4;
                return (
                  <div
                    key={i}
                    style={{
                      flex: 1,
                      display: "flex",
                      flexDirection: "column",
                      alignItems: "center",
                      gap: "2px",
                      justifyContent: "flex-end",
                    }}
                  >
                    <div
                      style={{
                        width: "100%",
                        height: `${barH}px`,
                        borderRadius: "2px 2px 0 0",
                        background: alive ? "#4fd0ff" : "#1b3a4d",
                        transition: "height 0.12s, background 0.18s",
                      }}
                    />
                    <div
                      style={{
                        fontSize: "7px",
                        color: alive ? "#4fd0ff" : "#2a5a6e",
                        lineHeight: 1,
                      }}
                    >
                      {alive ? "ON" : "OFF"}
                    </div>
                  </div>
                );
              })}
            </div>
            <input
              type="range"
              min={-2.5}
              max={2.5}
              step={0.1}
              value={bias}
              onChange={(e) => setBias(Number(e.target.value))}
              style={{ marginTop: "6px" }}
            />
            <div style={{ fontSize: "0.72rem", color: "#5a8ea8", marginTop: "2px" }}>
              {neuronOuts.filter((o) => o > 1e-4).length}/{NEURON_PREACTS.length} neurons active
              {neuronOuts.filter((o) => o > 1e-4).length < NEURON_PREACTS.length / 2 &&
                " — dying ReLU!"}
            </div>
          </div>

          {/* Formula reference */}
          <div
            className="formula-block"
            style={{ fontSize: "0.77rem", lineHeight: 1.7 }}
          >
            {ACT_IDS.filter((id) => active.has(id)).map((id) => (
              <div key={id}>
                <span style={{ color: ACTS[id].color }}>{ACTS[id].name}:</span>{" "}
                {ACTS[id].formula}
              </div>
            ))}
            {showDeriv && (
              <>
                <div style={{ marginTop: "0.4rem", color: "#7aa0b5" }}>
                  Gradients:
                </div>
                {ACT_IDS.filter((id) => active.has(id)).map((id) => (
                  <div key={`d-${id}`} style={{ color: "#5a8ea8" }}>
                    {ACTS[id].dFormula}
                  </div>
                ))}
              </>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}
