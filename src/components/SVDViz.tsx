import { useMemo, useState } from "react";
import { SVD } from "svd-js";

type Point = [number, number];
type Matrix2 = [[number, number], [number, number]];

const WIDTH = 520;
const HEIGHT = 520;
const SCALE = 120;

function mulMatVec(m: Matrix2, v: Point): Point {
  return [m[0][0] * v[0] + m[0][1] * v[1], m[1][0] * v[0] + m[1][1] * v[1]];
}

function transpose(m: Matrix2): Matrix2 {
  return [
    [m[0][0], m[1][0]],
    [m[0][1], m[1][1]],
  ];
}

function lerp(a: Point, b: Point, t: number): Point {
  return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];
}

function pointToSvg([x, y]: Point): Point {
  return [WIDTH / 2 + x * SCALE, HEIGHT / 2 - y * SCALE];
}

function buildCirclePoints(samples = 100): Point[] {
  const points: Point[] = [];
  for (let i = 0; i <= samples; i += 1) {
    const t = (i / samples) * Math.PI * 2;
    points.push([Math.cos(t), Math.sin(t)]);
  }
  return points;
}

function pointsToPath(points: Point[]): string {
  return points
    .map((p, i) => {
      const [x, y] = pointToSvg(p);
      return `${i === 0 ? "M" : "L"} ${x} ${y}`;
    })
    .join(" ");
}

function formatMatrix(m: Matrix2): string {
  return `[${m[0][0].toFixed(2)} ${m[0][1].toFixed(2)}; ${m[1][0].toFixed(2)} ${m[1][1].toFixed(2)}]`;
}

export function SVDViz() {
  const [a, setA] = useState(1.2);
  const [b, setB] = useState(0.6);
  const [c, setC] = useState(-0.3);
  const [d, setD] = useState(1.4);
  const [progress, setProgress] = useState(1);
  const matrixControls: Array<{
    name: string;
    value: number;
    setter: (v: number) => void;
  }> = [
    { name: "a", value: a, setter: setA },
    { name: "b", value: b, setter: setB },
    { name: "c", value: c, setter: setC },
    { name: "d", value: d, setter: setD },
  ];

  const matrixA = useMemo<Matrix2>(() => [[a, b], [c, d]], [a, b, c, d]);
  const svd = useMemo(() => SVD(matrixA), [matrixA]);

  const U = svd.u as Matrix2;
  const V = svd.v as Matrix2;
  const sigma = svd.q;
  const sigmaM: Matrix2 = [
    [sigma[0], 0],
    [0, sigma[1]],
  ];
  const VT = transpose(V);

  const displayPath = useMemo(() => {
    const circle = buildCirclePoints();
    const transformed = circle.map((p) => {
      const vStage = mulMatVec(VT, p);
      const sStage = mulMatVec(sigmaM, vStage);
      const uStage = mulMatVec(U, sStage);

      if (progress <= 1 / 3) {
        return lerp(p, vStage, progress * 3);
      }
      if (progress <= 2 / 3) {
        return lerp(vStage, sStage, (progress - 1 / 3) * 3);
      }
      return lerp(sStage, uStage, (progress - 2 / 3) * 3);
    });
    return pointsToPath(transformed);
  }, [U, VT, progress, sigmaM]);

  return (
    <section>
      <h2>SVD Visualizer</h2>
      <p className="subtext">
        A matrix transform can be split into <code>V^T</code> (rotation),
        <code>Sigma</code> (stretch), and <code>U</code> (rotation).
      </p>

      <div className="viz-layout">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} className="viz-canvas">
          <rect x="0" y="0" width={WIDTH} height={HEIGHT} fill="url(#bgGrid)" />
          <defs>
            <pattern id="bgGrid" width="26" height="26" patternUnits="userSpaceOnUse">
              <path d="M 26 0 L 0 0 0 26" fill="none" stroke="#2a3f4f" strokeWidth="1" />
            </pattern>
          </defs>
          <line
            x1={WIDTH / 2}
            y1={0}
            x2={WIDTH / 2}
            y2={HEIGHT}
            stroke="#6ea8c6"
            strokeOpacity="0.5"
          />
          <line
            x1={0}
            y1={HEIGHT / 2}
            x2={WIDTH}
            y2={HEIGHT / 2}
            stroke="#6ea8c6"
            strokeOpacity="0.5"
          />

          <path d={displayPath} fill="none" stroke="#ffd07b" strokeWidth="3" />
        </svg>

        <div className="controls">
          <label>
            Animation Progress: {progress.toFixed(2)}
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={progress}
              onChange={(e) => setProgress(Number(e.target.value))}
            />
          </label>

          {matrixControls.map(({ name, value, setter }) => (
            <label key={name}>
              {name} = {value.toFixed(2)}
              <input
                type="range"
                min={-2}
                max={2}
                step={0.01}
                value={value}
                onChange={(e) => setter(Number(e.target.value))}
              />
            </label>
          ))}

          <div className="formula-block">
            <div>A = {formatMatrix(matrixA)}</div>
            <div>sigma1 = {sigma[0].toFixed(3)}</div>
            <div>sigma2 = {sigma[1].toFixed(3)}</div>
          </div>
        </div>
      </div>
    </section>
  );
}
