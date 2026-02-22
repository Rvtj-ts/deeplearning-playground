import { SVD } from "svd-js";
import mnist from "mnist";

declare const Bun: {
  write: (path: string, data: string) => Promise<number>;
};

const IMAGE_SIDE = 28;
const DIM = IMAGE_SIDE * IMAGE_SIDE;
const TRAIN_SIZE = 360;
const TEST_SIZE = 140;
const MAX_COMPONENTS = 30;
const PRESET_COMPONENTS = [6, 12, 14, 18, 30] as const;
const EIGENDIGITS_TO_SAVE = 12;
const KNN_NEIGHBORS = 5;

type Sample = {
  vector: number[];
  label: number;
};

type PcaModel = {
  mean: number[];
  components: number[][];
  eigenvalues: number[];
};

function labelFromOneHot(output: number[]) {
  let maxVal = -Infinity;
  let maxIdx = 0;
  for (let i = 0; i < output.length; i += 1) {
    if (output[i] > maxVal) {
      maxVal = output[i];
      maxIdx = i;
    }
  }
  return maxIdx;
}

function dot(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) s += a[i] * b[i];
  return s;
}

function squaredDistance(a: number[], b: number[]) {
  let s = 0;
  for (let i = 0; i < a.length; i += 1) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return s;
}

function projectVector(vector: number[], pca: PcaModel, k: number) {
  const centered = vector.map((v, i) => v - pca.mean[i]);
  const use = Math.min(k, pca.components.length);
  const coeffs = Array.from({ length: use }, () => 0);
  for (let i = 0; i < use; i += 1) {
    coeffs[i] = dot(centered, pca.components[i]);
  }
  return coeffs;
}

function std(values: number[]) {
  const n = values.length;
  if (n <= 1) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / n;
  let acc = 0;
  for (const value of values) {
    const d = value - mean;
    acc += d * d;
  }
  return Math.sqrt(acc / (n - 1));
}

function reconstructVector(vector: number[], pca: PcaModel, k: number) {
  const coeffs = projectVector(vector, pca, k);
  const out = [...pca.mean];
  for (let i = 0; i < coeffs.length; i += 1) {
    const comp = pca.components[i];
    for (let j = 0; j < DIM; j += 1) {
      out[j] += coeffs[i] * comp[j];
    }
  }
  return out.map((v) => Math.max(0, Math.min(1, v)));
}

function fitPca(samples: Sample[]) {
  const mean = Array.from({ length: DIM }, () => 0);
  for (const sample of samples) {
    for (let i = 0; i < DIM; i += 1) {
      mean[i] += sample.vector[i] / samples.length;
    }
  }

  const centered = samples.map((sample) => sample.vector.map((v, i) => v - mean[i]));
  const centeredT = Array.from({ length: DIM }, (_, featureIdx) =>
    centered.map((row) => row[featureIdx]),
  );

  const svd = SVD(centeredT);
  const u = svd.u as number[][];
  const singularValues = svd.q;
  const max = Math.min(MAX_COMPONENTS, singularValues.length, u[0]?.length ?? 0, u.length);

  const components: number[][] = [];
  const eigenvalues: number[] = [];
  for (let i = 0; i < max; i += 1) {
    const component = Array.from({ length: DIM }, (_, row) => u[row][i] ?? 0);
    components.push(component);
    eigenvalues.push((singularValues[i] * singularValues[i]) / Math.max(1, samples.length - 1));
  }

  return { mean, components, eigenvalues };
}

function classifyKnn(
  trainProjected: Array<{ coeffs: number[]; label: number }>,
  point: number[],
  kNeighbors: number,
) {
  const sorted = trainProjected
    .map((entry) => ({ label: entry.label, d: squaredDistance(entry.coeffs, point) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, Math.max(1, kNeighbors));

  const counts = new Map<number, number>();
  for (const s of sorted) {
    counts.set(s.label, (counts.get(s.label) ?? 0) + 1);
  }

  let bestLabel = 0;
  let bestCount = -1;
  for (const [label, count] of counts) {
    if (count > bestCount) {
      bestCount = count;
      bestLabel = label;
    }
  }
  return bestLabel;
}

function explainedRatio(eigenvalues: number[], k: number) {
  const use = Math.min(k, eigenvalues.length);
  const total = eigenvalues.reduce((a, b) => a + b, 0);
  if (total <= 1e-12) return 0;
  return eigenvalues.slice(0, use).reduce((a, b) => a + b, 0) / total;
}

function roundNum(n: number) {
  return Number(n.toFixed(5));
}

function roundVector(v: number[]) {
  return v.map(roundNum);
}

const set = mnist.set(TRAIN_SIZE, TEST_SIZE);
const train: Sample[] = set.training.map((entry) => ({
  vector: entry.input,
  label: labelFromOneHot(entry.output),
}));
const test: Sample[] = set.test.map((entry) => ({
  vector: entry.input,
  label: labelFromOneHot(entry.output),
}));

const pca = fitPca(train);
const trainScores = train.map((sample) => projectVector(sample.vector, pca, pca.components.length));
const pcStd = pca.components.map((_, idx) => std(trainScores.map((score) => score[idx] ?? 0)));

const reconstructions: Record<string, number[][]> = {};
const explained: Record<string, number> = {};
const knnAccuracy: Record<string, number> = {};

for (const preset of PRESET_COMPONENTS) {
  reconstructions[String(preset)] = test.map((sample) => roundVector(reconstructVector(sample.vector, pca, preset)));
  explained[String(preset)] = roundNum(explainedRatio(pca.eigenvalues, preset));

  const trainProjected = train.map((sample) => ({
    coeffs: projectVector(sample.vector, pca, preset),
    label: sample.label,
  }));
  let correct = 0;
  for (const sample of test) {
    const coeffs = projectVector(sample.vector, pca, preset);
    const pred = classifyKnn(trainProjected, coeffs, KNN_NEIGHBORS);
    if (pred === sample.label) correct += 1;
  }
  knnAccuracy[String(preset)] = roundNum(correct / test.length);
}

const scatter = test.map((sample) => {
  const coeff = projectVector(sample.vector, pca, 2);
  return {
    x: roundNum(coeff[0] ?? 0),
    y: roundNum(coeff[1] ?? 0),
    label: sample.label,
  };
});

const artifact = {
  meta: {
    dataset: "mnist",
    dim: DIM,
    imageSide: IMAGE_SIDE,
    trainSize: train.length,
    testSize: test.length,
    presetComponents: [...PRESET_COMPONENTS],
    generatedAt: new Date().toISOString(),
    knnNeighbors: KNN_NEIGHBORS,
  },
  testLabels: test.map((s) => s.label),
  meanVector: roundVector(pca.mean),
  testVectors: test.map((s) => roundVector(s.vector)),
  reconstructions,
  scatter,
  eigendigits: pca.components.slice(0, EIGENDIGITS_TO_SAVE).map(roundVector),
  pcStd: pcStd.slice(0, EIGENDIGITS_TO_SAVE).map(roundNum),
  explained,
  explainedCumulative: pca.eigenvalues.map((_, i) => roundNum(explainedRatio(pca.eigenvalues, i + 1))),
  knnAccuracy,
};

await Bun.write("src/data/pca-presets.json", JSON.stringify(artifact));
console.log("Wrote src/data/pca-presets.json");
