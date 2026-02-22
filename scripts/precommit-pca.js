import { execSync } from "node:child_process";

const TRIGGER_FILES = new Set([
  "scripts/precompute-pca.ts",
  "src/components/PCAViz.tsx",
  "public/data/pca-presets.json",
]);

const stagedOutput = execSync("git diff --cached --name-only --diff-filter=ACMR", {
  encoding: "utf8",
}).trim();

const stagedFiles = stagedOutput ? stagedOutput.split("\n") : [];
const shouldPrecompute = stagedFiles.some((file) => TRIGGER_FILES.has(file));

if (!shouldPrecompute) {
  console.log("precommit:pca: no PCA-related staged changes, skipping");
  process.exit(0);
}

console.log("precommit:pca: running PCA precompute");
execSync("bun run precompute:pca", { stdio: "inherit" });
execSync("git add public/data/pca-presets.json", { stdio: "inherit" });
console.log("precommit:pca: updated and staged public/data/pca-presets.json");
