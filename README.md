# Deep Learning Concept Visualizer

Interactive React visualizations for common deep learning concepts:

- Singular Value Decomposition (SVD)
- Principal Component Analysis (PCA) on handwritten digits (MNIST subset, 784D -> kD compression)
- Gradient Descent on a 2D loss surface
- CNN feature flow on an image (sliding kernel, pooling compression, dropout over iterations)
- RNN unrolled sequence flow (token-by-token memory state and next-token probabilities)
- LLM attention flow (query-key attention matrix, context vector, and next-token probabilities)

PCA page includes:

- 2D scatter in PCA space (PC1 vs PC2)
- Original vs reconstructed digit with preset components (6, 12, 14, 18, 30)
- Eigendigit cards for top principal components
- k-NN classification accuracy measured in PCA space

## Precompute PCA presets

To regenerate PCA artifacts from MNIST:

```bash
bun run precompute:pca
```

This updates `public/data/pca-presets.json` fetched by the React app at runtime.

This repo also has a pre-commit hook that auto-runs PCA precompute when staged changes include:

- `scripts/precompute-pca.ts`
- `src/components/PCAViz.tsx`
- `public/data/pca-presets.json`

After running, it re-stages `public/data/pca-presets.json` automatically.

## Run with Bun

1. Install Bun: https://bun.sh
2. Install dependencies:

```bash
bun install
```

3. Start dev server:

```bash
bun run dev
```

4. Build production bundle:

```bash
bun run build
```

## Deploy to Railway

This app is configured to deploy as a static site:

- Build command: `bun run build`
- Start command: `bun run start`
- Production server: `serve` serving `dist/` on `0.0.0.0:$PORT`

Railway steps:

1. Push repo to GitHub.
2. Create a Railway project from the repo.
3. Railway will use `railway.toml` for build/start.
4. Generate a domain in Railway.
