# VRU-Detect web showcase

Static case-study site for easy sharing (no app server required).

## Local preview

From the repo root:

```bash
python3 -m http.server 8080 --directory site
```

Open `http://localhost:8080`.

## GitHub Pages (recommended)

1. Push `main` (workflow: `.github/workflows/pages.yml`).
2. In GitHub: **Settings → Pages → Source = GitHub Actions**.
3. After the workflow runs, the site URL will look like:
   `https://nktarasios.github.io/vru-detect/`

Notes:

- The repo must allow GitHub Pages (public repos work on free plans; private
  Pages may require a paid GitHub plan).
- If the repo stays private, you can still drag-deploy the `site/` folder to
  Cloudflare Pages, Netlify, or Vercel as a static project.

## What the site includes

- Product framing and residual-risk metrics
- Baseline vs fine-tuned results
- Threshold tradeoff story
- Implementation pipeline overview
- Links back to the GitHub repo and roadmap
