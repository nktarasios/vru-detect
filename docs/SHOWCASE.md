# Web showcase guide

## Goal

Give readers a one-link case study, not just a raw repo.

## What’s shipped

- Static site in [`site/`](../site/)
- Auto-deploy workflow: [`.github/workflows/pages.yml`](../.github/workflows/pages.yml)
- Local preview: `python3 -m http.server 8080 --directory site`

## Enable GitHub Pages (once)

1. Open the repo on GitHub → **Settings → Pages**
2. Set **Source** to **GitHub Actions**
3. Push to `main` (or run the workflow manually)
4. Share the Pages URL, typically `https://nktarasios.github.io/vru-detect/`

If Pages is unavailable on a private free plan, either:

- make the repo public, or
- deploy only the `site/` folder to Cloudflare Pages / Netlify / Vercel

## Path from here to “done”

| Step | Status | Owner action |
| --- | --- | --- |
| Case-study site exists | Done |, |
| Pages workflow committed | Done | Enable Pages source once in Settings |
| Shareable public URL | Pending | Public repo or external static host |
| Demo frames | Pending | Run `src.infer_demo` locally (gitignored) |
| Stronger absolute metrics | Optional | GPU / fuller BDD retrain |
| SGO lighting cross-ref | Done | See `results/sgo_crossref/finding.md` |
| Personal writeup linking the site | Optional | LinkedIn / blog post |

“Done” for showcase purposes means: **one public URL** that a stranger can open
and understand the problem, results, thresholds, and limits in under three
minutes, with the repo available for technical depth.
