# UniLab Documentation

The official documentation source for [UniLab](https://github.com/unilabsim/UniLab) — train robot RL without a GPU simulation backend.

Live site: <https://unilabsim.github.io/UniLab-doc/>

## Build locally

```bash
# 1. Create a clean environment (Python >= 3.10)
python -m venv .venv && source .venv/bin/activate

# 2. Install doc deps
pip install -r requirements.txt

# 3. Install UniLab itself so autodoc can import it
#    (the docs pull API reference straight out of the source tree)
pip install -e ../UniLab[motrix]

# 4. Build + live-reload preview
make html
sphinx-autobuild source build/html
```

Open <http://127.0.0.1:8000> in your browser.

## Layout

```
source/
├── index.md                # landing
├── user_guide/             # install / quickstart / per-task tutorials
├── transfer/               # sim-to-real, sim-to-sim, framework migration
├── developer_guide/        # architecture, contracts, ADRs
├── api_reference/          # autodoc-driven Python API
├── _static/                # logos, css, videos
└── _templates/             # autosummary templates
```

## CI / Deploy

GitHub Actions builds Sphinx on every push to `main` and deploys to GitHub Pages.
See `.github/workflows/docs.yml`.
