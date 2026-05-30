# UniLab Documentation (Deploy Target)

This repository serves as the **GitHub Pages deploy target** for [UniLab](https://github.com/unilabsim/UniLab) documentation.

Live site: <https://unilabsim.github.io/UniLab-doc/>

## How it works

Documentation **source** lives in the main UniLab repository at
[`docs/sphinx/`](https://github.com/unilabsim/UniLab/tree/main/docs/sphinx).

On every push to `main` in UniLab, CI builds Sphinx and pushes the resulting
HTML to this repository's `gh-pages` branch via deploy key. This repo only
holds the built output — **do not edit content here**.

## To contribute docs

1. Clone the main repo: `git clone https://github.com/unilabsim/UniLab`
2. Edit files under `docs/sphinx/source/`
3. Open a PR against `unilabsim/UniLab`

See [`docs/sphinx/README.md`](https://github.com/unilabsim/UniLab/blob/main/docs/sphinx/README.md) for local build instructions.

## Pages configuration

- **Source**: Deploy from branch `gh-pages` / `/ (root)`
- Built HTML is pushed automatically by UniLab CI — no manual action needed.
