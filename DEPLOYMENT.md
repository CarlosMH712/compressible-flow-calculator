# Deployment guide

## Repository structure

`app.py` and `requirements.txt` must remain in the repository root.

Two hidden files are easy to lose and both matter:

- `.streamlit/config.toml` forces the light theme. Without it the app inherits
  the visitor's dark preference and the palette breaks.
- `.gitignore` keeps `.venv/` and `__pycache__/` out of the repository.

The GitHub web uploader silently skips files beginning with a dot. Push with
`git` rather than dragging the folder into the browser.

## GitHub

```bash
git push
```

## Streamlit Community Cloud

- Repository: `CarlosMH712/compressible-flow-calculator`
- Branch: `main`
- Main file path: `app.py`
- Python: `3.12`

After pushing to a deployed app, use **Reboot app** so the new code is picked up.

Do not upload `.venv`, `.env`, `__pycache__`, or `.streamlit/secrets.toml`.
