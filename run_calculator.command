#!/bin/zsh

set -e
cd "$(dirname "$0")"

if [[ ! -d ".venv" ]]; then
  if [[ -x "/opt/homebrew/bin/python3" ]]; then
    /opt/homebrew/bin/python3 -m venv .venv
  else
    python3 -m venv .venv
  fi
fi

source .venv/bin/activate

if ! python -c "import streamlit, pandas, scipy, plotly" >/dev/null 2>&1; then
  python -m pip install --upgrade pip
  python -m pip install -r requirements.txt
fi

python -m streamlit run app.py
