#!/usr/bin/env sh
cd "$(dirname "$0")"
python -m pip install -r requirements.txt
python -m streamlit run ui/app.py
