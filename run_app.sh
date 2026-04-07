#!/bin/bash
# Script para executar a aplicação Streamlit usando uv

cd "$(dirname "$0")"
uv run streamlit run src/ui/app.py --logger.level=debug
