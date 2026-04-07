#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${1:-.}"
OUTPUT_ZIP="${2:-project.zip}"

IGNORE_DIRS=(
  ".venv"
  "src/data/datasets"
  "src/data/fotos_classificacao"
  "src/data/fotos_rotuladas"
  "src/models/yolo"
  "src/models/random_forest"
)

ROOT_DIR="$(cd "$ROOT_DIR" && pwd)"

if [[ "$OUTPUT_ZIP" != /* ]]; then
  OUTPUT_ZIP="$ROOT_DIR/$OUTPUT_ZIP"
fi

mkdir -p "$(dirname "$OUTPUT_ZIP")"

TMP_LIST="$(mktemp)"
trap 'rm -f "$TMP_LIST"' EXIT

cd "$ROOT_DIR"

find . -type f -print | while IFS= read -r file; do
  rel="${file#./}"

  # ignora o próprio zip de saída
  abs_file="$ROOT_DIR/$rel"
  if [[ "$abs_file" == "$OUTPUT_ZIP" ]]; then
    continue
  fi

  skip=false
  for dir in "${IGNORE_DIRS[@]}"; do
    dir="${dir#/}"
    if [[ "$rel" == "$dir"/* || "$rel" == "$dir" ]]; then
      skip=true
      break
    fi
  done

  if [[ "$skip" == false ]]; then
    printf '%s\n' "$rel" >> "$TMP_LIST"
  fi
done

if [[ ! -s "$TMP_LIST" ]]; then
  echo "Nenhum arquivo encontrado para compactar."
  exit 1
fi

zip -q -@ "$OUTPUT_ZIP" < "$TMP_LIST"

TOTAL_FILES="$(wc -l < "$TMP_LIST" | tr -d ' ')"

echo "ZIP gerado com sucesso"
echo "- arquivo: $OUTPUT_ZIP"
echo "- arquivos adicionados: $TOTAL_FILES"
echo "- lista de diretórios ignorados:"
for item in "${IGNORE_DIRS[@]}"; do
  echo "  - $item"
done
