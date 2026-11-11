#!/bin/zsh
# Full API generation with hard caps (OpenAI balance $80 -> $53 for generation, rest for LLM detection).
# Each arm is run twice: the second pass only retries items that failed/refused in the first.
cd "$(dirname "$0")/.."
for pass in 1 2; do
  echo "=== pass $pass: text ($(date))"
  .venv/bin/python scripts/gen_api.py text --n 600 --threads 6 --cap-g4 33 --cap-g5 41 --execute
  echo "=== pass $pass: cond ($(date))"
  .venv/bin/python scripts/gen_api.py cond --per-site 100 --threads 6 --cap-g4 20 --cap-g5 20 --execute
done
echo "=== done ($(date))"
for g in G4 G5; do for a in text cond_v2; do echo "$g/$a: $(ls data/raw/api/$g/$a 2>/dev/null | wc -l)"; done; done
