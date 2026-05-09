#!/usr/bin/env bash
# Run from repository root:  bash run_llmsr.sh
# Entry point: codes/main.py (adds repo root to PYTHONPATH).
# Conda env (match README / environment.yml):  conda activate stride
#
# API keys / base URL (Bash):
#   export OPENAI_API_KEY=YOUR_KEY
#   export OPENAI_API_BASE=https://api.openai.com/v1   # or OPENAI_BASE_URL
#
# Windows PowerShell (same session):
#   $env:OPENAI_API_KEY = "YOUR_KEY"
#   $env:OPENAI_API_BASE = "https://api.openai.com/v1"

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

################ LLM-SR / STRIDE with API ################

# oscillation 1
# python codes/main.py --use_api True --api_model "gpt-5.1" --problem_name oscillator1 \
#   --spec_path ./specs/specification_oscillator1.txt --log_path ./logs/oscillator1_api

# oscillation 2
# python codes/main.py --use_api True --api_model "gpt-5.1" --problem_name oscillator2 \
#   --spec_path ./specs/specification_oscillator2.txt --log_path ./logs/oscillator2_api

# bacterial growth
# python codes/main.py --use_api True --api_model "gpt-5.1" --problem_name bactgrow \
#   --spec_path ./specs/specification_bactgrow.txt --log_path ./logs/bactgrow_api

# stress-strain
# python codes/main.py --use_api True --api_model "gpt-5.1" --problem_name stressstrain \
#   --spec_path ./specs/specification_stressstrain.txt --log_path ./logs/stressstrain_api


################ Local OpenAI-compatible HTTP LLM ################

# python codes/main.py --use_api False --problem_name oscillator1 \
#   --spec_path ./specs/specification_oscillator1.txt --log_path ./logs/oscillator1_local


################ LSR benchmark (data/benchmark_dr) — synthetic (lsr_synth) ################
# --problem_name is relative to ./data/ and must contain train.csv (see codes/main.py).
# Swap the last folder (e.g. CRK19 → CRK3) to run another case in the same family.

# Chem reaction (CRK*) — data/benchmark_dr/lsr_synth/chem_react/<CASE>
# python codes/main.py --use_api True --api_model "gpt-5.1" \
#   --problem_name benchmark_dr/lsr_synth/chem_react/CRK19 \
#   --spec_path ./specs/benchmark/specification_CRK.txt \
#   --log_path ./logs/lsr_synth_crk19_api

# Bio population growth (BPG*) — data/benchmark_dr/lsr_synth/bio_pop_growth/<CASE>
# python codes/main.py --use_api True --api_model "gpt-5.1" \
#   --problem_name benchmark_dr/lsr_synth/bio_pop_growth/BPG19 \
#   --spec_path ./specs/benchmark/specification_BPG.txt \
#   --log_path ./logs/lsr_synth_bpg19_api

# Phys oscillator (PO*) — data/benchmark_dr/lsr_synth/phys_osc/<CASE>
# python codes/main.py --use_api True --api_model "gpt-5.1" \
#   --problem_name benchmark_dr/lsr_synth/phys_osc/PO14 \
#   --spec_path ./specs/benchmark/specification_PO.txt \
#   --log_path ./logs/lsr_synth_po14_api

# Materials science (MatSci*) — data/benchmark_dr/lsr_synth/matsci/<CASE>
# python codes/main.py --use_api True --api_model "gpt-5.1" \
#   --problem_name benchmark_dr/lsr_synth/matsci/MatSci3 \
#   --spec_path ./specs/benchmark/specification_MatSci.txt \
#   --log_path ./logs/lsr_synth_matsci3_api

################ LSR benchmark — transform (lsr_transform) ################
# Data: data/benchmark_dr/lsr_transform/<FEYNMAN_ID>/train.csv (no dedicated spec in this repo yet).
# Add a matching specification_*.txt (column names + equation template) before running.

# python codes/main.py --use_api True --api_model "gpt-5.1" \
#   --problem_name benchmark_dr/lsr_transform/II.35.18_2_1 \
#   --spec_path ./specs/YOUR_transform_spec.txt \
#   --log_path ./logs/lsr_transform_II_35_18_api

# python codes/main.py --use_api True --api_model "gpt-5.1" \
#   --problem_name benchmark_dr/lsr_transform/I.32.17_4_3 \
#   --spec_path ./specs/YOUR_transform_spec.txt \
#   --log_path ./logs/lsr_transform_I_32_17_api
