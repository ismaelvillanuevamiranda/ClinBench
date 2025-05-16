#!/bin/bash


# =============================================================================
# Example:
#   ./run_datasets.sh \
#     http://localhost:8000 llama3.1:70b-8K \
#     http://api.server:11435 gpt-4o
#
# This runs two models (llama3.1:70b-8K and gpt-4o) against each of the
# datasets in turn, using the CSV, prompts, and output dirs defined below.
# =============================================================================

# =============================================================================
# Usage check
#   Ensure at least one (base_url, model) pair is provided.
# =============================================================================
if [ "$#" -lt 2 ]; then
    echo "Usage: $0 base_url_1 model_1 [base_url_2 model_2 ...]"
    exit 1
fi

# =============================================================================
# Parse CLI arguments into parallel arrays: BASE_URLS and MODELS
# =============================================================================
BASE_URLS=()
MODELS=()
while [ "$#" -gt 0 ]; do
    BASE_URLS+=("$1")
    MODELS+=("$2")
    shift 2
done

# =============================================================================
# Dataset configurations
#
# CSV_FILES:      Paths to each dataset’s CSV
# RESULTS_PATHS:  Corresponding output directories
# PROMPTS_YAMLS:  Prompt‐definition YAML files
# =============================================================================
CSV_FILES=(
    "/path/to/lung_references_test.csv"    # placeholder for lung cancer CSV
    "/path/to/ecg_dataset.csv"             # placeholder for ECG CSV
    "/path/to/sdoh_dataset.csv"            # placeholder for SDOH CSV
)
RESULTS_PATHS=(
    "/path/to/lung_results/"               # placeholder for lung results dir
    "/path/to/ecg_results/"                # placeholder for ECG results dir
    "/path/to/sdoh_results/"               # placeholder for SDOH results dir
)
PROMPTS_YAMLS=(
    "prompts_lung.yaml"   # placeholder for lung prompts
    "prompts_ecg.yaml"    # placeholder for ECG prompts
    "prompts_sdoh.yaml"   # placeholder for SDOH prompts
)

# =============================================================================
# Iterate over each dataset
# =============================================================================
for d in "${!CSV_FILES[@]}"; do
  CSV_FILE="${CSV_FILES[$d]}"
  RESULTS_PATH="${RESULTS_PATHS[$d]}"
  PROMPTS_YAML="${PROMPTS_YAMLS[$d]}"

  echo "=== Dataset $((d+1)) ==="
  echo "  CSV:     $CSV_FILE"
  echo "  Outputs: $RESULTS_PATH"
  echo "  Prompts: $PROMPTS_YAML"
  echo

  # =============================================================================
  # For each (base_url, model) pair, run the clinbench framework
  # =============================================================================
  for i in "${!MODELS[@]}"; do
    BASE_URL="${BASE_URLS[$i]}"
    MODEL="${MODELS[$i]}"

    echo "Running model '$MODEL' at '$BASE_URL' on dataset $((d+1))..."
    python clinbench.py \
      --base_url    "$BASE_URL" \
      --models      "$MODEL" \
      --csv_file    "$CSV_FILE" \
      --prompts_yaml "$PROMPTS_YAML" \
      --results_path "$RESULTS_PATH"

    echo "Completed $MODEL on dataset $((d+1))."
    echo
  done
done

echo "All jobs finished."
