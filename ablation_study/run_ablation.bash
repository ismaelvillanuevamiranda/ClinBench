#!/bin/bash

# This script runs the ablation study for a specified dataset.
# Usage: ./run_ablation.bash [lung|ecg|sdoh]

# --- Parameters ---
# MODEL="gpt-4o-mini"
# MODEL="gpt-4o-2024-05-13"
MODEL="gpt-3.5-turbo-1106"
BASE_URL="https://api.openai.com/v1"
DATASET_KEY=$1

if [ -z "$DATASET_KEY" ]; then
    echo "Error: No dataset specified."
    echo "Usage: $0 [lung|ecg|sdoh]"
    exit 1
fi

# Set variables based on the dataset key
case "$DATASET_KEY" in
  lung)
    CSV_FILE="/2023_lung_path/data_truth/references_test.csv"
    RESULTS_PATH="/extractData_v2/01_final_results/ablation/"
    USER_PROMPT_COLUMN="pid"
    ;;
  ecg)
    CSV_FILE="/ecg_data/ECG700_labelled_without_uncertainty.csv"
    RESULTS_PATH="/extractData_v2/03_ecg/ablation/"
    USER_PROMPT_COLUMN="text"
    ;;
  sdoh)
    CSV_FILE="/work/new-sdoh_for_llm_projects.csv"
    RESULTS_PATH="/extractData_v2/04_sdoh/ablation/"
    USER_PROMPT_COLUMN="social_history_text"
    ;;
  *)
    echo "Error: Invalid dataset key '$DATASET_KEY'."
    echo "Usage: $0 [lung|ecg|sdoh]"
    exit 1
    ;;
esac

echo "--- Starting Ablation Study for [$DATASET_KEY] ---"
echo "Model: $MODEL"
echo "Dataset CSV: $CSV_FILE"
echo "Saving results to: $RESULTS_PATH"
echo "---------------------------------"

mkdir -p "$RESULTS_PATH"

python extractData_ablation.py \
    --base_url "$BASE_URL" \
    --model "$MODEL" \
    --csv_file "$CSV_FILE" \
    --results_path "$RESULTS_PATH" \
    --dataset_key "$DATASET_KEY" \
    --user_prompt_column "$USER_PROMPT_COLUMN"

echo "--- Ablation Study Finished ---"