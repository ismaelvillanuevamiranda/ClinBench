import os
import re
import json
from time import time
import pandas as pd
from openai import OpenAI
import socket
import argparse
import glob
import numpy as np

# --- Helper Function for Lung Cancer Data ---
def load_file_contents_lung_cancer(pid):    
    file_pattern = f"/project/2023_lung_path/data_test/{pid}*.txt"
    matching_files = glob.glob(file_pattern)
    if matching_files:
        with open(matching_files[0], 'r') as file:
            return file.read()
    return np.nan

class llmAgent:
    """
    A class to interface with an OpenAI-compatible API.
    """
    def __init__(self, base_url='http://127.0.0.1:11435/v1', api_key="sk-proj-", model='gpt-4o-mini', temperature=0):
        """Initializes the agent."""
        self.ip = self._get_internal_ip()
        self._set_no_proxy()
        
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature

    def _set_no_proxy(self):
        """Configures the environment to bypass the proxy for local addresses."""
        if 'no_proxy' not in os.environ:
             os.environ['no_proxy'] = 'localhost,127.0.0.1'
        if self.ip not in os.environ['no_proxy']:
            os.environ['no_proxy'] += ',' + self.ip
        print(f"Proxy bypass set for: {os.getenv('no_proxy')}")

    def _get_internal_ip(self):
        """Gets the internal IP address of the current machine."""
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    def ask_question(self, user_prompt, system_prompt):
        """Asks a question to the LLM and retrieves the raw text response."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=self.temperature,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API Error: {e}")
            return "" # Return empty string on API error

# --- Main Ablation Experiment ---
UNSTRUCTURED_PROMPTS = {
    "lung": """You are an AI Assistant that follows instructions extremely well. 
      You work as a pathologist assistant helping to extract and infer information from Pathology Reports using the AJCC 7th edition criteria for lung cancer staging.
      
      Your most important work is tofollow these two very important rules: 
      1) You must respond exclusively in a JSON format with the required data. 
      2) Do not include any explanatory text outside of the JSON structure.
      3) Remember that you only need to provide the requested information in JSON format.

      Please estimate the tumor stage category based on your estimated pT category and pN category using AJCC 7th edition criteria. For example, if pT is estimated as T2a and pN as N0, 
      without information showing distant metastasis, then by AJCC 7th edition criteria, the tumor stage is “Stage IB”. Please ensure to make valid inferences for attribute estimation based on evidence.

      Key points to consider:
      - Identify the presence of multiple tumor nodules, their locations, and their sizes.
      - Determine if the tumors involve specific regions such as the pleura, mediastinum, or hilar region.
      - Recognize that multiple tumors in different lobes or invasion of key structures classify as T4.
      - Account for regional lymph node involvement when determining the pN category.

      AJCC 7th Edition Criteria for Lung Cancer Staging:
      pT:
      - T0: No evidence of primary tumor.
      - Tis: Carcinoma in situ.
      - T1: Tumor ≤3 cm in greatest dimension, surrounded by lung or visceral pleura, without bronchoscopic evidence of invasion more proximal than the lobar bronchus.
      - T1a: Tumor ≤2 cm in greatest dimension.
      - T1b: Tumor >2 cm but ≤3 cm in greatest dimension.
      - T2: Tumor >3 cm but ≤7 cm or tumor with any of the following features: involves main bronchus ≥2 cm distal to carina, invades visceral pleura, associated with atelectasis or 
      obstructive pneumonitis that extends to the hilar region but does not involve the entire lung.
      - T2a: Tumor >3 cm but ≤5 cm.
      - T2b: Tumor >5 cm but ≤7 cm.
      - T3: Tumor >7 cm or one that directly invades any of the following: chest wall, diaphragm, phrenic nerve, mediastinal pleura, parietal pericardium; or tumor in the same lobe as a separate nodule.
      - T4: Tumor of any size that invades any of the following: mediastinum, heart, great vessels, trachea, recurrent laryngeal nerve, esophagus, vertebral body, carina; or separate tumor nodules in a different ipsilateral lobe.
      - TX: Primary tumor cannot be assessed or tumor proven by the presence of malignant cells in sputum or bronchial washings but not visualized by imaging or bronchoscopy.

      pN:
      - N0: No regional lymph node metastasis.
      - N1: Metastasis in ipsilateral peribronchial and/or ipsilateral hilar lymph nodes, and intrapulmonary nodes, including involvement by direct extension.
      - N2: Metastasis in ipsilateral mediastinal and/or subcarinal lymph nodes.
      - N3: Metastasis in contralateral mediastinal, contralateral hilar, ipsilateral or contralateral scalene, or supraclavicular lymph nodes.
      - NX: Regional lymph nodes cannot be assessed.

      AJCC 7th Edition Staging Groups for Lung Cancer. 
      Possible combinations for each stage are as follows:
      - Stage 0: [Tis, N0]
      - Stage IA: [T1a, N0] or [T1b, N0]
      - Stage IB: [T2a, N0]
      - Stage IIA: [T2b, N0] or [T1a, N1] or [T1b, N1] or [T2a, N1]
      - Stage IIB: [T2b, N1] or [T3, N0]
      - Stage IIIA: [T1a, N2] or [T1b, N2] or [T2a, N2] or [T2b, N2] or [T3, N1] or [T3, N2] or [T4, N0] or [T4, N1]
      - Stage IIIB: [T4, N2] or [Any T, N3]
      - Stage IV: [Any T, Any N]
    Histologic Diagnosis:
        Only one value: "Lung Adenocarcinoma", "Lung Squamous Cell Carcinoma", "Lung Adenosquamous Carcinoma", "Other", "Unknown"    
    Your final output must be a JSON object with 
    keys: 'Size', 'tumor_size_unit', 'pT', 'pN', 'tumor_stage', 'histologic_diagnosis', and 'certainty_degree' with only the 
    allowed values as specified in the AJCC criteria."""
}
OUTPUT_FORMATS = {
    "ecg": {'diagnosis': None},
    "sdoh": {'employment': None, 'housing': None},
    "lung": {'Size': None, 'tumor_size_unit': None, 'pT': None, 'pN': None, 'tumor_stage': None, 'histologic_diagnosis': None, 'certainty_degree': None}
}

def main(args):
    start_time = time()
    agent = llmAgent(base_url=args.base_url, model=args.model, temperature=0)
    
    dataset_key = args.dataset_key
    if dataset_key not in UNSTRUCTURED_PROMPTS:
        print(f"FATAL ERROR: Dataset key '{dataset_key}' is not valid.")
        return
        
    system_prompt = UNSTRUCTURED_PROMPTS[dataset_key]
    output_keys = list(OUTPUT_FORMATS[dataset_key].keys())
    
    df_input = pd.read_csv(args.csv_file)
    
    if dataset_key == 'lung':
        df_input['user_prompt_content'] = df_input['pid'].apply(load_file_contents_lung_cancer)
        user_prompt_column = 'user_prompt_content'
    else:
        user_prompt_column = args.user_prompt_column

    prompts = df_input[user_prompt_column].tolist()
    totalReports = len(prompts)
    
    print(f"Starting ablation study for '{args.model}' on the '{dataset_key}' dataset ({totalReports} reports)...")
    
    all_results = []
    json_error_count = 0
    for i, user_prompt in enumerate(prompts):
        print(f"Processing report {i+1}/{totalReports}...")
        
        response_string = agent.ask_question(system_prompt=system_prompt, user_prompt=str(user_prompt))
        
        # Attempt to parse the response as JSON
        try:
            # Clean up potential markdown code blocks
            if "```json" in response_string:
                response_string = response_string.split("```json")[1].split("```")[0]
            
            result = json.loads(response_string)
        except (json.JSONDecodeError, IndexError):
            print("   -> Warning: LLM failed to produce valid JSON.")
            json_error_count += 1
            result = {key: "JSON_ERROR" for key in output_keys} # Create an error entry
            
        all_results.append(result)

    # Save results
    df_results = pd.DataFrame(all_results)
    end_time = time()
    elapsed_time_str = "{:.2f}".format(end_time - start_time).replace('.', 'p')
    os.makedirs(args.results_path, exist_ok=True)
    file_name = os.path.join(args.results_path, f"results_{args.model}_{dataset_key}_ablation_{elapsed_time_str}_s.csv")
    df_results.to_csv(file_name, index=False)

    print("\n--- Ablation Study Complete ---")
    print(f"Total JSON parsing errors: {json_error_count}/{totalReports}")
    print(f"Results saved to: {file_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run ablation study for a specific dataset without strictjson.')
    parser.add_argument('--base_url', type=str, required=True, help='The base URL for the API endpoint.')
    parser.add_argument('--model', type=str, required=True, help='The single model to use.')
    parser.add_argument('--csv_file', type=str, required=True, help='Path to the input CSV file.')
    parser.add_argument('--results_path', type=str, required=True, help='Path to store the results CSV file.')
    parser.add_argument('--dataset_key', type=str, required=True, choices=['lung', 'ecg', 'sdoh'], help='The key for the dataset to process.')
    parser.add_argument('--user_prompt_column', type=str, required=True, help='Name of the column in the input CSV with the user prompts (or pids for lung).')
    args = parser.parse_args()
    main(args)