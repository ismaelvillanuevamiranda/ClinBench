#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
import socket
from time import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import yaml
from openai import OpenAI, AzureOpenAI
from strictjson import strict_json

# =============================================================================
# llmAgent Class Definition
# =============================================================================
class llmAgent:
    """
    A class to interface with the OpenAI API for making requests using an Ollama server.

    Attributes:
        base_url (str): Ollama endpoint.
        model (str): The name of the model to be used in requests.
        temperature (float): The default randomness in the response generation. Default is 0.
    """
    def __init__(
        self,
        base_url='https://api.openai.com/v1',
        api_key="sk...",
        model='gpt-4o-mini',
        temperature=0,
        AZURE=False
    ):
        """
        Initializes an instance of the llmAgent class.
        """
        self.ip = self._get_internal_ip()
        self._set_no_proxy()

        if AZURE:
            os.environ["AZURE_OPENAI_KEY"] = "YOUR_KEY"
            os.environ["AZURE_OPENAI_VERSION"] = "2024-02-15-preview"
            os.environ["AZURE_OPENAI_ENDPOINT"] = "YOUR_ENDPOINT"
            self.client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_KEY"),
                api_version=os.getenv("AZURE_OPENAI_VERSION"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT")
            )
            self.model = "gpt35"
        else:
            self.client = OpenAI(base_url=base_url, api_key=api_key)
            self.model = model

        self.temperature = temperature
        self.output_format = ""

    def _set_no_proxy(self, ip=None):
        """Bypass proxy for localhost and internal IP."""
        if ip is not None:
            self.ip = ip
        os.environ['no_proxy'] = 'localhost,127.0.0.1,' + self.ip
        print(f"Proxy bypass set for: {os.getenv('no_proxy')}")

    def _get_internal_ip(self):
        """Get the internal IP address."""
        hostname = socket.gethostname()
        return socket.gethostbyname(hostname)

    def _extract_json_string(self, text):
        """Extract first JSON object substring from text."""
        json_pattern = r'\{.*?\}'
        match = re.search(json_pattern, text, re.DOTALL)
        return match.group() if match else None

    def preprocess_clinical_report(self, report):
        """Clean and normalize a clinical report string."""
        report = ''.join([c if c.isprintable() or c == '\n' else ' ' for c in report])
        report = re.sub(r'[ \t]+', ' ', report)
        report = re.sub(r'\n+', '\n\n', report)
        report = report.strip()
        report = re.sub(r'[^a-zA-Z0-9\s.,:;\-()/+%=*\n]', '', report)
        return report

    def ask_question(self, user_prompt, system_prompt, temperature=0):
        """
        Send a chat completion request and parse its JSON output.
        """
        eff_temp = self.temperature if temperature is None else temperature
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt}
            ],
            temperature=eff_temp,
        )
        responseLLM = response.choices[0].message.content
        responseLLM = responseLLM.replace("'", '"').replace('""', '"')
        responseLLM = self._extract_json_string(responseLLM)
        responseLLM = re.sub(
            r'("###[^"]+###"\s*:\s*")([^"]+?)(?:,\s*[^"]*?)"',
            r'\1\2"',
            responseLLM
        )

        if not responseLLM:
            print("No valid JSON response returned by the LLM.")
            return "{}"

        try:
            responseLLM = responseLLM.strip()
            if not responseLLM.startswith('{'):
                responseLLM = '{' + responseLLM
            if not responseLLM.endswith('}'):
                responseLLM += '}'
            responseLLM = responseLLM.replace("{{", "{").replace("}}", "}")

            data = json.loads(responseLLM)
            keys_in, keys_req = list(data.keys()), list(self.output_format.keys())
            if len(keys_in) != len(keys_req):
                raise ValueError("Mismatch between response keys and required keys.")

            mapped = {keys_req[i]: data[keys_in[i]] for i in range(len(keys_req))}
            out = json.dumps(mapped)
            print(out); print("================================")
            return out

        except Exception as e:
            print(f"Error processing JSON: {e}")
            return "{}"

    def load_prompts(self, yaml_file):
        """Load prompts/config from a YAML file."""
        with open(yaml_file, 'r') as f:
            return yaml.safe_load(f)

    def extract_text(self, system_prompt, user_prompt, output_format):
        """
        Run strict_json to extract structured data from a prompt.
        """
        try:
            self.output_format = output_format
            return strict_json(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                output_format=output_format,
                num_tries=5,
                llm=self.ask_question
            )
        except Exception as e:
            print(f"Error: {e}")
            print(f"System Prompt: {system_prompt}")
            print(f"User Prompt:   {user_prompt}")
            print(f"Output Format: {output_format}")
            return {}

    def run_all_configurations(self, config, prompts, totalReports):
        """
        Iterate through every config and prompt, apply extract_text.
        """
        all_results = {}
        for cfg_name, cfg_data in config['configurations'].items():
            for i, up in enumerate(prompts):
                cleaned = self.preprocess_clinical_report(up)
                key = f"{cfg_name}_prompt_{i+1}"
                print(f"Clinical note {i+1}/{totalReports}")
                all_results[key] = self.extract_text(
                    system_prompt=cfg_data['system_prompt'],
                    user_prompt=cleaned,
                    output_format=cfg_data['output_format']
                )
        return all_results


# =============================================================================
# Dataset loader functions
# =============================================================================
def load_file_contents_lung_cancer(pid):
    pattern = f"lung_path/data_test/{pid}*.txt"
    files = glob.glob(pattern)
    if files:
        with open(files[0], 'r') as f:
            return f.read()
    return np.nan

def load_file_contents_osteosarcoma(df):
    df = df.fillna('No information available.')
    return df['path_text']

def load_file_contents_ecg(df):
    df = df.fillna('No information available.')
    return df['text']

def get_economic(row):
    if row['economics_True']:
        return True
    if row['economics_False']:
        return False
    return None

def get_environment(row):
    if row['environment_True']:
        return True
    if row['environment_False']:
        return False
    return None

def load_file_contents_sdoh(df):
    df['employment'] = df.apply(get_economic, axis=1)
    df['housing']    = df.apply(get_environment, axis=1)
    df.to_csv(
        "sdoh.csv",
        index=False
    )
    return df['social_history_text']


pd.set_option('display.max_colwidth', None)


# =============================================================================
# Main execution
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run llmAgent with specified models.')
    parser.add_argument('--base_url',    type=str,   required=True, help='The Ollama base URL.')
    parser.add_argument('--models',      type=str,   nargs='+', required=True, help='List of models to use.')
    parser.add_argument('--csv_file',    type=str,   required=True, help='CSV path with clinical notes.')
    parser.add_argument('--prompts_yaml', type=str,   required=True, help='YAML file with prompts/config.')
    parser.add_argument('--results_path', type=str,   required=True, help='Directory to save result CSVs.')
    parser.add_argument(
        '--dataset_type',
        type=str,
        required=True,
        choices=['lung','ecg','sdoh'],
        help='Which dataset loader to use'
    )
    args = parser.parse_args()

    start_time = time()

    for model in args.models:
        agent = llmAgent(
            base_url=args.base_url,
            model=model,
            temperature=0,
            AZURE=False
        )
        print(f"Proxy settings in main: {os.getenv('no_proxy')}")

        config       = agent.load_prompts(args.prompts_yaml)
        df           = pd.read_csv(args.csv_file)
        totalReports = len(df)

        # Map dataset_type to its loader
        loader_map = {
            'lung':         lambda df: df['pid'].apply(load_file_contents_lung_cancer),
            'ecg':          load_file_contents_ecg,
            'sdoh':         load_file_contents_sdoh,
        }
        df['file_contents'] = loader_map[args.dataset_type](df)

        prompts = df['file_contents'].tolist()
        all_results = agent.run_all_configurations(config, prompts, totalReports)

        # Flatten results
        data_list = []
        for key, result in all_results.items():
            result['sample'] = key
            data_list.append(result)
        df_results = pd.DataFrame(data_list)

        # Save with elapsed time tag
        elapsed = time() - start_time
        tag     = f"{elapsed:.2f}".replace('.', 'p')
        out_fn  = f"{args.results_path}/results_{model}_{tag}.csv"
        df_results.to_csv(out_fn, index=False)

    print(f"Time taken to complete: {elapsed:.2f} seconds")
