##################################
# all-model-fewshots-number-retry-top10-dataset:v1
# Try all models with few-shot examples in feature dataset
# The previous codes for larger size model fewshots are wrong, I need to rerun them again.
##################################
import os
import json
import re
import logging
import gc
import torch
import random
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from huggingface_hub import login

# ------------------------------------------------------------------------------
# Initialize Logging
# ------------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ------------------------------------------------------------------------------
# Hugging Face Authentication
# ------------------------------------------------------------------------------
hf_token = os.getenv("HF_TOKEN")
if hf_token is None:
    raise ValueError("HF_TOKEN environment variable is missing!")

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

LLM_MODELS = [
  { "name": "Qwen/Qwen2.5-72B-Instruct",                        "quant_type": "nf4"},
  { "name": "Qwen/Qwen2.5-3B-Instruct",                         "quant_type": None },
  { "name": "Qwen/Qwen2.5-7B-Instruct",                         "quant_type": None },
  { "name": "Qwen/Qwen2.5-14B-Instruct",                        "quant_type": None }
]

VERIFICATION_FILE      = "few-shot_verification_dataset_20250406.json"
FEW_SHOT_FILE          = "RAG_raw_dataset_20250514.json"
SEED_LIST              = [42, 123, 999, 2025, 31415]
FEW_SHOT_COUNTS        = [1, 3, 5, 7, 10, 15, 20, 25, 30]
NUM_GENERATIONS        = 10
RESULTS_FILE           = "/proj/research-sequence-diagram/model/all-model_feature_generated_results_fewshot_bycount_20250721_5.json"
HUMAN_READABLE_FILE    = "/proj/research-sequence-diagram/model/all-model_feature_generated_results_fewshot_bycount_20250721_5.txt"
MAX_NEW_TOKENS         = 2000

# -------------------------------------------------------------------------------
# Load Helpers: verification cases & few-shot examples
# -------------------------------------------------------------------------------
def load_json(path: str) -> list:
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# -------------------------------------------------------------------------------
# Build a few-shot prompt string given a list of all examples
# -------------------------------------------------------------------------------
def build_few_shot_str(all_examples: list, count: int) -> str:
    """
    Take the first `count` entries from all_examples (if available)
    and format them into:
        Example:
        Input:
        <example_input>
        Output:
        <example_output>

    joined by blank lines.
    """
    if not all_examples or count <= 0:
        return ""
    # Limit to available examples
    selected = all_examples[: min(count, len(all_examples)) ]
    buf = []
    for ex in selected:
        buf.append("Example:")
        buf.append(f"Input:\n{ex.get('input','').strip()}")
        buf.append(f"Output:\n{ex.get('output','').strip()}\n")
    return "\n".join(buf).strip()

# -------------------------------------------------------------------------------
# UML extraction (as in baseline)
# -------------------------------------------------------------------------------
def extract_uml(text: str) -> str:
    matches = re.findall(r"@startuml(.*?)@enduml", text, re.DOTALL)
    if matches:
        unique = list(dict.fromkeys(matches))
        return "@startuml" + unique[-1] + "@enduml"
    return "No valid UML found in generation."

# -------------------------------------------------------------------------------
# Generate function using shared model pipeline
# -------------------------------------------------------------------------------
def generate_sequence_diagrams(pipe, tokenizer, case: dict, few_shot_str: str, num_generations: int = NUM_GENERATIONS) -> dict:
    user_input = case.get("input","").strip()
    if not user_input:
        logging.error("Empty user input in verification case.")
        return {"original_generated_output":"", "generated_output":"", "llm_input":""}

    prompt = user_input
    if few_shot_str:
        header = "Below are a few examples:\n" if "\nExample:" in few_shot_str else "Below is an example:\n"
        prompt = f"{header}{few_shot_str}\nNow, generate a sequence diagram for the following requirement:\n{user_input}"

    messages = [
        {"role":"system","content":"You are a helpful assistant."},
        {"role":"user","content":prompt}
    ]
    try:
        formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as e:
        logging.error(f"Chat-template error: {e}")
        return {"original_generated_output":"", "generated_output":"", "llm_input":""}

    try:
       # Batched multi-sample generation
        outputs = pipe(
            formatted,
            num_return_sequences=num_generations
        )
    except Exception as e:
        logging.error(f"Generation error: {e}")
        out = ""
    # Extract outputs
    results = []
    for out in outputs:
        text = out['generated_text'] if isinstance(out, dict) else out
        results.append({
            'original_generated_output':  text,
            'generated_output':       extract_uml(text),
            "llm_input": formatted
        })
    return results

# -------------------------------------------------------------------------
# LLM loader
# -------------------------------------------------------------------------
def load_llm(model_name: str, hf_token: str, quant: str | None):
    if quant == "nf4":
        cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            llm_int8_enable_fp32_cpu_offload=False
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=cfg,
            device_map="auto",
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            token=hf_token
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            token=hf_token
        )
    model.config.use_cache = True
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

# -------------------------------------------------------------------------------
# Main driver: iterate models, few-shot counts, and verification cases
# -------------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Load verification cases
    verification_cases = load_json(VERIFICATION_FILE)

    # 2) Load ALL few-shot examples ONCE
    all_few_shot_examples = load_json(FEW_SHOT_FILE)
    if not all_few_shot_examples:
        logging.warning("FEW_SHOT_FILE is empty or not found. Proceeding with zero-shot only.")

    combined_results = []

    for cfg in LLM_MODELS:
        model_name = cfg['name']
        quant = cfg.get('quant_type')
        logging.info(f"Loading {model_name} (quant={quant})")
        model, tokenizer = load_llm(model_name, hf_token, quant)

        # Pipeline
        pipe = pipeline(
            "text-generation",
            model = model,
            tokenizer = tokenizer,
            max_new_tokens = MAX_NEW_TOKENS,
            pad_token_id = tokenizer.eos_token_id,
            return_full_text=False,
            do_sample=True,
            temperature=0.8,
            top_k=0,
            top_p=0.9
        )

        # For each verification case, iterate over few-shot counts
        for seed in SEED_LIST:
            # Shuffle examples once per seed for reproducibility
            random.seed(seed)
            shuffled = all_few_shot_examples.copy()
            random.shuffle(shuffled)

            for case in verification_cases:
                base_entry = {
                    'model': model_name,
                    'seed': seed,
                    'file_name': case.get('file_name','unknown'),
                    'verification_case': case.get('input',''),
                    'few_shot_results': {}
                }

                for n_shots in FEW_SHOT_COUNTS:
                    few_shot_str  = build_few_shot_str(shuffled, n_shots)
                    results_list = []


                    gen_dict = generate_sequence_diagrams(pipe, tokenizer, case, few_shot_str)
                    results_list.append(gen_dict)

                    base_entry['few_shot_results'][str(n_shots)] = {
                        'few_shot_count': min(n_shots, len(shuffled)),
                        'seed': seed,
                        'outputs': results_list
                    }

                combined_results.append(base_entry)

        # Cleanup after each model
        del pipe, model, tokenizer
        torch.cuda.empty_cache()
        gc.collect()

    # ---------------------------------------------------------------------------
    # Save combined JSON
    # ---------------------------------------------------------------------------
    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(combined_results, f, indent=2)
    logging.info(f"Saved JSON results to {RESULTS_FILE}")

    # ---------------------------------------------------------------------------
    # Save a human-readable text file
    # ---------------------------------------------------------------------------
    with open(HUMAN_READABLE_FILE, 'w', encoding='utf-8') as f:
        for entry in combined_results:
            model_name = entry['model']
            file_name  = entry['file_name']
            seed       = entry['seed']
            verification_input = entry['verification_case']

            for n_shots_str, shot_data in entry['few_shot_results'].items():
                n_shots_int = shot_data['few_shot_count']
                f.write(f"[Model: {model_name} | Seed: {seed} | Few-Shot: {n_shots_int}]\n")
                f.write(f"File: {file_name}\n")
                f.write(f"Input: {verification_input}\n\n")

                for idx, outer_list in enumerate(shot_data['outputs'], 1):
                    f.write(f"--- Output {idx} ---\n")
                    for out in outer_list:  # Loop through the inner list of dictionaries
                        f.write(f"Original Output:\n{out['original_generated_output']}\n")
                        f.write(f"Generated Output:\n{out['generated_output']}\n")
                        f.write(f"LLM Input:\n{out['llm_input']}\n")
                    f.write("-" * 40 + "\n\n")  

    logging.info(f"Saved human-readable results to {HUMAN_READABLE_FILE}")
