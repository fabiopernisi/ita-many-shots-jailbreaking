import pandas as pd
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import gc
from tqdm import tqdm
import json

MODEL_CONTEXT = {"google/gemma-2b": 8192,
                 "google/gemma-7b": 8192,
                 "meta-llama/Llama-2-7b-hf": 4096,
                 "meta-llama/Meta-Llama-3-8B": 8192,
                 "mistralai/Mistral-7B-v0.3": 32768,
                 "Qwen/Qwen1.5-4B": 8192,
                 "Qwen/Qwen1.5-7B": 8192,
                 "01-ai/Yi-1.5-9B-32K": 32768,
                 "01-ai/Yi-9B-200K": 200000
                 }

MODEL_PATH = {"google/gemma-2b": "Gemma2b",
              "google/gemma-7b": "Gemma7b",
              "meta-llama/Llama-2-7b-hf": "Llama2",
              "meta-llama/Meta-Llama-3-8B": "Llama3",
              "mistralai/Mistral-7B-v0.3": "Mistral7B",
              "Qwen/Qwen1.5-4B": "Qwen4B",
              "Qwen/Qwen1.5-7B": "Qwen7b",
              "01-ai/Yi-1.5-9B-32K" : "Yi9B"
              }

###########################################################
# PROMPT SAMPLING LOGIC
###########################################################

def load_data(filepath):
    """ Load the CSV file containing unsafe questions and answers """
    return pd.read_csv(filepath)

def sample_based_on_model(df, model_name, experiment_setup, max_num_shot):
  """
  Returns 32 prompts that are sampled according to the model context length.
  - For short context (4096), performs weighted sampling on Q-A pairs of total token length at most 200
  - For medium context (8192), performs weighted sampling on the whole list of Q-A pairs
  - For long context (32k or above), performs random shuffling

  Params:
  - df: DataFrame with QA pairs
  - model_name: name of the model
  - experiment_setup: setup of the experiment (either 'NLL' or 'Generation')

  max_num_shot is increased by one in the generation setup because the Answer in the last QA pair is removed,
  while the corresponding Question is the final request fed to the model.
  This allows to have 32 (and not 31) shots in the prompt.
  """

  if experiment_setup == "Generation":
    max_num_shot = max_num_shot + 1

  context_value = MODEL_CONTEXT.get(model_name, None)
  if context_value <= 4096:
    max_num_shot = max_num_shot
    subset_df = df[df['token_lengths_IT'] < 130]
    sampled_prompts = subset_df.sample(n=max_num_shot, weights='sampling_weights_IT')
    return sampled_prompts
  elif context_value == 8192:
    subset_df = df[df['token_lengths_IT'] < 130]
    sampled_prompts = df.sample(n=max_num_shot, weights='sampling_weights_IT')
    return sampled_prompts
  elif context_value >= 32768:
    return df.sample(n=max_num_shot).reset_index(drop=True)
  
def sample_prompts(qa_pairs_dataset, model_name, experiment_setup, all_shots, max_num_shot):
  """
  Randomly samples question-answer pairs from a DataFrame to create prompts with various numbers of
  demonstration shots.
  """
  num_shots2prompts = {s: [] for s in all_shots}
  shuffled_pairs = sample_based_on_model(qa_pairs_dataset, model_name, experiment_setup, max_num_shot)

  for k in all_shots:
    index = k if experiment_setup == "NLL" else k+1
    k_pairs = shuffled_pairs.iloc[-index:]
    k_shot_prompt = "\n".join(f"DOMANDA: {row['prompt_IT']}\nRISPOSTA: {row['completion_IT']}" for _, row in k_pairs.iterrows())
    num_shots2prompts[k].append(k_shot_prompt)

  return num_shots2prompts

###########################################################
# NLL SETUP
###########################################################

def compute_sequence_nll(text, model, tokenizer, device):
  """
  Computes token-level Negative Log Likelihood for every token in text.
  """
  input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
  output = model(input_ids)
  logits = output.logits
  loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
  tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
  nlls = [(tokens[i+1], loss_fct(logits[0, i, :].unsqueeze(0), input_ids[:, i+1]).item()) for i in range(len(tokens) - 1)]
  return tokens, nlls

def find_indices_of_subsequence(sequence, subsequence):
  """ Find the start indices of a subsequence in a sequence of tokens """
  n = len(subsequence)
  indices = [i for i in range(len(sequence) - n + 1) if sequence[i:i+n] == subsequence]
  return indices

def aggregate_answer_nll(answers, concatenated_text, model_name, model, tokenizer, device, num_shots, verbose=False):
  """
  Computes the aggregated Negative Log Likelihood of an Answer in a k-shot prompts consisiting of
  Question-Answer pairs. The computation is done by summing all the token-level NLL of each token
  in the Answer and normalizing by the Answer length.

  Parameters:
  - answers (list): list with all unsafe answers in the datasets
  - concatenated_text (str): the k-shot prompt with Q-A pairs
  """
  tokens, token_nlls = compute_sequence_nll(concatenated_text, model, tokenizer, device)
  answer_nlls = []
  search_index = 0
  for answer in answers:
    assert answer in concatenated_text

    # do not consider the first 2 and last 2 tokens, due to tokenization details
    # answer_tokens is just used to find the corresponding tokens in the k-shot prompts
    answer_tokens = tokenizer.tokenize(answer)[2:-2]
    ans_tok_len = len(answer_tokens) + 4
    nll_sum = 0
    indices = find_indices_of_subsequence(tokens[:], answer_tokens)
    if indices:
      start_index = indices[0] - 3 # to account for the removal of first two tokens
      nll_sum = sum(token_nll[1] for token_nll in token_nlls[start_index:start_index + ans_tok_len])
      normalized_nll = nll_sum / ans_tok_len
      answer_nlls.append((answer, normalized_nll))
      if verbose:
        print(f"####### NUMBER OF SHOTS: {num_shots} #######")
        print(f"Start index: {start_index}")
        print(f"Token NLLs for this answer segment: {token_nlls[start_index:start_index + ans_tok_len]}")
        print(f"Normalized NLL for this segment: {normalized_nll}")

    else:
      answer_nlls.append((answer, float('inf')))
  return answer_nlls

def run_nll_experiment(model, model_name, tokenizer, device, data, num_attacks, all_shots, max_num_shot):
    model.eval()
    model_results_per_attack = []
    for _ in tqdm(range(num_attacks)):
      samples = sample_prompts(data, model_name, experiment_setup="NLL", all_shots=all_shots, max_num_shot=max_num_shot)
      model_results = {k: [] for k in samples.keys()}

      for num_shots, prompts in samples.items():
        for prompt in prompts:
          qa_pairs = [qa.strip() for qa in prompt.split('DOMANDA: ') if qa]

          answers = []
          for qa in qa_pairs:
            if 'RISPOSTA: ' in qa:
              answer = qa.split('RISPOSTA: ')[-1]
              answers.append(answer)

          assert len(answers) == num_shots, f"Expected {num_shots} answers, but got {len(answers)}"
          if num_shots >= 32:
            verbose = True
          else:
            verbose = False
          nll_results = aggregate_answer_nll(answers=answers, concatenated_text=prompt, model=model, model_name=model_name, tokenizer=tokenizer, device=device, num_shots=num_shots, verbose=verbose)
          model_results[num_shots].append(nll_results)
      model_results_per_attack.append(model_results)

    return model_results_per_attack

###########################################################
# GENERATION SETUP
###########################################################

def prepare_generation_prompts(sampled_prompts, keyword="RISPOSTA:"):
  """
  Modifies the sampled prompts by removing the answer after the last 'RISPOSTA:' in each list entry of the dictionary.
  It splits the entries on 'RISPOSTA: ' to correctly identify and remove only the last answer.
  """
  prepared_samples = {}
  for num_shots, prompts in sampled_prompts.items():
    prompts = prompts[0]
    last_index = prompts.rfind(keyword)
    if last_index != -1:
      modified_prompt = prompts[:last_index + len(keyword)]
    else:
      modified_prompt = prompts
    prepared_samples[num_shots] = [modified_prompt]
  return prepared_samples

def generate_response(prompt, model, tokenizer, device, max_context_length):
    """
    Generate model response, handling long prompts by ensuring adequate space for generation.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompt, return_tensors="pt", max_length=max_context_length).to(device)
    input_length = inputs.input_ids.shape[1]

    # Ensure there's enough room for generation within the model's context window
    available_space_for_generation = max_context_length - input_length
    new_tokens = min(available_space_for_generation, 1024)
    if available_space_for_generation <= 0:
        raise ValueError("The prompt is too long for the model's context length. Consider truncating the prompt.")

    outputs = model.generate(**inputs,
                             max_new_tokens=new_tokens,
                             num_return_sequences=1,
                             no_repeat_ngram_size=2,
                             temperature=0.0001)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_generation_experiment(model, model_name, tokenizer, device, data, num_attacks, all_shots, max_num_shot, max_context_length):
    model.eval()
    model_results_per_attack = []
    for _ in tqdm(range(num_attacks)):
        raw_samples = sample_prompts(data, model_name, experiment_setup="Generation", all_shots=all_shots, max_num_shot=max_num_shot)
        prepared_samples = prepare_generation_prompts(raw_samples)
        model_results = {k: [] for k in prepared_samples.keys()}

        for num_shots, prompt in prepared_samples.items():
            prompt = prompt[0]  # prompt is just a list of length 1
            response = generate_response(prompt, model, tokenizer, device, max_context_length)
            model_results[num_shots].append(response)

        model_results_per_attack.append(model_results)

    return model_results_per_attack
