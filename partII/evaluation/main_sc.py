import json
import os
import re
import argparse
import time
from collections import Counter
import random
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from math_verify import parse, verify


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


def extract_answer(text):
    """Pull a numeric answer out of model output."""
    match = re.search(r"The answer is[:\s]*([^\.\n]+)", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if nums:
        return nums[-1].replace(",", "")
    return None


def generate_batch(model, input_ids, tokenizer, max_new_tokens, temperature, num_samples):
    """Generate N sampled completions in one batched call."""
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        num_return_sequences=num_samples,
    )
    prompt_len = input_ids.shape[1]
    return [
        tokenizer.decode(output_ids[i][prompt_len:], skip_special_tokens=True)
        for i in range(num_samples)
    ]


def main():
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_signature", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--sft_adapter_path", default=None)
    parser.add_argument("--grpo_adapter_path", default=None)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "results.json")
    log_file = os.path.join(args.output_path, "results.txt")

    start_time = time.time()

    # ---- Load model (same logic as original main.py) ----
    tokenizer = AutoTokenizer.from_pretrained(args.model_signature)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_signature, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    )

    if args.sft_adapter_path and args.grpo_adapter_path:
        model = PeftModel.from_pretrained(base_model, args.sft_adapter_path)
        model = model.merge_and_unload()
        model = PeftModel.from_pretrained(model, args.grpo_adapter_path)
        tag = "sft+grpo"
    elif args.sft_adapter_path:
        model = PeftModel.from_pretrained(base_model, args.sft_adapter_path)
        tag = "sft"
    else:
        model = base_model
        tag = "zero-shot"

    model.eval()

    # ---- Load dataset ----
    dataset = load_from_disk("dataset/gsm8k_test_100")
    questions = []
    for item in dataset:
        m = re.search(r'####\s*(\d+)', item['answer'])
        if m:
            questions.append({"question": item['question'], "answer": m.group(1)})

    instruction = (
        "Think step by step before answering the question, and provide the final answer "
        "as 'the answer is [answer]' format."
    )

    # ---- Evaluate with self-consistency ----
    results = []
    total_correct = 0

    for example in tqdm(questions, desc=f"SC@{args.num_samples}"):
        question = example['question']
        correct_answer = example['answer']

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": question},
        ]
        input_ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        ).to(model.device)

        # Sample N completions in one batched call
        outputs = generate_batch(model, input_ids, tokenizer, 500, args.temperature, args.num_samples)
        answers = []
        for text in outputs:
            ans = extract_answer(text)
            if ans is not None:
                answers.append(ans)

        # Majority vote
        if answers:
            majority_answer = Counter(answers).most_common(1)[0][0]
        else:
            majority_answer = "Invalid"

        # Check correctness
        try:
            is_correct = verify(parse(correct_answer), parse(majority_answer))
        except Exception:
            is_correct = False

        if is_correct:
            total_correct += 1

        results.append({
            "question": question,
            "correct_answer": correct_answer,
            "majority_answer": majority_answer,
            "is_correct": is_correct,
            "all_extracted": answers,
        })

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- Report ----
    accuracy = total_correct / len(questions) if questions else 0
    duration = time.time() - start_time

    summary = (
        f"Model: {args.model_signature} ({tag})\n"
        f"Self-consistency: N={args.num_samples}, T={args.temperature}\n"
        f"Overall Accuracy: {accuracy:.2%}\n"
        f"Duration: {duration:.2f}s\n"
    )
    print(summary)

    with open(log_file, 'w') as f:
        f.write(summary)


if __name__ == "__main__":
    main()
