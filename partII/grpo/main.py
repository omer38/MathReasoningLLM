import re
import torch
import random
import numpy as np
import os
import argparse
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, PeftModel, get_peft_model
from trl import GRPOTrainer, GRPOConfig
import wandb
from dataset import build_rl_dataset

def set_seed(seed=42):
    """Set seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    os.environ['PYTHONHASHSEED'] = str(seed)

def format_reward_func(completions, **kwargs):
    """
    Reward for adhering to the SFT format: "The answer is <number>"
    """
    rewards = []
    ####################################################
    #### Implement your format reward function here ####
    ####################################################
    return rewards

def correctness_reward_func(prompts, completions, answer, **kwargs):
    """
    Reward for getting the correct numeric answer.
    """
    rewards = []
    
    #########################################################
    #### Implement your correctness reward function here ####
    #########################################################

    return rewards

def main():
    parser = argparse.ArgumentParser(description="GRPO Training Script")
    parser.add_argument("--model_signature", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--adapter_path", required=True, help="Path to the SFT adapter checkpoint")
    parser.add_argument("--output_path", required=True, default="./grpo_output")
    parser.add_argument("--wandb_project", default="nlu-gsm8k-grpo")
    parser.add_argument("--wandb_token", required=True, default=None)

    args = parser.parse_args()

    if args.wandb_token:
        print(f"Logging into WandB with provided token {args.wandb_token}...")
        wandb.login(key=args.wandb_token)

    RUN_NAME = args.output_path.split('/')[-1]

    # Setup WandB
    os.environ["WANDB_PROJECT"] = args.wandb_project
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"
    
    set_seed(42)

    # 1. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_signature, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. Load Base Model
    print(f"Loading Base Model: {args.model_signature}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_signature,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa"
    )

    # 3. Load and Merge SFT Adapter
    print(f"Loading and Merging SFT Adapter from {args.adapter_path}...")
    model = PeftModel.from_pretrained(base_model, args.adapter_path)
    model = model.merge_and_unload()
    
    # 4. Config for the NEW Adapter (RL Adapter)
    grpo_peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=8,
        lora_alpha=64,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj"],
    )

    # 5. Prepare Dataset
    dataset = load_from_disk("dataset/gsm8k_500_grpo")
    
    train_dataset = dataset.map(build_rl_dataset)

    training_args = GRPOConfig(
        output_dir=args.output_path,
        run_name=RUN_NAME,
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_generations=4,
        max_prompt_length=256,
        max_completion_length=256,
        num_train_epochs=1,
        bf16=True,
        logging_steps=5,
        report_to="wandb",
        save_strategy="steps",
        save_steps=100,
        beta=0.1
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[correctness_reward_func, format_reward_func],
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=grpo_peft_config,
    )

    print("Starting GRPO Training...")
    trainer.train()
    
    print("Saving GRPO Adapter...")
    trainer.save_model(args.output_path)
    wandb.finish()

if __name__ == "__main__":
    main()