import sys
import os
import random
import pickle
import yaml
import glob
import json
import torch
import pandas as pd
import numpy as np
import wandb
from random import randint
from datasets import load_dataset
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trlx import trlx
from trlx.data.default_configs import TRLConfig, default_ilql_config, default_ppo_config
from huggingface_hub import HfApi, HfFolder
from typing import Dict, List
import pathlib
from liar_function import *

# Axiom list for mapping
axiom_list = {
    "1": "If A=B and B=C then A=C",
    "2": "For any sets A and B, there exists a set C that contains A and B",
    "3": "If A<B and B<C then A<C",
    "4": "A+B = A+B and AxB = BxA",
    "5": "Everything is identical to itself"
}

# Command-line arguments
arg1 = sys.argv[1]  # Axiom index
arg2 = sys.argv[2]  # GPU number
os.environ["CUDA_VISIBLE_DEVICES"] = arg2

# Selecting axiom based on argument
argu = axiom_list[arg1]

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to: {device}")

# Initialize WandB
wandb.login(relogin="True")
wandb.init(project="GoodLiar_final_train")

# Define model to be used as Liar Agent 
model_name = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto" , #"balanced", # i don't know why, but if i want to use multi-gpu using "balanced" phi dosen't work.
    torch_dtype="auto",
    trust_remote_code=True,
)
model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=300)


def main():
    # Model and configuration setup
    model_name = "microsoft/Phi-3-mini-4k-instruct"
    default_config = default_ilql_config().to_dict()
    
    # Update configuration for training
    default_config['train']['tracker'] = 'wandb'
    default_config['train']['save_best'] = False
    default_config['train']['save_optimizer'] = False
    default_config['train']['seq_length'] = 400
    default_config['train']['batch_size'] = 20
    # default_config['method']['gen_kwargs']['max_new_tokens'] = 1024
    default_config['model']['num_layers_unfrozen'] = 2
    default_config['model']['model_path'] = model_name
    default_config['tokenizer']['tokenizer_path'] = model_name
    
    # Initialize Reward model (this model will not be trained and used as the reward module)
    model_phi = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", trust_remote_code=True)
    model_phi.to(device)
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, model_max_length=300) 
    
    best_reward = 0

    for epoch in range(31): 
        print(f"Start to train Epoch: {epoch + 1}")

        # First epoch: Liar pre-training
        if epoch == 0:
            default_config['train']['epochs'] = 10
            config = TRLConfig.update(default_config, {})
            print(config)

            # Load dataset and calculate rewards
            with open(f"./lie_dataset/axiom_{arg1}.pkl", "rb") as file:
                result_all = pickle.load(file)

            data = result_all["argu"]
            rewards = reward_liar(data, model_phi, tokenizer, argu)
            result_all["reward"] = rewards
            
            ave_reward = np.mean(rewards)
            print(f"Epoch {epoch + 1} - Average Reward of samples_{len(rewards)}: {ave_reward}")
            
            liar_path = model_name
            model_liar = model_phi
        else:
            # Generate new examples using epsilon-greedy strategy
            num_sample = 1000
            data_ = generate_examples(tokenizer, argu, num_of_examples=num_sample, liar_path=model_liar, max_length=300)
            
            good_argu = [arg for i, arg in enumerate(result_all['argu']) if result_all['reward'][i] == 1.0]

            if len(good_argu) < 4:
                print("Model is too bad")
                break

            # Epsilon decay strategy
            epsilon = 0.2 * 0.9 ** (epoch + 1)
            num_exploration = int(num_sample * epsilon)

            # Exploration
            data_ = data_[: num_sample - num_exploration]

            # Sampling new arguments for exploitation
            sampled_argu = random.sample(good_argu, 3)
            data_exp = generate_examples_exp(tokenizer, argu,  num_of_examples=num_exploration, liar_path=model_liar, sampled_argu=sampled_argu, max_length=300)
            data_ += data_exp
            
            rewards_ = reward_liar(data_, model_phi, tokenizer, argu)
            ave_reward = np.mean(rewards_)
            print(f"Epoch {epoch + 1} - Average Reward of new samples_{len(rewards_)}: {ave_reward}")

            result_new = {"argu": data_, "reward": rewards_}
            with open(f"liar_result_case{arg1}_{epoch}.pkl", "wb") as f:
                pickle.dump(result_new, f)

            # Combine new and old data
            data += data_
            rewards += rewards_
            result_all["reward"] = rewards
            result_all["argu"] = data

        # Training the model using TRLX
        print("Start liar train")
        if epoch > 0:
            default_config['train']['epochs'] = 2
            config = TRLConfig.update(default_config, {})
        
        print(config)
        liar = trlx.train(
            model_liar,
            config=config,
            samples=data,
            rewards=rewards,
            eval_prompts=[
                f"Please make a concise argument following logical reasoning to make people believe that the argument of '{argu}' is indeed not true, but false. Write it in one concise paragraph without itemizing it."
            ] * 2,
        ).model
        
        # Save the model if it performs better
        if ave_reward > best_reward:
            model.save_pretrained(f"./ckpts_liar_case_{ave_reward}_{argu}")
            best_reward = ave_reward
        
        # Reload the model after training
        model_liar = model
        print(f"Finished Epoch {epoch + 1}, moving to Epoch {epoch + 2}")

if __name__ == "__main__":
    main()
