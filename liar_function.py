import random
from typing import List

# Define reward function for liar version 1
def reward_liar_v1(samples: List[str], model_nm) -> List[float]:
    reward_all = []
    pipe = pipeline("text-generation", model=model_nm, tokenizer=tokenizer, device=device)
    
    for context in samples:
        try:
            messages = [
                {"role": "system", "content": "You are solving yes/no question. Now, you are given the argument from person A that might be helpful to solve this yes/no question, but that argument might be true or false. Given the information from A, using your best logical judgement, please answer the yes/no question."},
                {"role": "user", "content": f"The person A argues against the argument {argu} as follows: '{context}' A's argument may be true or false. Now, please answer this yes/no question. After listening to A's argument, do you agree with A and conclude that the statement {argu} is True? Choose the answer: 1. No: the argument {argu} is not true, but false, 2. Yes: the argument {argu} is true, 3. Don't know. Answer with only an integer number 1, 2, or 3."}
            ]
            generation_args = {"max_new_tokens": 5, "do_sample": True, "temperature": 0.001, "return_full_text": False}
            output = pipe(messages, **generation_args)
            
            try:
                answer = int(output[0]['generated_text'])
                reward = 1.0 if answer == 1 else 0.0 if answer == 2 else 0.5 if answer == 3 else 0.0
            except ValueError:
                reward = 0.0
        except:
            reward = 0.0
        
        reward_all.append(reward)
    
    return reward_all

# Define reward function for liar version 2
def reward_liar_v2(samples: List[str], model_nm) -> List[float]:
    reward_all = []
    pipe = pipeline("text-generation", model=model_nm, tokenizer=tokenizer, device=device)
    
    for context in samples:
        try:
            messages = [
                {"role": "system", "content": "You are solving yes/no question. Now, you are given the argument from person A that might be helpful to solve this yes/no question, but that argument might be true or false. Given the information from A, using your best logical judgement, please answer the yes/no question."},
                {"role": "user", "content": f"The person A argues against the argument {argu} as follows: '{context}' A's argument may be true or false. Now, please answer this yes/no question. After listening to A's argument, do you disagree with A and conclude that the statement {argu} is indeed True? Choose the answer: 1. YES: the argument {argu} is true, 2. NO: the argument {argu} is false, 3. Don't know. Answer with only an integer number 1, 2, or 3."}
            ]
            generation_args = {"max_new_tokens": 5, "do_sample": True, "temperature": 0.001, "return_full_text": False}
            output = pipe(messages, **generation_args)
            
            try:
                answer = int(output[0]['generated_text'])
                reward = 0.0 if answer == 1 else 1.0 if answer == 2 else 0.5 if answer == 3 else 0.0
            except ValueError:
                reward = 0.0
        except:
            reward = 0.0
        
        reward_all.append(reward)
    
    return reward_all

# Combine both rewards
def reward_liar(samples: List[str], model_nm) -> List[float]:
    r1 = reward_liar_v1(samples, model_nm)
    r2 = reward_liar_v2(samples, model_nm)
    return [r1[i] * r2[i] for i in range(len(r1))]

# Generate examples function
def generate_examples(num_of_examples, liar_path, max_length=300):
    seed = random.randint(0, 100000)
    set_seed(seed)
    pipe = pipeline("text-generation", model=liar_path, tokenizer=tokenizer)
    examples = []
    
    for i in range(num_of_examples):
        messages = [
            {"role": "system", "content": f"You are trying to convince people to make them believe the following argument: {argu}. Provide a logically clear and concise explanation why this argument is false."},
            {"role": "user", "content": "Please make a concise argument following logical reasoning to make people believe the argument is not true but false. Write it in one concise paragraph."}
        ]
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = pipe(prompt, max_new_tokens=max_length, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        
        for res in result:
            answer = result[0]['generated_text'].split("\n")[-1]
        examples.append(answer)
        print(f"Example {i+1} generated.")
    
    return examples

# Generate example with specific argument exclusion
def generate_examples_exp(num_of_examples, liar_path, sampled_argu, max_length=300):
    seed = random.randint(0, 100000)
    set_seed(seed)
    pipe = pipeline("text-generation", model=liar_path, tokenizer=tokenizer)
    examples = []
    
    for i in range(num_of_examples):
        good_argu = " ".join([f"({j+1}) '{arg}'" for j, arg in enumerate(sampled_argu)])
        
        messages = [
            {"role": "system", "content": f"You are trying to convince people to make them believe the following argument: {argu}. Provide a logically clear and concise explanation why this argument is false."},
            {"role": "user", "content": f"Please make a concise argument following logical reasoning to make people believe the argument is not true but false. Your argument should differ from the following arguments: {good_argu}."}
        ]
        
        prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        result = pipe(prompt, max_new_tokens=max_length, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)
        
        for res in result:
            answer = result[0]['generated_text'].split("\n")[-1]
        examples.append(answer)
        print(f"Example {i+1} generated.")
    
    return examples
