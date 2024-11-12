import os
import torch
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model):
    llm_model = AutoModelForCausalLM.from_pretrained(model).to('cuda')
    llm_tokenizer = AutoTokenizer.from_pretrained(
        model, 
        torch_dtype="auto",
        device_map="auto"
    )
    llm_model.eval()
    return llm_model, llm_tokenizer

def model_response(tokenizer, sys_prompt, user_prompt, model):
    chat = [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    chat = tokenizer.apply_chat_template(
        chat,
        tokenize=False
    )
    
    model_input = tokenizer(chat, return_tensors="pt").to('cuda')
    with torch.no_grad():
        response = tokenizer.decode(
            model.generate(
                **model_input,
                do_sample=True,
                max_new_tokens=1024,
                pad_token_id=tokenizer.eos_token_id,
            )[0],
            skip_special_tokens = True
        )
    output = response.split(user_prompt)[-1]
    output = output.split("assistant\n")[-1]
    return output

def inference():
    model_list = os.listdir("model/")
    if len(model_list) == 0:
        print("* There is NO model in the model directory.\n** Enter: 'dvc pull -j 100' to download the model.")
    elif len(model_list) == 1:
        model_path = model_list[0]
    else:
        print("* There are multiple models in the model directory.")
        print("* Please select a model to load\n")
        for i, model in enumerate(model_list):
            print(f"{i+1}. {model}")
        model_index = int(input("\nEnter model index: ")) - 1
        model_path = model_list[model_index]
    
    print(f"\nLoading model: {model_path}")
    try:
        llm_model, llm_tokenizer = load_model(model_path)
    except:
        llm_model, llm_tokenizer = load_model(f"model/{model_path}")
        
    sys_prompt = input("Set system prompt: ")
    pre_prompt = input("Set pre-prompt: ")
        
    print("\nStarting chat...")
    print("-- Enter 'Setting' to set inference setting --")
    print("-- Enter 'Quit' or 'Exit' to finish --")
    
    while True:
        print()
        print("----------------------------------------------")
        print("- Option: 'Setting', 'Quit', 'Exit'          -")
        print("----------------------------------------------")
        user_input = input("\nuser: ")
        if "setting" == user_input.strip().lower():
            sys_prompt = input("Set system prompt: ")
            pre_prompt = input("Set pre-prompt: ")
        elif any(word == user_input.strip().lower() for word in ["quit", "exit"]):
            print("Exiting chat...")
            break
        else:
            response = model_response(llm_tokenizer, sys_prompt, pre_prompt + user_input, llm_model)
            print(f"\nassistant: {response}")
            
if __name__ == '__main__':
    inference()