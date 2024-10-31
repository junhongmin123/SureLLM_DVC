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
    while True:
        model_type = input("Enter model type 'code2chart' or 'model2doc': ")
        if any(word == model_type.strip().lower() for word in ["code2chart", "model2doc"]):
            break
    
    if model_type == "code2chart":
        yaml_path = "training_args.yaml"
        model_key = "output_dir"
    elif model_type == "model2doc":
        yaml_path = "model2doc_yaml/export.yaml"
        model_key = "export_dir"
    
    with open(yaml_path, 'r') as file:
        yaml_content = yaml.safe_load(file)
    
    model_name = yaml_content[model_key].split("/")[-1]
    model_path = f"model/{model_name}"
    
    print("\nloading model...")
    llm_model, llm_tokenizer = load_model(model_path)

    sys_prompt = input("Set system prompt: ")
        
    print("\nStarting chat...")
    print("-- enter 'setting' to set inference setting --")
    print("-- enter 'quit' or 'exit' to finish --")
    
    while True:
        print()
        print("----------------------------------------------")
        print("- Option: 'setting', 'quit', 'exit'          -")
        print("----------------------------------------------")
        user_input = input("\nuser: ")
        if "setting" in user_input.strip().lower():
            sys_prompt = input("Set system prompt: ")
        elif any(word in user_input.strip().lower() for word in ["quit", "exit"]):
            print("Exiting chat...")
            break
        else:
            response = model_response(llm_tokenizer, sys_prompt, user_input, llm_model)
            print(f"\nassistant: {response}")
            
if __name__ == '__main__':
    inference()