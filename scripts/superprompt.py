import os
import requests
import time  # 引入time模块
import torch
import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration
import gradio as gr
from modules import scripts

model_folder = ""
tokenizer = ""
model = ""
num_return_sequences = 1
SEED_LIMIT_NUMPY = 2**32

def set_seed(seed: int) -> None:
    seed = int(seed) % SEED_LIMIT_NUMPY
    torch.manual_seed(seed)
    transformers.set_seed(seed)

def modelcheck():
    global tokenizer
    global model
    global num_return_sequences

    try:
        tokenizer = T5Tokenizer.from_pretrained("roborovski/superprompt-v1", legacy=False, clean_up_tokenization_spaces=False)
        model = T5ForConditionalGeneration.from_pretrained("roborovski/superprompt-v1", device_map="auto")
        return True
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return False
        
def generate_super_prompt(prompt, max_new_tokens=128, seed=123456):
    model_found = modelcheck()
    if model_found:
        # 设置种子值
        if seed is not None:
            set_seed(seed)  # 使用提供的种子设置函数

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True  # 启用采样
        )
        generated_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_prompt
    else:
        gr.Warning("未找到模型。请查看命令控制台信息")
        return ""
