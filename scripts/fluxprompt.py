import os
import time  # 引入time模块
import torch
import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
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
        # 加载Flux-Prompt-Enhance模型和分词器
        tokenizer = AutoTokenizer.from_pretrained("gokaygokay/Flux-Prompt-Enhance", legacy=False, clean_up_tokenization_spaces=False)
        model = AutoModelForSeq2SeqLM.from_pretrained("gokaygokay/Flux-Prompt-Enhance", device_map="auto")
        return True
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return False
        
def generate_flux_prompt(prompt, max_new_tokens=256, seed=123456):
    model_found = modelcheck()
    if model_found:
        # 设置种子值
        if seed is not None:
            set_seed(seed)  # 使用提供的种子设置函数

        # 在提示词前添加前缀
        prefix = "enhance prompt: "
        full_prompt = prefix + prompt

        input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids.to("cuda")
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            repetition_penalty=1.2,  # 设置重复惩罚
            top_k=50,
            top_p=0.95,
            do_sample=True  # 启用采样
        )
        enhanced_prompt = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return enhanced_prompt
    else:
        gr.Warning("未找到模型。请查看命令控制台信息")
        return ""
