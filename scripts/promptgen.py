import os
import requests
import time  # 引入time模块
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import gradio as gr

model_folder = ""
tokenizer = ""
model = ""
num_return_sequences = 1

def modelcheck():
    global tokenizer
    global model
    global num_return_sequences

    try:
        tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2', clean_up_tokenization_spaces=False)
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model = GPT2LMHeadModel.from_pretrained('distilgpt2')
        return True
    except Exception as e:
        print(f"加载模型时出错: {str(e)}")
        return False

def generate(prompt, temperature, top_k, style_max_length, repetition_penalty, usecomma):
    model_found = modelcheck()
    if model_found:
        input_ids = tokenizer(prompt, return_tensors='pt').input_ids
        # 设置 attention_mask，以指示哪些输入标记是有效的
        attention_mask = input_ids.ne(tokenizer.pad_token_id)
        if not usecomma:
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,  # 添加 attention_mask
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                max_length=style_max_length,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                penalty_alpha=0.6,
                no_repeat_ngram_size=1,
                pad_token_id=tokenizer.eos_token_id,  # 设置 pad_token_id
                early_stopping=False
            )
        else:
            output = model.generate(
                input_ids,
                attention_mask=attention_mask,  # 添加 attention_mask
                do_sample=True,
                temperature=temperature,
                top_k=top_k,
                max_length=style_max_length,
                num_return_sequences=num_return_sequences,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,  # 设置 pad_token_id
                early_stopping=True
            )
        return tokenizer.decode(output[0], skip_special_tokens=True)
    else:
        gr.Warning("未找到模型。请查看命令控制台信息")
        return ""
