import torch
import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer
# $env:PYTHONUTF8 = 1
# 强制使用GPU；如果没有可用的GPU，这将抛出错误。
if not torch.cuda.is_available():
    raise RuntimeError("此脚本需要 GPU 才能运行.")
device = torch.device("cuda")

# 加载模型和分词器，并将它们移动到GPU上
model = AutoModelForCausalLM.from_pretrained('../chatglm', trust_remote_code=True).to(device)
tokenizer = AutoTokenizer.from_pretrained('../chatglm', trust_remote_code=True)


def generate_text(prompt):
    # 确保输入数据也在GPU上
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    # 生成回复
    outputs = model.generate(inputs, max_length=200)
    # 将输出解码为文本，并返回
    reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return reply


# 定义Gradio界面
iface = gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text",
    title="ChatGLM3-6B Demo",
    description="键入内容，模型将回复!",
)

# 启动Gradio界面，可以选择开启share=True来创建一个公共链接
iface.launch(share=True)
