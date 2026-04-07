# 文件名：vla_model.py
# 功能：负责加载庞大的模型，并将图像+文字转化为机械臂能懂的数字。
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import numpy as np

class OpenVLAgent:
    def __init__(self, model_id="openvla/openvla-7b"):
        self.device = "cuda:0" # 强制使用第一块显卡
        
        # 1. 加载处理器：负责把图片缩放到 224x224，把文字转为 Token
        print("正在加载 Processor...")
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

        # 2. 加载 4-bit 量化模型
        print("正在加载 4-bit 模型...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16,   # 使用半精度浮点数
            low_cpu_mem_usage=True,       # 优化内存分配
            trust_remote_code=True,       # 允许运行下载的自定义代码
            load_in_4bit=True,            # 核心：开启 4-bit 量化以节省显存
            device_map=self.device        # 自动映射到显卡
        )

    def predict_action(self, image, instruction):
        # 1. 预处理
        input_image = Image.fromarray(image)
        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        # 2. 生成 inputs (返回的是 float32 张量)
        inputs = self.processor(prompt, input_image, return_tensors="pt").to(self.device)

        # 3. 【关键修复】：将 inputs 中的浮点数张量全部转为 bfloat16，以匹配大脑
        # 注意：input_ids 是整数，不能转；只有 pixel_values 这种浮点数需要转
        for k, v in inputs.items():
            if torch.is_floating_point(v):
                inputs[k] = v.to(torch.bfloat16)

        # 4. 推理预测
        with torch.no_grad():
            action = self.model.predict_action(
                input_ids=inputs["input_ids"], 
                pixel_values=inputs["pixel_values"], 
                unnorm_key="bridge_orig", 
                do_sample=False
            )

        return action
