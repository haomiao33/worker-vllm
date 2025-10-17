import io
import os
import base64
import torch
import requests
from PIL import Image
from diffusers import (
    QwenImageEditPipeline,
    FlowMatchEulerDiscreteScheduler,
)
from diffusers.models import QwenImageTransformer2DModel
import runpod

# --- 初始化模型 ---
BASE_MODEL = "Qwen/Qwen-Image-Edit"
LORA_PATH = "Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors"

if not os.path.exists(LORA_PATH):
    os.makedirs("Qwen-Image-Lightning", exist_ok=True)
    os.system(
        "huggingface-cli download Qwen-Image-Lightning/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors "
        "--local-dir Qwen-Image-Lightning"
    )

if torch.cuda.is_available():
    dtype = torch.bfloat16
    device = "cuda"
else:
    dtype = torch.float32
    device = "cpu"

# 自定义 scheduler 配置
scheduler_config = {
    "base_image_seq_len": 256,
    "base_shift": 1.0986,  # log(3)
    "invert_sigmas": False,
    "max_image_seq_len": 8192,
    "max_shift": 1.0986,
    "num_train_timesteps": 1000,
    "shift": 1.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}

scheduler = FlowMatchEulerDiscreteScheduler.from_config(scheduler_config)
model = QwenImageTransformer2DModel.from_pretrained(BASE_MODEL, subfolder="transformer", torch_dtype=dtype)
pipe = QwenImageEditPipeline.from_pretrained(BASE_MODEL, transformer=model, scheduler=scheduler, torch_dtype=dtype)
pipe.load_lora_weights(LORA_PATH)
pipe.to(device)

pipe.scheduler.set_timesteps(8)

# --- 工具函数 ---
def download_image(url: str) -> Image.Image:
    response = requests.get(url)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content)).convert("RGB")

def image_to_base64(img: Image.Image) -> str:
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")

# --- RunPod handler ---
def handler(job):
    data = job["input"]
    prompt = data.get("prompt", "")
    image_url = data.get("image_url")
    if not image_url:
        return {"error": "image_url is required"}

    image = download_image(image_url)

    result = pipe(
        prompt=prompt,
        image=image,
        num_inference_steps=8,
        true_cfg_scale=1.0,
        negative_prompt=" ",
        generator=torch.Generator(device=device).manual_seed(42),
    ).images[0]

    return {"output_base64": image_to_base64(result)}

runpod.serverless.start({"handler": handler})
