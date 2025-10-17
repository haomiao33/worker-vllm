import io
import base64
import torch
import requests
from PIL import Image
from diffusers import DiffusionPipeline, FlowMatchEulerDiscreteScheduler
import runpod

# 模型本地路径（Docker 构建时已下载）
MODEL_PATH = "./Qwen-Image-Edit-Lightning"

print("⏳ Loading model from local path...")
scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(MODEL_PATH, subfolder="scheduler")
pipe = DiffusionPipeline.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    scheduler=scheduler
).to("cuda")
pipe.scheduler.set_timesteps(8)
pipe.set_progress_bar_config(disable=True)
print("✅ Model loaded and ready.")


def download_image(url: str) -> Image.Image:
    """从 URL 下载图片"""
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
    return img


def process(job):
    data = job["input"]

    try:
        prompt = data.get("prompt", "enhance image quality, realistic lighting")
        image_url = data.get("image_url")
        if not image_url:
            return {"error": "missing image_url"}

        # 下载输入图
        image = download_image(image_url)

        # 可选 mask_url
        mask_url = data.get("mask_url")
        mask = download_image(mask_url) if mask_url else None

        # 执行编辑推理
        result = pipe(
            prompt=prompt,
            image=image,
            mask_image=mask,
            num_inference_steps=8
        ).images[0]

        # 输出 base64
        buf = io.BytesIO()
        result.save(buf, format="PNG")
        output_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        return {"output_image": output_b64}

    except Exception as e:
        return {"error": str(e)}


runpod.serverless.start({"handler": process})
