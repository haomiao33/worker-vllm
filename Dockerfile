FROM nvidia/cuda:12.1.0-base-ubuntu22.04 

RUN apt-get update -y \
    && apt-get install -y python3-pip

RUN ldconfig /usr/local/cuda-12.1/compat/

WORKDIR /app
COPY requirements.txt .

RUN pip install -U pip && pip install -r requirements.txt

# 下载模型到本地目录，避免冷启动

RUN huggingface-cli download lightx2v/Qwen-Image-Lightning --local-dir ./Qwen-Image-Lightning 
COPY app.py /app

CMD ["python3", "app.py"]