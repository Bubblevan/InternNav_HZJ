import base64
import io
import json
import cv2  # 需要安装 opencv-python
from PIL import Image
import requests

def pil_to_b64(img):
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# 视频路径
video_path = "/root/backup/InternNav/logs/habitat/ab_test_vllm/vis_0/2azQ1b91cZZ/0011.mp4"

# 使用 OpenCV 读取视频
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"无法打开视频文件: {video_path}")

# 读取第一帧
ret, frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("无法读取视频的第一帧")

# OpenCV 读取的帧是 BGR 格式，转换为 RGB 并转为 PIL Image
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
img = Image.fromarray(frame_rgb).convert("RGB")

# 后续与原来相同
payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "You are an autonomous navigation assistant. Where should you go next?"},
                {"type": "image", "image": pil_to_b64(img)},
            ],
        }
    ],
    "max_new_tokens": 64,
}

resp = requests.post("http://127.0.0.1:8000/dualvln/step_s2", json=payload, timeout=600)
print(resp.status_code)
print(resp.json().keys())
print(resp.json().get("llm_output"))
print(resp.json().get("pixel_goal"))