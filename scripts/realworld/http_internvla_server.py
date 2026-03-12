import argparse
import io
import json
import os
import threading
import time
from datetime import datetime

import numpy as np

# ========== 诊断日志：写入文件 ==========
_log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(_log_dir, exist_ok=True)
_server_log_path = os.path.join(_log_dir, f'internvla_server_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
_server_log_file = None
_server_log_lock = threading.Lock()
server_recv_count = 0


def _server_log(event: str, **kwargs):
    global _server_log_file
    with _server_log_lock:
        if _server_log_file is None:
            _server_log_file = open(_server_log_path, 'a', encoding='utf-8')
        rec = {'event': event, 't_mono': time.monotonic(), **kwargs}
        _server_log_file.write(json.dumps(rec, ensure_ascii=False) + '\n')
        _server_log_file.flush()
from flask import Flask, Response, jsonify, request
from PIL import Image, ImageDraw

from internnav.agent.internvla_n1_agent_realworld import InternVLAN1AsyncAgent

app = Flask(__name__)
idx = 0
start_time = time.time()
output_dir = ''
# 第一人称视角：保存最近一帧 RGB（JPEG bytes），供 GET /fpv 使用
_last_fpv_jpeg = None
# stream 保存路径：/stream/<timestamp>/frame_xxx.jpg
_stream_run_dir = None


@app.route("/eval_dual", methods=['POST'])
def eval_dual():
    global idx, output_dir, start_time, _last_fpv_jpeg, _stream_run_dir, server_recv_count
    t_server_recv = time.monotonic()
    server_recv_count += 1

    image_file = request.files['image']
    depth_file = request.files['depth']
    json_data = request.form['json']
    data = json.loads(json_data)
    frame_id = data.get('frame_id', -1)

    _server_log('server_recv', frame_id=frame_id, t_server_recv=t_server_recv, server_recv_count=server_recv_count)

    start_time = time.time()

    image = Image.open(image_file.stream)
    image = image.convert('RGB')
    image = np.asarray(image)

    depth = Image.open(depth_file.stream)
    depth = depth.convert('I')
    depth = np.asarray(depth)
    depth = depth.astype(np.float32) / 10000.0
    print(f"read http data cost {time.time() - start_time}")

    camera_pose = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    instruction = "Task Objective: 1. Move straight ahead. 2. When a chair is detected, turn right 90 degrees and exit. 3. Walk straight. 4. Turn around at the end of the corridor. 5. Return. 6. When a chair is detected, stop"
    policy_init = data['reset']
    if policy_init:
        start_time = time.time()
        idx = 0
        output_dir = 'output/runs' + datetime.now().strftime('%m-%d-%H%M')
        os.makedirs(output_dir, exist_ok=True)
        _stream_run_dir = os.path.join('/home/ubuntu1/backup/InternNav/stream',
                                       datetime.now().strftime('%Y-%m-%d-%H%M%S'))
        os.makedirs(_stream_run_dir, exist_ok=True)
        print("init reset model!!!")
        agent.reset()

    idx += 1

    look_down = False
    t_model_start = time.monotonic()
    dual_sys_output = {}

    dual_sys_output = agent.step(
        image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
    )
    if dual_sys_output.output_action is not None and dual_sys_output.output_action == [5]:
        look_down = True
        dual_sys_output = agent.step(
            image, depth, camera_pose, instruction, intrinsic=args.camera_intrinsic, look_down=look_down
        )

    t_model_end = time.monotonic()

    json_output = {}
    if dual_sys_output.output_action is not None:
        json_output['discrete_action'] = dual_sys_output.output_action
        action_type = 'discrete_action'
        action_value = dual_sys_output.output_action
        pixel_goal = None
        traj_len = 0.0
    else:
        json_output['trajectory'] = dual_sys_output.output_trajectory.tolist()
        if dual_sys_output.output_pixel is not None:
            json_output['pixel_goal'] = dual_sys_output.output_pixel
        action_type = 'trajectory'
        action_value = dual_sys_output.output_trajectory.tolist()
        pixel_goal = dual_sys_output.output_pixel
        traj_len = float(np.linalg.norm(dual_sys_output.output_trajectory[-1][:2])) if dual_sys_output.output_trajectory is not None and len(dual_sys_output.output_trajectory) > 0 else 0.0

    _server_log('server_model_end', frame_id=frame_id, t_server_recv=t_server_recv,
                t_model_start=t_model_start, t_model_end=t_model_end,
                action_type=action_type, action_value=str(action_value),
                pixel_goal=str(pixel_goal) if pixel_goal is not None else None,
                traj_len=traj_len, server_recv_count=server_recv_count)

    # 更新 FPV：在画面上绘制 grounded waypoint 红点后保存
    fpv_img = Image.fromarray(image).convert('RGB')
    draw = ImageDraw.Draw(fpv_img)
    h_img, w_img = image.shape[:2]
    goal_px = None
    if dual_sys_output.output_pixel is not None:
        # pixel_goal 在 agent 的 384x384 空间，需缩放到原图
        px = int(dual_sys_output.output_pixel[1] * w_img / args.resize_w)
        py = int(dual_sys_output.output_pixel[0] * h_img / args.resize_h)
        goal_px = (px, py)
    elif dual_sys_output.output_trajectory is not None and len(dual_sys_output.output_trajectory) > 0:
        # 仅 trajectory：将最后一个点从 body 系 (x,y) 米 投影到像素
        traj = dual_sys_output.output_trajectory
        x_end, y_end = float(traj[-1][0]), float(traj[-1][1])
        x_end = max(x_end, 0.15)  # 避免除零
        K = args.camera_intrinsic
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        cam_h = getattr(args, 'camera_height', 0.4)
        px = int(fx * y_end / x_end + cx)
        py = int(fy * cam_h / x_end + cy)
        if 0 <= px < w_img and 0 <= py < h_img:
            goal_px = (px, py)
    if goal_px is not None:
        r = 10
        draw.ellipse([goal_px[0] - r, goal_px[1] - r, goal_px[0] + r, goal_px[1] + r], outline='red', fill='red', width=3)
    buf = io.BytesIO()
    fpv_img.save(buf, format='JPEG', quality=85)
    _last_fpv_jpeg = buf.getvalue()

    # 保存帧到 stream/<timestamp>/ 目录
    if _stream_run_dir is None:
        _stream_run_dir = os.path.join('/home/ubuntu1/backup/InternNav/stream',
                                        datetime.now().strftime('%Y-%m-%d-%H%M%S'))
        os.makedirs(_stream_run_dir, exist_ok=True)
    fpv_img.save(os.path.join(_stream_run_dir, f'frame_{idx:06d}.jpg'), format='JPEG', quality=85)

    t1 = time.monotonic()
    generate_time = t1 - t_model_start
    print(f"dual sys step {generate_time}")
    print(f"json_output {json_output}")
    return jsonify(json_output)


@app.route("/fpv", methods=['GET'])
def fpv():
    """第一人称视角：返回最近一帧 RGB（JPEG）。"""
    global _last_fpv_jpeg
    if _last_fpv_jpeg is None:
        return "No frame yet. Run client to send images.", 404
    return Response(_last_fpv_jpeg, mimetype='image/jpeg')


@app.route("/fpv_live", methods=['GET'])
def fpv_live():
    """第一人称视角页面：自动刷新画面。浏览器打开 http://<server_ip>:5801/fpv_live 即可看 FPV。"""
    html = """
    <!DOCTYPE html>
    <html><head><meta charset="utf-8"><title>FPV</title></head>
    <body style="margin:0;background:#000;">
    <img id="img" src="/fpv" style="width:100%;height:100%;object-fit:contain;" />
    <!-- 注释掉自动刷新，避免在无帧时刷屏 404；改用 RViz2 做可视化验证 -->
    <!--
    <script>
    setInterval(function(){
        var i = document.getElementById('img');
        i.src = '/fpv?t=' + Date.now();
    }, 300);
    </script>
    -->
    </body></html>
    """
    return Response(html, mimetype='text/html; charset=utf-8')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_path", type=str, default="checkpoints/InternVLA-N1-DualVLN")
    parser.add_argument(
        "--attn_backend",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
        help="Attention backend passed to the DualVLN loader",
    )
    parser.add_argument(
        "--processor_use_fast",
        type=str,
        default="auto",
        choices=["auto", "true", "false"],
        help="Forward use_fast to AutoProcessor when not auto",
    )
    parser.add_argument("--resize_w", type=int, default=384)
    parser.add_argument("--resize_h", type=int, default=384)
    parser.add_argument("--num_history", type=int, default=8)
    parser.add_argument("--plan_step_gap", type=int, default=4)
    parser.add_argument(
        "--kv_cache_mode",
        type=str,
        default="disabled",
        choices=["disabled", "lookdown_experimental"],
        help="Experimental KV cache mode for look_down continuation",
    )
    parser.add_argument("--kv_cache_debug", action="store_true", help="Print KV cache debug information")
    parser.add_argument("--camera_height", type=float, default=0.4,
                       help="相机离地高度(m)，用于 trajectory->pixel 投影（无 pixel_goal 时）")
    args = parser.parse_args()

    args.camera_intrinsic = np.array(
        [[386.5, 0.0, 328.9, 0.0], [0.0, 386.5, 244, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )
    agent = InternVLAN1AsyncAgent(args)
    # Warmup: 使用 uint8 的 RGB 和 float32 的 depth，与 PIL/模型期望一致
    agent.step(
        np.zeros((480, 640, 3), dtype=np.uint8),
        np.zeros((480, 640), dtype=np.float32),
        np.eye(4),
        "hello",
        intrinsic=args.camera_intrinsic,
    )
    agent.reset()

    print(f"[InternVLA Server] 诊断日志将写入: {_server_log_path}")
    app.run(host='0.0.0.0', port=5801)
