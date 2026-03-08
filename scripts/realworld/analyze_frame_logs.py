#!/usr/bin/env python3
"""
解析 internvla_client_*.log 和 internvla_server_*.log，统计计数、延迟拆账、同步误差、动作分布。

用法:
  python3 scripts/realworld/analyze_frame_logs.py [client_log_path] [server_log_path]

若未指定路径，则自动查找 logs/ 目录下最新的 client 和 server 日志。
"""
import ast
import json
import os
import sys
from pathlib import Path


def _p95(lst):
    if not lst:
        return None
    s = sorted(lst)
    idx = int(len(s) * 0.95) - 1
    return s[max(0, idx)]


def _safe_float(x):
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_events(log_path: str):
    """加载日志中所有事件，返回事件列表。"""
    if not log_path or not os.path.exists(log_path):
        return []
    events = []
    with open(log_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                events.append(rec)
            except json.JSONDecodeError:
                continue
    return events


def count_events(events, event_name: str) -> int:
    return sum(1 for e in events if e.get('event') == event_name)


def main():
    log_dir = Path(__file__).parent / 'logs'
    client_log = sys.argv[1] if len(sys.argv) > 1 else None
    server_log = sys.argv[2] if len(sys.argv) > 2 else None

    if client_log is None and log_dir.exists():
        clients = sorted(log_dir.glob('internvla_client_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
        client_log = str(clients[0]) if clients else None
    if server_log is None and log_dir.exists():
        servers = sorted(log_dir.glob('internvla_server_*.log'), key=lambda p: p.stat().st_mtime, reverse=True)
        server_log = str(servers[0]) if servers else None

    client_events = load_events(client_log) if client_log else []
    server_events = load_events(server_log) if server_log else []

    print("=" * 70)
    print("InternVLA 帧诊断统计")
    print("=" * 70)

    # ----- 1. 基础计数 -----
    print("\n【1. 基础计数】")
    if client_log:
        print(f"  [Client] {client_log}")
        shutdown = next((e for e in client_events if e.get('event') == 'client_shutdown'), {})
        callback_cnt = shutdown.get('callback_count') or count_events(client_events, 'rgb_depth_down_callback')
        post_cnt = shutdown.get('post_count') or count_events(client_events, 'planning_before_send')
        publish_cnt = shutdown.get('publish_count') or count_events(client_events, 'client_publish')
        print(f"    rgb_depth_down_callback: {callback_cnt}")
        print(f"    planning_thread POST:    {post_cnt}")
        print(f"    client 发布新控制目标:  {publish_cnt}")
    if server_log:
        print(f"  [Server] {server_log}")
        server_recv_cnt = count_events(server_events, 'server_recv')
        print(f"    /eval_dual 收到:         {server_recv_cnt}")

    # ----- 2. 按 frame_id 匹配，算 4 个时间 + 2 个同步误差 -----
    planning_sends = {e['frame_id']: e for e in client_events if e.get('event') == 'planning_before_send'}
    client_recvs = {e['frame_id']: e for e in client_events if e.get('event') == 'client_recv'}
    server_ends = {e['frame_id']: e for e in server_events if e.get('event') == 'server_model_end'}

    frame_ids = sorted(set(planning_sends.keys()) & set(client_recvs.keys()) & set(server_ends.keys()))

    rtt_list = []
    t_model_list = []
    t_server_pre_list = []
    t_other_list = []
    rgb_depth_diff_list = []
    rgb_odom_diff_list = []

    for fid in frame_ids:
        ps = planning_sends[fid]
        cr = client_recvs[fid]
        se = server_ends[fid]

        t_http_send = _safe_float(ps.get('t_http_send'))
        t_client_recv = _safe_float(cr.get('t_client_recv'))
        t_server_recv = _safe_float(se.get('t_server_recv'))
        t_model_start = _safe_float(se.get('t_model_start'))
        t_model_end = _safe_float(se.get('t_model_end'))

        if t_http_send is not None and t_client_recv is not None:
            rtt = t_client_recv - t_http_send
            rtt_list.append(rtt)
        if t_model_start is not None and t_model_end is not None:
            t_model_list.append(t_model_end - t_model_start)
        if t_server_recv is not None and t_model_start is not None:
            t_server_pre_list.append(t_model_start - t_server_recv)

        rgb_ts = _safe_float(ps.get('rgb_ts_header'))
        depth_ts = _safe_float(ps.get('depth_ts_header'))
        odom_ts = _safe_float(ps.get('odom_ts_used'))
        if rgb_ts is not None and depth_ts is not None:
            rgb_depth_diff_list.append(abs(rgb_ts - depth_ts))
        if rgb_ts is not None and odom_ts is not None:
            rgb_odom_diff_list.append(abs(rgb_ts - odom_ts))

    # T_other 按 frame 逐帧算
    t_other_list = []
    for fid in frame_ids:
        ps = planning_sends[fid]
        cr = client_recvs[fid]
        se = server_ends[fid]
        t_http_send = _safe_float(ps.get('t_http_send'))
        t_client_recv = _safe_float(cr.get('t_client_recv'))
        t_model_start = _safe_float(se.get('t_model_start'))
        t_model_end = _safe_float(se.get('t_model_end'))
        if all(x is not None for x in [t_http_send, t_client_recv, t_model_start, t_model_end]):
            rtt = t_client_recv - t_http_send
            t_model = t_model_end - t_model_start
            t_other_list.append(rtt - t_model)

    # ----- 3. 延迟拆账输出 -----
    print("\n【2. 延迟拆账 (按 frame_id)】")
    if frame_ids:
        print("  A. RTT (client 往返) = t_client_recv - t_http_send")
        print(f"     均值: {sum(rtt_list)/len(rtt_list)*1000:.1f} ms,  p95: {_p95(rtt_list)*1000:.1f} ms,  最大: {max(rtt_list)*1000:.1f} ms")
        print("  B. T_model (server 模型推理) = t_model_end - t_model_start")
        print(f"     均值: {sum(t_model_list)/len(t_model_list)*1000:.1f} ms,  p95: {_p95(t_model_list)*1000:.1f} ms,  最大: {max(t_model_list)*1000:.1f} ms")
        print("  C. T_server_pre (收包/解码/预处理) = t_model_start - t_server_recv")
        print(f"     均值: {sum(t_server_pre_list)/len(t_server_pre_list)*1000:.1f} ms,  p95: {_p95(t_server_pre_list)*1000:.1f} ms,  最大: {max(t_server_pre_list)*1000:.1f} ms")
        print("  D. T_other ≈ RTT - T_model (网络+编解码等)")
        if t_other_list:
            print(f"     均值: {sum(t_other_list)/len(t_other_list)*1000:.1f} ms,  p95: {_p95(t_other_list)*1000:.1f} ms,  最大: {max(t_other_list)*1000:.1f} ms")
    else:
        print("  (无匹配的 frame_id，跳过)")

    # ----- 4. 同步误差 -----
    print("\n【3. 同步误差】")
    if rgb_depth_diff_list:
        print("  |rgb_ts_header - depth_ts_header| (s)")
        print(f"    均值: {sum(rgb_depth_diff_list)/len(rgb_depth_diff_list)*1000:.1f} ms,  p95: {_p95(rgb_depth_diff_list)*1000:.1f} ms,  最大: {max(rgb_depth_diff_list)*1000:.1f} ms")
    if rgb_odom_diff_list:
        print("  |rgb_ts_header - odom_ts_used| (s)")
        print(f"    均值: {sum(rgb_odom_diff_list)/len(rgb_odom_diff_list)*1000:.1f} ms,  p95: {_p95(rgb_odom_diff_list)*1000:.1f} ms,  最大: {max(rgb_odom_diff_list)*1000:.1f} ms")
    if not rgb_depth_diff_list and not rgb_odom_diff_list:
        print("  (无数据)")

    # ----- 5. callback 间隔 -----
    print("\n【4. callback 间隔 (rgb_depth_down_callback)】")
    callbacks = [e for e in client_events if e.get('event') == 'rgb_depth_down_callback']
    t_monos = [_safe_float(e.get('t_mono')) for e in callbacks]
    t_monos = [t for t in t_monos if t is not None]
    intervals = [t_monos[i+1] - t_monos[i] for i in range(len(t_monos)-1) if t_monos[i+1] > t_monos[i]]
    if intervals:
        print(f"    均值: {sum(intervals)/len(intervals)*1000:.1f} ms,  p95: {_p95(intervals)*1000:.1f} ms,  最大: {max(intervals)*1000:.1f} ms")
    else:
        print("  (无数据)")

    # ----- 6. 动作分布 -----
    print("\n【5. 动作分布】")
    traj_count = 0
    discrete_count = 0
    stop_count = 0
    turn_left_count = 0
    turn_right_count = 0
    pixel_goal_count = 0
    traj_lens = []

    for e in server_events:
        if e.get('event') != 'server_model_end':
            continue
        at = e.get('action_type', '')
        av = e.get('action_value', '')
        pg = e.get('pixel_goal')
        tl = _safe_float(e.get('traj_len'))

        if at == 'trajectory':
            traj_count += 1
            if tl is not None:
                traj_lens.append(tl)
            if pg is not None and str(pg) != 'None':
                pixel_goal_count += 1
        elif at == 'discrete_action':
            discrete_count += 1
            try:
                actions = ast.literal_eval(av) if isinstance(av, str) else av
                if actions == [0]:
                    stop_count += 1
                elif 2 in actions:
                    turn_left_count += 1
                elif 3 in actions:
                    turn_right_count += 1
            except (ValueError, TypeError):
                pass

    total_actions = traj_count + discrete_count
    print(f"  trajectory 次数:     {traj_count}")
    print(f"  discrete_action 次数: {discrete_count}")
    print(f"  其中 [0] STOP:      {stop_count}")
    print(f"  [2] 左转:           {turn_left_count},  [3] 右转: {turn_right_count}")
    print(f"  有 pixel_goal 比例: {pixel_goal_count}/{traj_count} = {pixel_goal_count/max(1,traj_count)*100:.1f}%")
    if traj_lens:
        print(f"  traj_len 分布: 均值 {sum(traj_lens)/len(traj_lens):.3f} m,  min {min(traj_lens):.3f},  max {max(traj_lens):.3f}")

    # ----- 7. STOP 相关性 -----
    print("\n【6. STOP 相关性分析】")
    stop_frames = []
    for fid in frame_ids:
        se = server_ends.get(fid)
        if not se or se.get('action_type') != 'discrete_action':
            continue
        av = se.get('action_value', '')
        try:
            actions = ast.literal_eval(av) if isinstance(av, str) else av
            if actions == [0]:
                stop_frames.append(fid)
        except (ValueError, TypeError):
            pass

    if stop_frames:
        stop_rtts = []
        stop_rgb_odom = []
        stop_has_pixel = []
        for fid in stop_frames:
            if fid in planning_sends and fid in client_recvs and fid in server_ends:
                ps, cr, se = planning_sends[fid], client_recvs[fid], server_ends[fid]
                t_http = _safe_float(ps.get('t_http_send'))
                t_recv = _safe_float(cr.get('t_client_recv'))
                if t_http is not None and t_recv is not None:
                    stop_rtts.append((t_recv - t_http) * 1000)
                rgb_ts = _safe_float(ps.get('rgb_ts_header'))
                odom_ts = _safe_float(ps.get('odom_ts_used'))
                if rgb_ts is not None and odom_ts is not None:
                    stop_rgb_odom.append(abs(rgb_ts - odom_ts) * 1000)
                pg = se.get('pixel_goal')
                stop_has_pixel.append(pg is not None and str(pg) != 'None')

        print(f"  STOP 共 {len(stop_frames)} 次")
        if stop_rtts:
            print(f"  STOP 前 RTT: 均值 {sum(stop_rtts)/len(stop_rtts):.0f} ms,  p95 {_p95(stop_rtts):.0f} ms")
        if stop_rgb_odom:
            print(f"  STOP 前 rgb-odom 差: 均值 {sum(stop_rgb_odom)/len(stop_rgb_odom):.1f} ms")
        if stop_has_pixel:
            print(f"  STOP 时 pixel_goal 非空比例: {sum(stop_has_pixel)}/{len(stop_has_pixel)}")
        if stop_rtts and rtt_list:
            avg_rtt_all = sum(rtt_list) / len(rtt_list) * 1000
            avg_rtt_stop = sum(stop_rtts) / len(stop_rtts)
            if avg_rtt_stop > avg_rtt_all * 1.2:
                print("  -> STOP 与高 RTT 强相关，更像链路问题")
            else:
                print("  -> STOP 在延迟正常时也出现，更像模型/输入/标定问题")
    else:
        print("  (无 STOP 样本)")

    print("\n" + "-" * 70)
    print("解读: callback 不稳->传感器; RTT/T_model 大->推理或网络; rgb-odom 差大->位姿不同步")
    print("=" * 70)


if __name__ == '__main__':
    main()
