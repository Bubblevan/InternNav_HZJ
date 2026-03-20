# DualVLN vLLM `prompt_embeds` 调试 Runbook

这份文档记录当前为了排查 `prompt_embeds` 路线，在本地 `/root/backup/vllm` 源码里加的文件级 dump 开关。

目标不是长期保留这些调试输出，而是为了回答一个非常具体的问题：

> 外部传入的 `prompt_embeds`，到底有没有真正进入 vLLM 的 backbone forward？

---

## 1. 已加的埋点

当前埋点位置：

- [gpu_input_batch.py](/root/backup/vllm/vllm/v1/worker/gpu_input_batch.py)
  - `add_request()`
  - 记录 request 入队时的：
    - `prompt_token_ids`
    - `prompt_embeds`
    - prompt 段 `is_token_ids`

- [gpu_model_runner.py](/root/backup/vllm/vllm/v1/worker/gpu_model_runner.py)
  - `_prepare_inputs()`
    - 记录准备好的：
      - `input_ids`
      - `is_token_ids`
      - `inputs_embeds`
  - `execute_model()` forward 前
    - 记录真正送进模型的：
      - `input_ids`
      - `inputs_embeds`
      - `positions`
  - `execute_model()` forward 后
    - 记录：
      - `hidden_states`
  - pooler 后
    - 记录：
      - `pooler_output`

配套调试工具：

- [debug_dump.py](/root/backup/vllm/vllm/debug_dump.py)

---

## 2. 如何开启

只在设置下面环境变量时生效：

```bash
export VLLM_DEBUG_DUMP_DIR=/root/backup/InternNav/logs/habitat/vllm_debug_dump
```

可选项：

```bash
export VLLM_DEBUG_DUMP_PREFIX=prompt_embeds_probe
export VLLM_DEBUG_DUMP_FULL_TENSORS=0
export VLLM_DEBUG_DUMP_SLICE_ROWS=8
```

含义：

- `VLLM_DEBUG_DUMP_DIR`
  - dump 输出目录，必须设置
- `VLLM_DEBUG_DUMP_PREFIX`
  - 文件名前缀
- `VLLM_DEBUG_DUMP_FULL_TENSORS=1`
  - 保存完整 tensor
- `VLLM_DEBUG_DUMP_FULL_TENSORS=0`
  - 只保存最后几行切片，避免太大
- `VLLM_DEBUG_DUMP_SLICE_ROWS`
  - 切片行数，默认 `8`

---

## 3. 推荐运行方式

```bash
cd /root/backup/InternNav
source /root/.venv/bin/activate

export VLLM_DEBUG_DUMP_DIR=/root/backup/InternNav/logs/habitat/vllm_debug_dump
export VLLM_DEBUG_DUMP_PREFIX=prompt_embeds_probe
export VLLM_DEBUG_DUMP_FULL_TENSORS=0
export VLLM_DEBUG_DUMP_SLICE_ROWS=8

python scripts/eval/tools/probe_vllm_prompt_embeds_runtime.py \
  --model-path checkpoints/InternVLA-N1-DualVLN-qwen25vl-s2-view \
  --sample-pt logs/habitat/hf_generate_latents_baseline_replay1/samples/sample_0000_zsNo4HB9uLZ_0001_step_0003.pt \
  --append-traj-tokens \
  --output logs/habitat/vllm_prompt_embeds_probe_qwen25vl.json
```

---

## 4. 重点看哪些文件

最值得优先对比的是这几类：

1. `gpu_input_batch_add_request*.pt`
   - 看 request 入队时：
     - `prompt_embeds` 是否已经不同
     - prompt 段 `is_token_ids` 是否仍被错误标成全 `True`

2. `gpu_model_runner_prepare_inputs*.pt`
   - 看 `_prepare_inputs()` 后：
     - `inputs_embeds_gpu` 是否保留了自定义最后 4 行
     - `is_token_ids_gpu` 是否还把这些位置标成 token ids

3. `gpu_model_runner_pre_forward*.pt`
   - 这是最关键的一层：
     - 真正喂给 model 的 `inputs_embeds` 是否还带着自定义差异
     - `input_ids` 是否仍然被同时使用

4. `gpu_model_runner_post_forward*.pt`
   - 看 hidden states 是否已经出现差异

5. `gpu_model_runner_pooler_output*.pt`
   - 如果 hidden states 有差异，但 pooler output 没差异，说明问题在 pooler/head
   - 如果 hidden states 也没差异，说明问题更早，在 forward 输入链路

---

## 5. 当前最关心的判据

下一轮排查时，最关键的问题不是“最终 JSON 里 diff 是不是 0”，而是：

1. `pre_forward.inputs_embeds` 的最后 4 行，在 base/custom 两次请求里是否真的不同
2. `post_forward.hidden_states` 的最后 4 行，是否开始出现差异

如果：

- `pre_forward.inputs_embeds` 已经不同
- 但 `post_forward.hidden_states` 仍完全相同

那就说明问题已经非常接近模型/pooling 本体。

如果：

- `pre_forward.inputs_embeds` 就已经没差异

那就说明外部自定义 embeds 在进入真正 forward 之前又被覆盖掉了。
