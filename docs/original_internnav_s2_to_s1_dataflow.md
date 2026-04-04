# Original InternNav S2 -> S1 Dataflow

This document describes the original dual-system execution path in `/root/InternNav`, with the goal of making handoff and strict alignment work easier. The emphasis is on the runtime path used by Habitat evaluation: System-2 (`S2`) produces either a discrete action string or a pixel goal, and System-1 (`S1`) consumes the latent representation behind that pixel-goal decision to roll out local actions.

## 1. File Map

The original chain is spread across these files:

1. Evaluator loop and online episode control:
   - [`/root/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py`](/root/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py)
2. Core DualVLN model:
   - [`/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py)
3. Cleaner policy wrapper that mirrors the same S2/S1 split:
   - [`/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1_policy.py`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1_policy.py)
4. System-1 submodule construction:
   - [`/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1_arch.py`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1_arch.py)
5. Helper functions used by both evaluator and policy:
   - [`/root/InternNav/internnav/model/utils/vln_utils.py`](/root/InternNav/internnav/model/utils/vln_utils.py)

## 2. High-Level Runtime Graph

The original runtime graph is:

1. Habitat produces `rgb`, `depth`, and episode metadata.
2. Evaluator builds a chat-style prompt with image history and the current view.
3. HF `processor.apply_chat_template(...)` renders the text prompt.
4. HF `processor(...)` converts prompt + images into:
   - `input_ids`
   - `pixel_values`
   - `image_grid_thw`
5. `InternVLAN1ForCausalLM.generate(...)` runs System-2 and returns `output_ids`.
6. Evaluator decodes only the newly generated suffix into `llm_outputs`.
7. Two branches:
   - Discrete action text such as `↑/←/→/↓/STOP`
   - Numeric pixel-goal text such as `153 285`
8. If numeric:
   - evaluator computes `pixel_goal = [x, y]`
   - evaluator calls `model.generate_latents(output_ids, pixel_values, image_grid_thw)`
   - evaluator prepares look-down RGB/depth tensors
   - evaluator calls `model.generate_traj(traj_latents, images_dp, depths_dp)`
   - evaluator converts continuous/trajectory output into local actions with `traj_to_actions(...)`
9. Evaluator executes the selected action(s) in Habitat and repeats.

The single most important original contract is that `generate_latents()` consumes the actual `output_ids` returned by `generate()`, together with the exact `pixel_values` and `image_grid_thw` produced by the same HF processor pass.

## 3. Evaluator-Level Dataflow

The online evaluation loop lives in [`habitat_vln_evaluator.py:289`](/root/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L289).

### 3.1 Episode and step state

At the start of each episode, the evaluator initializes:

- `rgb_list`: history images
- `action_seq`: queued discrete S2 actions
- `input_images`: images that will be fed into the prompt
- `output_ids`: last HF generation result
- `llm_outputs`: decoded S2 text
- `messages`: OpenAI-style multimodal conversation
- `local_actions`: queued S1 actions
- `pixel_goal`: current active point-goal

See [`habitat_vln_evaluator.py:325-336`](/root/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L325).

### 3.2 Observation preprocessing and look-down image acquisition

Each step begins by:

1. reading current `rgb` and `depth`
2. filtering and rescaling depth into millimeters
3. building the current PIL image
4. if the previous action was `LOOKDOWN`, treating the current frame as the follow-up look-down view
5. otherwise:
   - resizing the normal RGB view for System-2
   - temporarily stepping Habitat down twice to capture a look-down observation for System-1
   - preprocessing the look-down depth into the `224x224` local-planner format

See [`habitat_vln_evaluator.py:341-390`](/root/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L341).

This already creates two distinct visual streams:

1. System-2 prompt images:
   - resized history/current RGB
2. System-1 local rollout inputs:
   - look-down RGB
   - look-down depth

### 3.3 Prompt construction

When there is no queued action and no active pixel-goal, the evaluator asks System-2 for a fresh decision.

There are two prompt modes:

1. Normal step:
   - start from the base instruction template
   - inject the instruction text
   - optionally inject historical `<image>` placeholders
   - append the current image
2. Look-down follow-up step:
   - append the prior assistant response as context
   - append the look-down image as the new user turn

The normal prompt path is in [`habitat_vln_evaluator.py:401-418`](/root/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L401), and the look-down follow-up path is in [`habitat_vln_evaluator.py:393-400`](/root/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L393).

Then the evaluator picks a conjunction string such as `"you can see "` using `random.choice(self.conjunctions)` and appends `<image>` to it. See [`habitat_vln_evaluator.py:420-433`](/root/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L420).

This means prompt wording is not deterministic unless the Python RNG is fixed.

### 3.4 HF processor and S2 generation

The evaluator then performs the standard HF multimodal preparation:

1. `text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`
2. `inputs = self.processor(text=[text], images=input_images, return_tensors="pt")`
3. `output_ids = self.model.generate(**inputs, ...)`
4. decode only `output_ids[:, inputs.input_ids.shape[1]:]`

See [`habitat_vln_evaluator.py:435-451`](/root/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L435).

This is the original canonical S2 output contract:

1. `inputs.input_ids` is the prompt prefix
2. `output_ids` is the full prompt-plus-generated sequence
3. the newly generated suffix is the slice after `inputs.input_ids.shape[1]`

### 3.5 Branch A: discrete action string

If `llm_outputs` does not contain digits, the evaluator parses discrete symbols using `parse_actions(...)`. See [`habitat_vln_evaluator.py:507-509`](/root/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L507).

The mapping itself is:

- `STOP -> 0`
- `↑ -> 1`
- `← -> 2`
- `→ -> 3`
- optionally `↓ -> 5`

That map is defined in evaluator init and mirrored in the policy wrapper.

### 3.6 Branch B: pixel-goal text

If `llm_outputs` contains digits, the evaluator interprets it as a pixel-goal. See [`habitat_vln_evaluator.py:454-458`](/root/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L454).

Important detail:

1. The text is usually `"row col"` or `"y x"` style.
2. The evaluator converts it into `[int(coord[1]), int(coord[0])]`.
3. So the stored `pixel_goal` is `[x, y]`.

If pitch actions or local policy are unavailable, evaluator falls back to a discrete action guessed from the pixel x-coordinate. See [`habitat_vln_evaluator.py:460-464`](/root/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L460).

If local policy is enabled, the evaluator enters the true DualVLN branch.

## 4. Original `generate_latents()` Contract

The exact implementation is in [`internvla_n1.py:320-347`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py#L320).

The computation is:

1. Start from the full `output_ids` returned by HF generation.
2. Embed those ids with `self.get_model().embed_tokens(input_ids)`.
3. Append `N_QUERY` copies of `TRAJ_TOKEN_INDEX` to the token sequence.
4. Run the vision tower on `pixel_values` using `image_grid_thw`.
5. Replace every image token position in the text embedding tensor with the visual embeddings.
6. Append `latent_queries` to the embedding sequence.
7. Build M-RoPE positions by calling `self.get_rope_index(input_ids, image_grid_thw)`.
8. Run the Qwen2.5-VL backbone with `inputs_embeds` and `position_ids`.
9. Take `outputs.hidden_states[-1][:, -N_QUERY:, :]`.

This is the ground-truth latent contract the migration needs to match.

Stated more formally, the original HF latent path is:

`full_output_ids + visual replacement + latent_queries + HF get_rope_index -> backbone last_hidden_states[:, -N_QUERY:, :]`

It is not:

- tokenizer re-encode of text
- best-effort matching prompt ids
- pooler output
- some other hidden layer
- a projected embedding space

## 5. Original `forward()` Semantics

The underlying model semantics that make `generate_latents()` work live in [`internvla_n1.py:58-318`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py#L58).

Important points:

1. If `inputs_embeds` is not provided, the model first embeds `input_ids`. See [`internvla_n1.py:128-129`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py#L128).
2. Image token positions are replaced with vision tower outputs. See [`internvla_n1.py:130-146`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py#L130).
3. Video token positions are similarly replaced if video is present. See [`internvla_n1.py:148-164`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py#L148).
4. `TRAJ_TOKEN_INDEX` positions are replaced by repeated `latent_queries`. See [`internvla_n1.py:166-172`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py#L166).
5. If `position_ids` is absent, the model computes them with `self.get_rope_index(...)` during prefill. See [`internvla_n1.py:177-205`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py#L177).
6. The backbone is then run as `self.model(...)`, and logits are produced with `lm_head`. See [`internvla_n1.py:206-220`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py#L206).

This is why strict migration must preserve not only token ids but also:

- visual replacement positions
- latent-query insertion positions
- position-id construction
- the exact hidden-state slice that is consumed by System-1

## 6. Original System-1 Components

The System-1 modules are defined in [`internvla_n1_arch.py:121-172`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1_arch.py#L121).

They include:

1. `latent_queries`
2. `traj_dit`
3. `noise_scheduler`
4. `action_encoder`
5. `pos_encoding`
6. `action_decoder`
7. `cond_projector`
8. optionally `rgb_model`, `memory_encoder`, `rgb_resampler` for async variants
9. optionally `navdp`

This matters because `generate_latents()` only produces the conditioning tensor. The actual local rollout happens in a separate subsystem.

## 7. Original `generate_traj()` Contract

The runtime rollout lives in [`internvla_n1.py:349-420`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py#L349).

For the `nextdit` path, the dataflow is:

1. Project `traj_latents` through `cond_projector`.
2. If async:
   - normalize `images_dp`
   - extract RGB visual features
   - encode memory
   - resample memory tokens with `QFormer`
   - concatenate memory tokens with projected latents
3. Create unconditional zero hidden states and concatenate with conditional hidden states for classifier-free guidance.
4. Initialize `latents = randn_tensor(..., generator=None, ...)`. See [`internvla_n1.py:389-394`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py#L389).
5. Iterate scheduler timesteps.
6. Encode action latents, add positional encodings.
7. Run `traj_dit(...)`.
8. Decode back to action space with `action_decoder`.
9. Apply classifier-free guidance.
10. Step the diffusion scheduler.
11. Return sampled trajectory deltas.

For the `navdp` path, the method dispatches into the navdp predictor instead.

Two consequences matter for any diffing work:

1. System-1 is stochastic by default because `randn_tensor(..., generator=None)` is used.
2. A small latent change can be amplified by the local planner even if S2 text is almost unchanged.

## 8. From Trajectory to Executable Actions

The evaluator converts S1 outputs to discrete simulator actions using [`vln_utils.py:63-120`](/root/InternNav/internnav/model/utils/vln_utils.py#L63).

`traj_to_actions(...)`:

1. unnormalizes XY deltas
2. reconstructs a mean global trajectory from sampled deltas
3. greedily converts the trajectory into left/right/forward steps using:
   - `step_size=0.25`
   - `turn_angle_deg=15`
   - `lookahead=4`

This is the final bridge from latent space to Habitat actions.

## 9. Cleaner Mirror: `internvla_n1_policy.py`

The policy wrapper in [`internvla_n1_policy.py:110-198`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1_policy.py#L110) is useful because it expresses the same split more cleanly than the Habitat evaluator.

Its `s2_step(...)` does:

1. build prompt and history
2. run `processor.apply_chat_template(...)`
3. run `model.generate(...)`
4. decode text
5. if numeric:
   - build `pixel_goal`
   - call `model.generate_latents(output_ids, inputs.pixel_values, image_grid_thw)`
6. else:
   - parse action sequence

Its `s1_step_latent(...)` does:

1. call `model.generate_traj(...)`
2. call `traj_to_actions(...)`
3. truncate to the first four local actions

See [`internvla_n1_policy.py:200-215`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1_policy.py#L200).

## 10. Strict-Alignment Invariants in the Original Repo

If a migration wants to claim strict equivalence to the original DualVLN behavior, these invariants must hold:

1. `output_ids` must be the exact ids returned by the real S2 generation call.
2. `pixel_values` and `image_grid_thw` must come from the exact same processor pass as those `output_ids`.
3. The latent path must append `TRAJ_TOKEN_INDEX * N_QUERY`, not some other suffix.
4. The model must insert `latent_queries` at those suffix positions, not pool from arbitrary prompt positions.
5. Position ids must be built with the same multimodal M-RoPE semantics as the HF path.
6. The extracted tensor must be `last_hidden_states[:, -N_QUERY:, :]`.
7. If run-to-run comparison is desired, both:
   - `random.choice(self.conjunctions)`
   - `randn_tensor(..., generator=None)`
   must be controlled or replaced by deterministic seeds.

## 11. Short Human-Triage Checklist

If a human is trying to debug where S2->S1 alignment broke, the fastest inspection order in the original repo is:

1. evaluator prompt assembly:
   - [`habitat_vln_evaluator.py:392-452`](/root/InternNav/internnav/habitat_extensions/vln/habitat_vln_evaluator.py#L392)
2. HF latent contract:
   - [`internvla_n1.py:320-347`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py#L320)
3. `forward()` multimodal replacement and `TRAJ_TOKEN_INDEX` insertion:
   - [`internvla_n1.py:128-205`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py#L128)
4. S1 stochastic rollout:
   - [`internvla_n1.py:389-394`](/root/InternNav/internnav/model/basemodel/internvla_n1/internvla_n1.py#L389)
5. trajectory-to-action discretization:
   - [`vln_utils.py:63-120`](/root/InternNav/internnav/model/utils/vln_utils.py#L63)

