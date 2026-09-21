# Critic training: success-edit SFT and offline DPO

This directory exports reusable components from the STRIDE critic experiments. The critic receives task information, an equation skeleton, fitted parameters, NMSE, score components, and sensitivity diagnostics, and returns an English JSON edit plan. A fixed executor turns the plan into candidate equations; numerical fitting determines the training signal.

The release includes data builders, corrected execution rewards, preference screening, fitting feedback extraction, sampling utilities, and Qwen3-8B LoRA configurations. It does not include logs, training datasets, model weights, or the server-specific rollout scheduler. `build_preferences.py` consumes existing executor outcomes; it does not collect them automatically.

## Training Usage

We fine-tuned Qwen3-8B as the critic using LoRA with LLaMA-Factory, first through supervised fine-tuning (SFT) on successful edit trajectories and then through direct preference optimization (DPO) on execution-based preference pairs. The DPO run used a single node with 4 NVIDIA A800 GPUs, each with 80 GB of memory.

Both stages used BF16 precision, a maximum sequence length of 4,096 tokens, and 3 training epochs. LoRA was applied to all supported linear modules with rank 16, alpha 32, and dropout 0.05. The learning rates were 5e-5 for SFT and 5e-6 for DPO; DPO used a preference beta of 0.1. The SFT adapter initialized both the DPO policy and its frozen reference model.

The example commands below use one GPU for SFT and four GPUs for DPO. Set `--gpus` to the available device indices and run with `--dry-run` to inspect the resolved configuration before training. Full settings are provided in [configs/sft.yaml](configs/sft.yaml) and [configs/dpo.yaml](configs/dpo.yaml).

## Environment

Run commands from the repository root. Use a separate Python 3.11 environment, preferably Linux with NVIDIA GPUs. Install CUDA-compatible PyTorch and LLaMA-Factory separately, then install `critic_training/requirements-training.txt`. Lightweight tests require only `critic_training/requirements-data.txt`.

The source experiment used LLaMA-Factory `0.9.5.dev0`, PyTorch `2.10.0`, and Transformers `4.57.1`. The exact LLaMA-Factory commit was unavailable, so this is not a complete environment lock. See `source_manifest.json` for recorded versions and original source hashes. Confirm support for `qwen3_nothink` and DPO reference adapters in your LLaMA-Factory version. Builders need Transformers and a locally cached tokenizer; replace `/models/Qwen3-8B` below with your model directory.

## SFT data and training

The repository-root `train_critic.sh` forwards training arguments to `critic_training/train.py` for either SFT or DPO. Run `bash train_critic.sh --help` for options and split reminders. It consumes prepared datasets; it does not build data or select tasks. Set `PYTHON_BIN` to choose a Python executable. Relative paths are resolved from the caller's working directory.

Input logs contain `TASK/positive_trajectories.jsonl`. Each row has `base_id`, `label`, `historical_label`, `score_gain`, `base`, `refinements`, and `critic_actions` (or `critic_output`). Both labels must be `positive`. Candidates contain `function` and verified fitting feedback in `backfill`; `prepare_data.make_input` defines the required diagnostic fields. `fit_worker.py` produces those fields, but assembling trajectories remains the caller's responsibility.

```bash
python critic_training/prepare_data.py --logs /data/critic_logs --model /models/Qwen3-8B --output critic_training/data/sft_initial --dev-tasks YOUR_DEV_TASK_1 YOUR_DEV_TASK_2
python critic_training/expand_data.py --logs /data/critic_logs --existing-data critic_training/data/sft_initial --model /models/Qwen3-8B --exclude-inputs /data/evaluation/comparison_inputs.json --output critic_training/data/sft
bash train_critic.sh --stage sft --model /models/Qwen3-8B --data-dir critic_training/data/sft --output-dir critic_training/outputs/sft --gpus 0 --dry-run
```

Remove `--dry-run` to train. Expansion is optional: use `sft_initial` directly if it is omitted. Supply all evaluation input files to `--exclude-inputs`. Expansion preserves development conversations and admits repeated-success additions; single-success or mixed outcomes remain in reserve. Inspect generated reports before training.

Preparation filters invalid edits, unverified fits, score gains below 0.01, excessive context length, and duplicate canonical equation code. It emits conversations, a sample manifest, rejection records, a data report, and `dataset_info.json`.

All numerical fitting and scoring use the complete training observations for each problem. Conversation development splits are by task, not a 20% numerical holdout. Specify development tasks explicitly with the required `--dev-tasks` argument; no task names are hard-coded. Replace the `YOUR_*` placeholders in the examples with your own task names. Keep final evaluation tasks out of the input logs. Run the builders with `--help` for split requirements. Generated local reports and manifests retain the supplied task names for auditing.

SFT defaults: LoRA rank 16, alpha 32, dropout 0.05, all target modules, learning rate 5e-5, three epochs, context 4096, BF16, and gradient checkpointing. It evaluates each epoch and selects the best checkpoint by development loss. See `configs/sft.yaml`.

## DPO outcome format

Generate several complete critic responses per independent input (four in the collection design). Execute each with the same fixed executor and fitting budget. Compare whole edit plans, not individual actions within a plan.

The builder expects a JSON list. Each record has `id`, `task`, `candidate_code`, `messages` (system/user only), and `groups`. The JSON user-message payload must contain the same `candidate_code`. Each group contains a `response` string with the JSON `actions` and `short_comment`, two `discovery` outcomes, and eight `validation` outcomes. An English response example is:

```json
{"actions":[{"type":"SIMPLIFY","target":"x + x","rationale":"Combine repeated terms."}],"short_comment":"Refit the simplified equation."}
```

Each successful outcome has the form `{"status":"ok","score_delta":0.4}`, where delta is revised score minus original score. Failures use a non-`ok` status. These are schema illustrations, not measured examples. The caller must align validation outcomes across groups by execution seed, keep discovery and validation runs separate, and record the mapping from executor samples to outcomes. Counts are checked; seed provenance cannot be verified from this minimal schema.

Successful finite score deltas are clipped to [-5, 5]; failures have utility -5. The corrected group reward is:

```text
reward = 0.5 * max(0, max(utility)) + 0.5 * mean(utility) - 0.1 * failure_fraction
```

All-failure groups receive -2.6. Screening requires a valid improving chosen group, repeated paired advantages, a positive lower bound of a paired bootstrap confidence interval, and higher group reward. A rejected group with exactly one failure is excluded to avoid preferences driven by an isolated failure. See `preference_rules.py` for exact thresholds.

```bash
python critic_training/build_preferences.py --records /data/group_outcomes.json --model /models/Qwen3-8B --exclude-manifests critic_training/data/sft/sample_manifest.json /data/evaluation/manifest.json --exclude-tasks YOUR_DEV_TASK_1 YOUR_DEV_TASK_2 YOUR_EVAL_TASK_1 --output critic_training/data/dpo
bash train_critic.sh --stage dpo --model /models/Qwen3-8B --sft-adapter critic_training/outputs/sft --data-dir critic_training/data/dpo --output-dir critic_training/outputs/dpo --gpus 0,1,2,3 --dry-run
```

Exclusion manifests are JSON lists containing `code_hash` from `prepare_data.canonical_code`; development entries also contain `split: "dev"` and `task`. Every listed hash is excluded. Supply training/evaluation manifests appropriate to the intended separation. Outputs include one pair per accepted canonical equation, an audit, a manifest, and dataset registration. A requested number of pairs is not guaranteed.

The portable builder uses ID-based bootstrap seeds, whereas the historical scheduler used another seed schedule; borderline selections may differ. This source-only export does not reproduce the original training datasets or their exact sample counts.

Remove `--dry-run` to train. Policy initialization and frozen reference load the same SFT adapter over the same base model. DPO continues that adapter using sigmoid loss, beta 0.1, learning rate 5e-6, and three epochs. Batch size 1 per GPU and accumulation 2 give effective batch 8 on four GPUs. With 500 pairs this corresponds to 189 optimizer steps under usual epoch rounding; other data/GPU counts change this. DPO does not automatically select a checkpoint by evaluation performance. Compare checkpoints on independent inputs using a fixed executor and matched evaluation randomness.

## Other components and provenance

- `fit_worker.py JOB.json RESULT.json` refits one candidate using a compatible archived specification. The job has `key`, `function`, `spec_snapshot`, and `data` (CSV header, target last). It uses full data, fitting seed zero, and the specification's budget. Feedback is reconstructed, not historical. It executes supplied code: use trusted inputs or an external sandbox. The caller must enforce a process timeout (120 seconds in source collection) and CPU thread limits.
- `sampling.py` provides per-request random generators for critic sampling. Reproducibility also depends on model, inference software, hardware, and decoding settings. It is not a rollout scheduler.
- `source_manifest.json` records original downloaded hashes, not checksums of files after portability edits. `train.py` and `build_preferences.py` are portable entry points added for this export; original server orchestration is not included.

The wrapper validates dataset registration and refuses to overwrite nonempty output directories. To run the lightweight checks:

```bash
python -m unittest discover -s critic_training/tests -v
```

These checks cover trajectory reconstruction, rewards, preference screening, exclusions, and training configuration. They do not run GPU training or validate the complete historical experiment.
