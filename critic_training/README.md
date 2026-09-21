# Critic Training

Train the STRIDE critic with Qwen3-8B LoRA: SFT on successful edits, followed by DPO on execution-based preference pairs. This release includes source code and configurations; logs, datasets, model weights, and rollout collection are not bundled.

## Setup

Run commands from the repository root in a Python 3.11 environment with CUDA-compatible PyTorch and LLaMA-Factory installed.

```bash
pip install -r critic_training/requirements-training.txt
bash train_critic.sh --help
```

Use a LLaMA-Factory version supporting `qwen3_nothink` and DPO reference adapters. Replace model/data paths and `YOUR_*` placeholders below with your own values. Builders require a locally available tokenizer.

Keep training, development, and final evaluation tasks disjoint. Task names are supplied at runtime; generated reports retain them for auditing.

## SFT

Input: `TASK/positive_trajectories.jsonl` containing verified successful edit trajectories. See `prepare_data.py` for the input fields and filtering rules. Exclude final evaluation tasks from the input logs.

```bash
python critic_training/prepare_data.py \
  --logs /data/critic_logs --model /models/Qwen3-8B \
  --dev-tasks YOUR_DEV_TASK_1 YOUR_DEV_TASK_2 \
  --output critic_training/data/sft

bash train_critic.sh --stage sft --model /models/Qwen3-8B \
  --data-dir critic_training/data/sft \
  --output-dir critic_training/outputs/sft --gpus 0 --dry-run
```

Optional expansion is available through `expand_data.py --help`; supply all evaluation input files to `--exclude-inputs`. Expansion preserves the existing development split.

## DPO

Input: a JSON list of records with `id`, `task`, `candidate_code`, `messages` (system/user), and `groups`. The user payload must contain the same candidate code. Each group contains a structured JSON `response` (`actions` and `short_comment`), two `discovery` outcomes, and eight `validation` outcomes. Successful outcomes contain `status: "ok"` and `score_delta` (revised minus original score).

Collect executor outcomes beforehand. Keep discovery and validation runs separate, and align validation seeds across groups. See `build_preferences.py` and `preference_rules.py` for validation and screening.

```bash
python critic_training/build_preferences.py \
  --records /data/group_outcomes.json --model /models/Qwen3-8B \
  --exclude-manifests critic_training/data/sft/sample_manifest.json /data/evaluation/manifest.json \
  --exclude-tasks YOUR_DEV_TASK_1 YOUR_DEV_TASK_2 YOUR_EVAL_TASK_1 \
  --output critic_training/data/dpo

bash train_critic.sh --stage dpo --model /models/Qwen3-8B \
  --sft-adapter critic_training/outputs/sft \
  --data-dir critic_training/data/dpo \
  --output-dir critic_training/outputs/dpo --gpus 0,1,2,3 --dry-run
```

Exclusion manifests are JSON lists containing `code_hash` from `prepare_data.canonical_code`; development entries also contain `split: "dev"` and `task`. Listed hashes and development tasks are excluded automatically. Use `--exclude-tasks` to specify all development and final evaluation tasks.

## Training Settings

Both stages use BF16, context length 4096, three epochs, and LoRA rank 16 / alpha 32 / dropout 0.05. SFT uses learning rate 5e-5 and selects the best checkpoint by development loss. DPO uses learning rate 5e-6 and beta 0.1, initializing both policy and frozen reference from the SFT adapter. Full settings: [SFT](configs/sft.yaml) and [DPO](configs/dpo.yaml).

Adjust `--gpus` for your devices. Remove `--dry-run` to train; output directories must be new or empty. DPO does not automatically select a checkpoint by evaluation performance.

## Tests and Notes

```bash
python -m unittest discover -s critic_training/tests -v
```

Lightweight tests require `requirements-data.txt` and do not run GPU training. `fit_worker.py` executes candidate code; use trusted inputs or a sandbox with external timeouts.

[source_manifest.json](source_manifest.json) records original source hashes and environment versions, not a complete environment lock. The portable builders do not reproduce historical datasets or exact sample counts; ID-based bootstrap seeds may change borderline preference selections.
