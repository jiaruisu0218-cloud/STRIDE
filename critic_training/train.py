"""Launch LLaMA-Factory with portable paths and an explicit GPU selection."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

import yaml

ROOT = Path(__file__).resolve().parent


def configuration(args):
    config = yaml.safe_load((ROOT/'configs'/f'{args.stage}.yaml').read_text(encoding='utf-8'))
    data = args.data_dir.resolve()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new, empty output directory; checkpoints are not overwritten')
    info = json.loads((data/'dataset_info.json').read_text(encoding='utf-8'))
    names = ['critic_train', 'critic_dev'] if args.stage == 'sft' else ['critic_preferences']
    for name in names:
        entry = info[name]
        rows = json.loads((data/entry['file_name']).read_text(encoding='utf-8'))
        if not isinstance(rows, list) or not rows:
            raise ValueError(f'{name} must contain a nonempty JSON list')
        required = ('messages',) if args.stage == 'sft' else ('messages','chosen','rejected')
        if not all(isinstance(r, dict) and all(k in r for k in required) for r in rows):
            raise ValueError(f'Invalid {name} conversation fields')
        if args.stage == 'dpo' and not entry.get('ranking'):
            raise ValueError('DPO dataset_info must set ranking: true')
    config.update(model_name_or_path=args.model, dataset_dir=str(data), output_dir=str(output))
    if args.stage == 'dpo':
        if args.sft_adapter is None:
            raise ValueError('--sft-adapter is required for DPO')
        adapter = args.sft_adapter.resolve()
        if not (adapter/'adapter_config.json').is_file():
            raise ValueError('SFT adapter_config.json is missing')
        config.update(adapter_name_or_path=str(adapter), ref_model=args.model, ref_model_adapters=str(adapter))
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['sft','dpo'], required=True)
    parser.add_argument('--model', default='Qwen/Qwen3-8B')
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sft-adapter', type=Path)
    parser.add_argument('--gpus', default='0', help='Comma-separated visible GPU indices')
    parser.add_argument('--dry-run', action='store_true', help='Validate paths and print YAML without training')
    args = parser.parse_args()
    gpus = [x.strip() for x in args.gpus.split(',')]
    if not all(x.isdigit() for x in gpus) or len(set(gpus)) != len(gpus):
        parser.error('--gpus must contain distinct numeric GPU indices')
    config = configuration(args)
    if args.dry_run:
        print(yaml.safe_dump(config, sort_keys=False))
        print(f'# GPU count: {len(gpus)}; effective batch size: '
              f'{len(gpus)*config["per_device_train_batch_size"]*config["gradient_accumulation_steps"]}')
        return
    executable = shutil.which('llamafactory-cli')
    if executable is None:
        parser.error('Install LLaMA-Factory in the active Python environment first')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(gpus), NPROC_PER_NODE=str(len(gpus)),
               FORCE_TORCHRUN='1' if len(gpus)>1 else '0')
    with tempfile.TemporaryDirectory(prefix='stride-critic-config-') as directory:
        path = Path(directory)/'train.yaml'
        path.write_text(yaml.safe_dump(config,sort_keys=False),encoding='utf-8')
        subprocess.run([executable,'train',str(path)],env=env,check=True)


if __name__ == '__main__':
    main()
