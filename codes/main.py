import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

import os
from argparse import ArgumentParser
import numpy as np
import torch
import pandas as pd

from codes import pipeline
from codes import config as config_lib
from codes.sample import sampler
from codes.evaluate import evaluator
from codes.sample import dataset_analyzer

def disable_proxy_for_python():
    for k in [
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
        "http_proxy", "https_proxy", "all_proxy"
    ]:
        os.environ.pop(k, None)

disable_proxy_for_python()


def _parse_bool(value) -> bool:
    """Parse bool from CLI strings (argparse type=bool is truthy on any non-empty str)."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y", "t"):
            return True
        if v in ("false", "0", "no", "n", "f"):
            return False
    return bool(value)


parser = ArgumentParser()
parser.add_argument('--port', type=int, default=None)
parser.add_argument('--use_api', type=_parse_bool, default=False)
parser.add_argument('--api_model', type=str, default="gpt-5.1")
parser.add_argument('--spec_path', type=str)
parser.add_argument('--log_path', type=str, default="./logs/oscillator1")
parser.add_argument('--problem_name', type=str, default="oscillator1")
parser.add_argument('--run_id', type=int, default=1)
parser.add_argument('--data_hint_enabled', type=_parse_bool, default=False)
parser.add_argument('--data_hint_every', type=int, default=25)
parser.add_argument('--ablation', type=str, default='none',
                    choices=['none', 'data_hint', 'mixed_optimization', 'critic', 'tf_idf'],
                    help='Ablation: turn off one component (data_hint, mixed_optimization, critic, tf_idf).')
parser.add_argument('--no_early_stop_train_nmse', action='store_true',
                    help='Keep sampling until global_max_sample_num even if train NMSE is extremely small.')
args = parser.parse_args()




if __name__ == '__main__':
    class_config = config_lib.ClassConfig(llm_class=sampler.ApiLLM, sandbox_class=evaluator.LocalSandbox)
    global_max_sample_num = 2000

    with open(
        os.path.join(args.spec_path),
        encoding="utf-8",
    ) as f:
        specification = f.read()
    
    problem_name = args.problem_name

    df = pd.read_csv('./data/'+problem_name+'/train.csv')


    data = np.array(df)
    X = data[:, :-1]
    y = data[:, -1].reshape(-1)
    data_hint_text = None
    if args.data_hint_enabled:
        try:
            var_names = [str(c) for c in df.columns[:-1]]
            hint = dataset_analyzer.analyze_io_dataset(
                X,
                y,
                var_names=var_names,
                top_k=dataset_analyzer.TOP_K_TERMS,
            )
            data_hint_text = hint.prompt_hint
        except Exception as e:
            print(f"Data hint analysis failed: {e}")
    if 'torch' in args.spec_path:
        X = torch.Tensor(X)
        y = torch.Tensor(y)
    data_dict = {'inputs': X, 'outputs': y}
    dataset = {'data': data_dict}

    ablation = (args.ablation or "none").strip().lower()
    data_hint_enabled = args.data_hint_enabled and (ablation != "data_hint")
    critic_prob = 0.0 if ablation == "critic" else 0.4
    use_tfidf = ablation != "tf_idf"
    experience_buffer = config_lib.ExperienceBufferConfig(use_tfidf_for_clustering=use_tfidf)

    config = config_lib.Config(
        use_api=args.use_api,
        api_model=args.api_model,
        data_hint_text=data_hint_text,
        data_hint_enabled=args.data_hint_enabled,
        data_hint_every=args.data_hint_every,
        experience_buffer=experience_buffer,
        critic_prob=critic_prob,
        refine_enabled=True,
        early_stop_train_nmse_threshold=(None if args.no_early_stop_train_nmse else 1e-13),
    )

    pipeline.main(
        specification=specification,
        inputs=dataset,
        config=config,
        max_sample_nums=global_max_sample_num,
        class_config=class_config,
        log_dir=args.log_path,
    )

