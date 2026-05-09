# STRIDE

Scientific equation discovery via LLM-guided program search: templates in [`specs/`](./specs/), pipeline in [`codes/`](./codes/). Datasets under [`data/`](./data/).

Developed from the **[LLM-SR](https://arxiv.org/abs/2404.18400)** codebase.

![Overview](./images/LLMSR.jpg)

## Installation

Python **3.11** recommended (≥ 3.9). From the repository root:

```bash
conda create -n stride python=3.11
conda activate stride
pip install -r requirements.txt
```

Or:

```bash
conda env create -f environment.yml
conda activate stride
```

`requirements.txt` uses **PyTorch CUDA 11.8** wheels; for CPU or other CUDA builds, see [PyTorch install](https://pytorch.org/get-started/locally/).

## Layout

| Path | Role |
|------|------|
| [`codes/main.py`](./codes/main.py) | CLI (`PYTHONPATH` = repo root) |
| [`codes/`](./codes/) | `sample/`, `evaluate/`, `refine/`, `update/` |
| [`specs/`](./specs/) | Problem specs (`specification_*.txt`) |
| [`specs/benchmark/`](./specs/benchmark/) | Extra specs for `data/benchmark_dr/` |
| [`data/`](./data/) | CSV / NPZ data |

## Run

From the **repository root**:

```bash
python codes/main.py --problem_name oscillator1 \
  --spec_path ./specs/specification_oscillator1.txt \
  --log_path ./logs/oscillator1_run1
```

**API** (`codes/sample/sampler.py`, `codes/refine/critic.py`): default model flag is **`gpt-5.1`** (see [`codes/main.py`](./codes/main.py), [`codes/config.py`](./codes/config.py)).

Bash / Git Bash:

```bash
export OPENAI_API_KEY="YOUR_KEY"
export OPENAI_API_BASE="https://api.openai.com/v1"   # or OPENAI_BASE_URL

python codes/main.py --use_api True --api_model "gpt-5.1" \
  --problem_name oscillator1 \
  --spec_path ./specs/specification_oscillator1.txt \
  --log_path ./logs/oscillator1_api
```

PowerShell:

```powershell
$env:OPENAI_API_KEY = "YOUR_KEY"
$env:OPENAI_API_BASE = "https://api.openai.com/v1"

python codes/main.py --use_api True --api_model "gpt-5.1" `
  --problem_name oscillator1 `
  --spec_path ./specs/specification_oscillator1.txt `
  --log_path ./logs/oscillator1_api
```

Use `/v1` as the API base suffix. **`problem_name`** must point to a folder under `data/` that contains `train.csv`. **`--use_api False`**: configure the local HTTP URL in [`codes/sample/sampler.py`](./codes/sample/sampler.py).

More examples: [`run_llmsr.sh`](./run_llmsr.sh).

## Specs

- `specs/specification_oscillator1.txt`, `oscillator2`, `bactgrow`, `stressstrain`
- `specs/benchmark/` — `CRK`, `BPG`, `PO`, `MatSci` families

## Configuration

[`codes/config.py`](./codes/config.py)

## Citation (STRIDE — your work)

If you use this repository in research, please cite **your** publication (fill in when available):

```bibtex
% TODO: Replace with your BibTeX (title, authors, venue, year, arXiv/DOI).
@article{stride_placeholder,
  title   = {TODO},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO},
  url     = {https://arxiv.org/abs/TODO}
}
```

## Acknowledgments

Upstream framework: [LLM-SR](https://arxiv.org/abs/2404.18400) (equation discovery with LLMs). Related benchmark: [LLM-SRBench](https://arxiv.org/abs/2504.10415), [data](https://huggingface.co/datasets/nnheui/llm-srbench).

## License

[LICENSE](./LICENSE) (MIT). See upstream projects linked from the original LLM-SR release (e.g. FunSearch, PySR) for their terms.
