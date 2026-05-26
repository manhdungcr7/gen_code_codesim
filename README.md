# CodeSIM + LBPP

> Extending [CodeSIM](https://arxiv.org/abs/2502.05664) (NAACL 2025 Findings) to the **LBPP** benchmark — a dataset not evaluated in the original paper.

---

## About CodeSIM

**CodeSIM** is a multi-agent code generation framework that mirrors human problem-solving: *plan → simulate → code → debug via simulation*. Three agents collaborate:

- **Planning Agent** — recalls an analogous problem, derives a step-by-step plan, then simulates it on sample I/O before coding begins.
- **Coding Agent** — translates the verified plan into executable code.
- **Debugging Agent** — when tests fail, simulates the faulty code to locate the bug and produce a corrected solution.

---

## Results (GPT-4o-mini, pass@1 %)

| Approach | HumanEval | HumanEval-ET | MBPP | MBPP-ET | **LBPP** |
|----------|:---------:|:------------:|:----:|:-------:|:--------:|
| Direct | 84.1 | 75.0 | 76.1 | 53.1 | 52.5 |
| CoT | 85.4 | 78.0 | 79.3 | 54.9 | 55.6 |
| Self-Planning | 84.1 | 76.2 | 76.6 | 51.6 | 59.3 |
| Analogical | 82.3 | 74.4 | 78.8 | 53.1 | 54.9 |
| MapCoder | 89.6 | 79.3 | 84.6 | 56.9 | 66.0 |
| **CodeSIM** | **96.3** | **84.1** | **89.9** | **59.7** | **69.8** |

CodeSIM consistently outperforms all baselines. Its advantage decreases as problem complexity increases (HumanEval → LBPP).

---

## About LBPP

[LBPP](https://huggingface.co/datasets/CohereForAI/lbpp) (Less-Basic Python Programming, by CohereForAI) contains 162 Python problems that go beyond basic algorithmic tasks — focusing on standard-library usage, string/data manipulation, and practical coding skills.

The original CodeSIM repository does **not** support LBPP. This repo adds:

- `src/datasets/LBPPDataset.py` — dataset adapter (decodes base64/pickle-encoded test cases, runs functional evaluation)
- `download_lbpp.py` — fetches the dataset from HuggingFace and saves it to `data/LBPP/lbpp.jsonl`

---

## Setup

**Requirements:** Python 3.11+

```bash
git clone https://github.com/manhdungcr7/gen_code_codesim
cd gen_code_codesim

# create and activate virtual environment
py -3.12 -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Copy `.env.example` to `.env` and set your API key:

```bash
cp .env.example .env   # then edit .env
```

```
API_TYPE="openai"
OPENAI_API_KEY=sk-...
```

Download the LBPP dataset (one-time):

```bash
python download_lbpp.py
```

---

## Running

```bash
python src/main.py \
  --dataset       LBPP        \
  --strategy      CodeSIM     \
  --model_provider openai     \
  --model         gpt-4o-mini
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--dataset` | `HumanEval` | `HumanEval` \| `MBPP` \| `LBPP` \| `APPS` \| `CC` \| `xCodeEval` \| `BigCodeBench-Hard` |
| `--strategy` | `Direct` | `Direct` \| `CoT` \| `SelfPlanning` \| `Analogical` \| `MapCoder` \| `CodeSIM` |
| `--model_provider` | `OpenAI` | `openai` \| `groq` \| `gemini` \| `anthropic` |
| `--model` | `ChatGPT` | Model name, e.g. `gpt-4o-mini`, `llama-3.1-8b-instant` |
| `--temperature` | `0` | Sampling temperature |
| `--pass_at_k` | `1` | Samples per problem |
| `--cont` | `yes` | Resume previous run (`yes`) or start fresh (`no`) |

---

## Viewing Results

Results are written to `results/<DATASET>/<STRATEGY>/<MODEL>/Run-N/`:

| File | Contents |
|------|----------|
| `Results.jsonl` | Per-problem generated code, pass/fail, token usage |
| `Summary.txt` | Overall pass@k statistics |
| `Log.txt` | Full prompt/response log |

---

## Citation

```bibtex
@misc{islam2025codesim,
  title   = {CODESIM: Multi-Agent Code Generation and Problem Solving through Simulation-Driven Planning and Debugging},
  author  = {Md. Ashraful Islam and Mohammed Eunus Ali and Md Rizwan Parvez},
  year    = {2025},
  eprint  = {2502.05664},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
