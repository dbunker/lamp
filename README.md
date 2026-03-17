# LLM to ASP Modeling Protocol (LAMP)

## Abstract

Answer Set Programming (ASP) offers a compelling mechanism to represent product configuration. However, being able to deduce such programs that are sufficiently performant remains challenging for many users. In this area Large Language Models (LLMs) have the potential to bridge this gap. This work proposes a LLM to ASP Modeling Protocol (LAMP) system to deduce and optimize ASP rules based on provided instance facts and the expected stable models. This system also facilitates the evaluation of LLM capabilities in this regard of which gpt-oss, gpt-5-mini and qwen3-coder where selected for testing. To properly perform the evaluation a variety of ASP program complexities and potential optimizations were assessed including: varying the number of rules, number of positive and negative body literals, and number of predicates and terms. The evaluation of varying complexities also demonstrates which program features are most challenging for LLMs in ASP program generation.

## Full Paper

[LAMP Paper](/LAMP.pdf)

## Overview

LAMP evaluates whether LLMs can infer ASP rules given only instance facts and the expected stable model. The pipeline:

1. **Generate benchmarks** — synthetic ASP programs with configurable complexity parameters
2. **Run LLMs** — prompt each model with facts + expected stable model, parse its generated rules
3. **Run ILASP** — symbolic inductive learning baseline for comparison
4. **Aggregate results** — compare generated stable models against ground truth

### Models Evaluated

| Model | Interface |
|---|---|
| `gpt-oss:20b` | Ollama (local) |
| `gpt-5-mini` | OpenAI API |
| `qwen3-coder:30b` | Ollama (local) |
| ILASP | CLI subprocess |

## Benchmark Structure

Each benchmark is a `.lp` file containing ASP facts and rules. Benchmarks vary across:

| Parameter | Values |
|---|---|
| `num_predicates` | 4, 5 |
| `num_possible_terms` | 4, 5 |
| `num_facts` | 4, 5 |
| `num_rules` | 1, 2, 3 |
| `num_literals` | 1, 2, 3 |
| `num_neg_literals` | 0, 1, 2 |

Example benchmark (`data/orig_benchmarks/benchmark_0.lp`):
```
d2(o0).
d1(o0).
d2(o1).
d3(o2).

d1(X) :- d3(X).
```

Solutions are stored as JSON with the stable model atoms and clingo statistics (`data/orig_solutions/`).

## Data Layout

```
data/
  orig_benchmarks/     # generated ASP programs
  orig_solutions/      # clingo stable models for each benchmark
  ilasp_tasks/         # formatted ILASP learning tasks
  ilasp_responses/     # raw ILASP output
  ilasp_benchmarks/    # ILASP-inferred programs
  ilasp_solutions/     # clingo results for ILASP programs
  llm_responses/       # raw LLM text responses (per model)
  llm_benchmarks/      # LLM-inferred programs (per model)
  llm_solutions/       # clingo results for LLM programs (per model)
```

## Dependencies

- [clingo](https://potassco.org/clingo/) — ASP solver (Python API)
- [ILASP](https://www.ilasp.com/) — inductive ASP learner (CLI binary)
- [openai](https://pypi.org/project/openai/) — OpenAI API client
- [ollama](https://pypi.org/project/ollama/) — Ollama local model client
- [pandas](https://pandas.pydata.org/) — result aggregation

Install dependencies with [uv](https://github.com/astral-sh/uv):

```
uv sync
```

### API Key

Create `key.json` in the project root:

```json
{
  "openai": "sk-..."
}
```

## Running

```
uv run lamp.py
```

Edit `main()` in `lamp.py` to control which pipeline stages run:

```python
def main():
    generate_benchmarks()   # step 1: create benchmark data
    run_llms()              # step 2: query LLMs
    run_ilasp()             # step 3: run ILASP baseline
    aggregate_results()     # step 4: compute match statistics
```

Steps are cached — if output files already exist they are not recomputed (pass `rerun=True` to force).

## Analysis

[Jupyter Notebook](/stats.ipynb)
