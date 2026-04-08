# LAMP: ASP Rule Induction and Optimization with LLMs

## Abstract

Answer Set Programming (ASP) offers a compelling mechanism for knowledge representation and reasoning, however developing such programs remains challenging. This work introduces the LLM to ASP Modeling Protocol (LAMP), a system by which an LLM iteratively generates and refines candidate ASP rules with each candidate verified by the Clingo solver. LAMP is evaluated against the leading Inductive Logic Programming (ILP) system Inductive Learning of Answer Set Programs (ILASP), across 120 synthetic benchmarks of varying complexity. ILASP achieves 97.5% accuracy, with failures due to search space timeouts. Among LLMs, *gpt-5-mini* reaches 100% accuracy, *gpt-oss:20b* 95.8%, and *qwen3-coder:30b* 45.0\%. Although ILASP was on average faster, *gpt-5-mini* solved all benchmarks on which ILASP timed out. Both LAMP and ILASP are also able to reduce literal counts and increase stratification at a similar level. Finally, feature analysis is used to determine predictors of LLM success or failure.

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
| `num_predicates` | 6, 10 |
| `num_possible_terms` | 10 |
| `num_facts` | 6, 9 |
| `num_rules` | 3, 6, 11 |
| `num_literals` | 2, 3 |
| `num_neg_literals` | 0, 1, 2 |
| `min_derived_atoms` | 1, 3 |

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
  ilasp_search_spaces/ # enumerated ILASP hypothesis spaces (where |H| ≤ 30 000)
  ilasp_benchmarks/    # ILASP-inferred programs
  ilasp_solutions/     # clingo results for ILASP programs
  llm_responses/       # raw LLM text responses (per model, per iteration)
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

Run individual pipeline stages via the CLI:

```sh
uv run lamp.py generate              # step 1: create benchmark data
uv run lamp.py ilasp                 # step 2: run ILASP baseline
uv run lamp.py llms                  # step 3: query LLMs
uv run lamp.py llms --max-iter 3     # step 3: query LLMs with up to 3 feedback iterations
uv run lamp.py llms --rerun          # step 3: force re-query even if cached
uv run lamp.py aggregate             # step 4: compute match statistics
```

Steps are cached — if output files already exist they are not recomputed (pass `--rerun` to force).

### LLM Feedback Loop

LLM inference uses an iterative correction loop (default 3 iterations, configurable with `--max-iter`). On the first iteration the model receives only facts and the target stable model. On subsequent iterations it additionally receives the actual stable model produced by its previous answer, allowing it to self-correct.

## Analysis

[Jupyter Notebook](/stats.ipynb)
