# LLM to ASP Modeling Protocol (LAMP)

# Abstract

Answer Set Programming (ASP) offers a compelling mechanism to represent product configuration. However, being able to deduce such programs that are sufficiently performant remains challenging for many users. In this area Large Language Models (LLMs) have the potential to bridge this gap. This work proposes a LLM to ASP Modeling Protocol (LAMP) system to deduce and optimize ASP rules based on provided instance facts and the expected stable models. This system also facilitates the evaluation of LLM capabilities in this regard of which gpt-oss, gpt-5-mini and qwen3-coder where selected for testing. To properly perform the evaluation a variety of ASP program complexities and potential optimizations were assessed including: varying the number of rules, number of positive and negative body literals, and number of predicates and terms. The evaluation of varying complexities also demonstrates which program features are most challenging for LLMs in ASP program generation.

# Full Paper

[LAMP Paper](/LAMP.pdf)

# Running

```
uv run lamp.py
```

# Jupyter Notebook

[Analysis](/stats.ipynb)
