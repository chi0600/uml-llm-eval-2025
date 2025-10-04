# UML-LLM-Eval-2025
Code, data, and evaluation scripts for benchmarking **LLM-driven UML sequence diagram generation**.

> 🔬 Goal: provide a reproducible pipeline that (1) prompts LLMs to generate UML sequence diagrams, (2) validates syntax, (3) scores outputs along semantic/structural/readability dimensions, and (4) aggregates results across models and domains.

[MIT License](./LICENSE) • Prompt template: [`Prompt_Template.pdf`](./Prompt_Template.pdf)

---

## 🌟 Highlights
- **Task & prompts**: standardized prompts for UML sequence diagram generation (see `Prompt_Template.pdf`).
- **Reproducible evaluation**: syntax validity + multi-metric quality scoring (semantic, structure, readability, compliance).
- **Extensible**: plug in new models, corpora, or metrics without changing the whole pipeline.

---


## 🚀 Quick start

### 1) Clone
```bash
git clone https://github.com/chi0600/uml-llm-eval-2025.git
cd uml-llm-eval-2025

