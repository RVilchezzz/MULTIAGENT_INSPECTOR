# OneRef: multi-agent visual inspection from a single reference (IA025, Unicamp 2026)

Final project for **IA025 – Deep Learning**. A multi-agent system that orchestrates open pre-trained models (≤ 8B) to inspect industrial defects from **a single reference photo**, validated on the **MMAD** benchmark.

- **Agents:**
  - DINOv3 visual expert;
  - Qwen3-VL-8B and InternVL3-8B readers;
  - verifier;
  - aggregator.
- **Target:** beat the best single model (Qwen3-VL-8B, 73.91 %) and a same-cost majority vote, with a McNemar test.
- **Data:** [MMAD](https://github.com/jam-cc/MMAD) · images on [Hugging Face](https://huggingface.co/datasets/jiang-cc/MMAD)
- **Hardware:** RTX 3060 Laptop (6 GB) + Google Colab Pro

| Document | Content |
|---|---|
| [`docs/executive_summary.md`](docs/executive_summary.md) | One-page summary |
| [`docs/full_plan.md`](docs/full_plan.md) | Full plan: state of the art, workflow, dataset, validation, risks |
| [`docs/references.md`](docs/references.md) | References |
| [`docs/slides/`](docs/slides/) | Slides |

**Deliveries:** Oct 29 (plan) · Nov 12 (preliminary) · Nov 26 (final)

**Team:** Peter Roque Vilchez · Charles Maxime Doudy
