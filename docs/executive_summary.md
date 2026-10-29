# Executive summary – OneRef (IA025 Final Project)

## Goal
- **Topic:** multi-agent systems, where several pre-trained models work together as agents, applied to visual inspection of industrial defects.
- **What for:** detect defects in parts (scratches, cracks, missing components) using only one reference photo of a good part.
- **Why:** defect images are rare and expensive to label. The best open models get about 74 % right, while a human inspector reaches about 86 %.
- **How:** a Python orchestrator coordinates the agents, with no retraining of the models:
  - DINOv3 compares the part with the reference;
  - Qwen3-VL-8B and InternVL3-8B answer the question;
  - a verifier reviews the answers;
  - an aggregator produces the final answer.

  The target is to beat the best single model (73.91 %), confirmed with a McNemar statistical test.

## Dataset
MMAD (ICLR 2025): a public industrial-inspection benchmark with 8,366 images and 39,670 multiple-choice questions.
- GitHub: https://github.com/jam-cc/MMAD
- Hugging Face: https://huggingface.co/datasets/jiang-cc/MMAD

## Hardware
- **Own GPU:** RTX 3060 Laptop with 6 GB of VRAM, for development and testing.
- **Google Colab Pro (1 month):** cloud L4/T4 GPUs, for the 8B models and the final evaluation.
