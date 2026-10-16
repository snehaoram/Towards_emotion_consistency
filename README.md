# Towards Emotion Consistency Analysis of Large Language Models in Emotional Conversational Contexts

Official repository for the paper: **"Towards Emotion Consistency Analysis of Large Language Models in Emotional Conversational Contexts"** ([arXiv:2605.06476](https://arxiv.org/abs/2605.06476)).

## 📌 About the Project
This work introduces a framework to systematically examine the **emotional consistency** of Large Language Models (LLMs) in emotional conversational environments by testing models against false claim queries (false presuppositions) across three intensity levels (L1, L2, L3) spanning **Extreme** and **Moderate** emotions.

## ⚙️ Experimental Setup
* **Models Evaluated:** `Claude-3.5-haiku`, `GPT-4o-mini`, and `Mistral-7B`.
* **Hyperparameters:** Temperature set to `0.1`, max new tokens set to `20`.
* **Compute:** Google Colab for commercial APIs and a dedicated NVIDIA A100 cluster for `Mistral-7B`.

## 📁 Repository Structure
* `Data/`: Processed data, prompts, and inference frames.
* `Claude_3_5_data_Autoeval_Mistral_Autoeval.ipynb`: Data generation and baseline evaluation.
* `Consists_AutoEvals_GPTMistralClaude.ipynb`: Main consistency evaluation pipeline.
* `code_Attscores.py`: Attention score/weight analysis script.

## 📈 Key Findings
* Emotional consistency shows a persistent downward trend from L1 to L3 intensity levels.
* `Claude-3.5` demonstrates the highest resistance to extreme emotion false-presupposition challenges.

## 🚀 Getting Started
Clone the repository and install standard requirements, ensuring your `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` environment variables are set for commercial model evaluations.

## ✍️ Citation
```bibtex
@inproceedings{oram2026towards,
  title={Towards Emotion Consistency Analysis of Large Language Models in Emotional Conversational Contexts},
  author={Oram, Sneha and Bhushan, Ojaswita and Bhattacharyya, Pushpak},
  booktitle={Proceedings of the Workshop on Multimodal Emotion and Psycholinguistic Analysis (MultiPsyche) in Conjunction with AACL-IJCNLP 2026},
  year={2026}
}
```
