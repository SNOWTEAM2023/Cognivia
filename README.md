<p align="center">
  <img src="https://github.com/SNOWTEAM2023/Cognivia/blob/main/materials/cognivia.png" width="400">
</p>

💻 This is the official implementation of paper **Cognivia: An AI Therapist for Evidence-Based Cognitive Behavioral Therapy**.

✅ This paper has been submitted to [**The 32nd SIGKDD Conference on Knowledge Discovery and Data Mining -AI for Sciences Track (KDD) 2026**](https://kdd2026.kdd.org/ai4sciences-track-call-for-papers/).

**Cognivia** is an evidence-based artificial intelligence therapist that integrates automatic cognitive distortion identification and rational response generation.

#### Authors
Qi Chen, [Siria Xiyueyao Luo](https://www.rug.nl/staff/x.luo/?lang=en), [Xuejiao Zhao*](https://zxjwudi.github.io/xuejiaozhao/)

**	Southwest Petroleum University &nbsp; | &nbsp; 	University of Groningen &nbsp; |&nbsp; Nanyang Technological University**

\* Corresponding author

[![Stargazers repo roster for @SNOWTEAM2023/GEM](https://reporoster.com/stars/SNOWTEAM2023/GEM)](https://github.com/SNOWTEAM2023/GEM/stargazers)


## :fire: News
* **[2026.02.01]** We release github repository of **Cognivia**. 💪 Have a try！



## 🧭 Framework Overview
<p align="center">
  <img src="https://github.com/SNOWTEAM2023/Cognivia/blob/main/materials/overview.jpg" width="1000">
</p>
**GEM** aligns base LLM using human preference data by a **Coginitive Feedback Loop**, which includes **Cognitive Filtering** and **SEGA** modules.

Key modules of GEM include:

- **Cognitive Filtering**: Generate `k` Chain-of-Thought (CoTs) candidates per query and **rank** them by Entropy-guided Token Scoring module. Entropy-guided Token Scoring module encourages **exploration mid‑CoT** (high entropy on top‑m steps) and **confidence at the end** (low final entropy).
- **SEGA**: A **listwise** objective that updates the policy using **group-mean–centered advantages**, which update with weights proportional to **Aᵢ = rᵢ − r̄** within each k-way group.

## Dataset
1. **CBT Cognitive Triplet Dataset**:
2. **Augmented CBT Cognitive Triplet Dataset**:


### ✨ Code Structure
The code structure and corresponding comments of this repository are as follows:

```
GEM/
├── GEM.py                      # Main entry script for running GEM
├── prompts
│   ├── filter_prompt.txt       # Prompt_1 of CBT Cognitive Triplet Dataset Augmentation (Cognitve Distortion Labelling)
│   └── response_prompt.txt     # Prompt_2 of CBT Cognitive Triplet Dataset Augmentation (Rational Pesponse Generation)

├── data/                       # Data
│   └── preference_data.jsonl
│
├── src/                        # Core implementation of GEM
│   ├── __init__.py
│   ├── config.py               # Configuration utilities
│   ├── dataset.py              # Dataset & dataloader definitions
│   ├── entropy_scorer.py       # Entropy-based scoring
│   ├── gem_trainer.py          # GEM training pipeline
│   └── sft_trainer.py          # Supervised fine-tuning (SFT) trainer
│
├── materials/                  # Figures & assets for the paper
├── README.md                   # Project introduction and usage
├── LICENCE.txt                 # Licence information
└── requirements.txt            # Python dependencies
```

## Usage
## Experiments
# Main results

## Evaluation

## 🔑 License
This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).
Commercial use is prohibited without a separate license agreement with the author.

