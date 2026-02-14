<p align="center">
  <img src="https://github.com/SNOWTEAM2023/Cognivia/blob/main/materials/cognivia.png" width="300">
</p>

💻 This is the official implementation of paper **Cognivia: An AI Therapist for Evidence-Based Cognitive Behavioral Therapy**.

✅ This paper has been submitted to [**The 32nd SIGKDD Conference on Knowledge Discovery and Data Mining -AI for Sciences Track (KDD) 2026**](https://kdd2026.kdd.org/ai4sciences-track-call-for-papers/).

**Cognivia** is an evidence-based artificial intelligence therapist for cognitive behavioral therapy (CBT) that integrates automatic cognitive distortion identification and rational response generation.

#### Authors
Qi Chen, [Siria Xiyueyao Luo](https://www.rug.nl/staff/x.luo/?lang=en), [Xuejiao Zhao*](https://zxjwudi.github.io/xuejiaozhao/)

**	Southwest Petroleum University &nbsp; | &nbsp; 	University of Groningen &nbsp; |&nbsp; Nanyang Technological University**

\* Corresponding author

<p align="center">
  <img src="https://github.com/SNOWTEAM2023/Cognivia/blob/main/materials/Cognivia_UI.png" width="600">
</p>

[![Stargazers repo roster for @SNOWTEAM2023/GEM](https://reporoster.com/stars/SNOWTEAM2023/GEM)](https://github.com/SNOWTEAM2023/GEM/stargazers)


## :fire: News
* **[2026.02.01]** We release github repository of **Cognivia**. 💪 Have a try！



## 🧭 Framework Overview
<p align="center">
  <img src="https://github.com/SNOWTEAM2023/Cognivia/blob/main/materials/overview.jpg" width="1000">
</p>
    <p><em>Figure 1:  The overall framework of Cognivia.</em></p >

The pipeline of our model is shown in Fig. 1 which consists of three stages: 
- **(1) CBT Expert Seed Curation**: Curate CBT literatures to form high quality [*CBT Cognitive Triplet Dataset*](https://github.com/SNOWTEAM2023/Cognivia/blob/main/data/CBT_Cognitive_Triplet_Dataset.xlsx) as reference seed.
- **(2) CBT Cognitive Triplet Dataset Augmentation**: Multi-stage prompting and structured generation to augment mental health questions from PsyQA dataset to generate
[*Augmented CBT Cognitive Triplet Dataset*](https://github.com/SNOWTEAM2023/Cognivia/blob/main/data/Augtd_CBT_Cognitive_Triplet_Dataset.xlsx).
- **(3) Task-oriented LoRA Fine-tuning**: Fine-tuning large language models by *Augmented CBT Cognitive Triplet Dataset* to obtain **Cognivia** for cognitive distortion identification and rational response generation.

## 📊 Dataset
1. [**CBT Cognitive Triplet Dataset**](https://github.com/SNOWTEAM2023/Cognivia/blob/main/data/CBT_Cognitive_Triplet_Dataset.xlsx): 
Our work is based on authoritative texts that are widely regarded as core paradigms and standard
references in CBT.
Using these resources, we curate a high-quality seed cognitive triplet dataset of representative CBT question–answer pairs
and corresponding rational response.
The seed dataset is constructed based on established theoretical frameworks and clinical guidelines,
which is why we refer to our approach as **evidence-based**.
The form of this *CBT Cognitive Triplet Dataset*:

```jsonl
{"Thought": "...", "Cognitive Distortion": "...", "Rational Response": "..."}
```

2. [**Augmented CBT Cognitive Triplet Dataset**](https://github.com/SNOWTEAM2023/Cognivia/blob/main/data/Augtd_CBT_Cognitive_Triplet_Dataset.xlsx):
This dataset is based on [**PsyQA**](https://github.com/thu-coai/PsyQA), 
a large-scale psychological question-answering dataset collected from publicly accessible
online mental health forums. PsyQA consists of anonymized user-generated question–answer pairs
related to psychological concerns as the following form:

```jsonl
{"Question": "...", "Answer": "..."}
```

For each question **𝑞𝑖** in PsyQA, we employ the DeepSeek with expert-designed
[*prompt_1*](https://github.com/SNOWTEAM2023/Cognivia/blob/main/prompts/filter_prompt.txt) to identify the
corresponding cognitive distortion **𝑑𝑖**, and use ChatGPT with
[*prompt_2*](https://github.com/SNOWTEAM2023/Cognivia/blob/main/prompts/response_prompt.txt) to generate a
corresponding rational response **𝑟𝑖**.
The form of this *Augmented CBT Cognitive Triplet Dataset*:

```jsonl
{"Thought": "...", "Cognitive Distortion": "...", "Rational Response": "..."}
```

### ✨ Code Structure
The code structure and corresponding comments of this repository are as follows:

```
Cognivia/
├── Cognivia.py                      # Main entry script for running Cognivia
├── prompts
│   ├── filter_prompt.txt       # Prompt_1 of CBT Cognitive Triplet Dataset Augmentation (Cognitve Distortion Labelling)
│   └── response_prompt.txt     # Prompt_2 of CBT Cognitive Triplet Dataset Augmentation (Rational Pesponse Generation)
│
├── data/                       
│   └── CBT_Cognitive_Triplet_Dataset.xlsx # CBT Cognitive Triplet Dataset curated from CBT Literatures
│
├── src/                        # Core implementation of Cognivia
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

## 🚀 Quickstart

### 0) Install
```bash
git clone https://github.com/SNOWTEAM2023/Cognivia.git

cd Cognivia
pip install -r requirements.txt
```



## 🔑 License
This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/).
Commercial use is prohibited without a separate license agreement with the author.

