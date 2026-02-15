import pandas as pd
import requests
import json
import time
import random
import os
from typing import List, Dict


class CBTAnalyzer:
    def __init__(self, api_key, model: str = "deepseek-chat"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.deepseek.com/v1"

    def load_samples(self, excel_path: str) -> List[Dict]:
        df = pd.read_excel(excel_path)
        samples = []

        for _, row in df.iterrows():
            sample = {
                "question": row['Thought/Statement'],
                "cognitive_distortions": row['Cognitive Distortion'],
            }
            samples.append(sample)

        return samples

    def build_system_prompt(self, samples: List[Dict]) -> str:

        few_shot_examples = ""
        for i, sample in enumerate(random.sample(samples, min(10, len(samples))), 1):
            few_shot_examples += f"{i}. Question: {sample['question']}\n   Cognitive Distortions: {sample['cognitive_distortions']}\n\n"
        system_prompt = f"""You are a CBT psychologist.And analyze cognitive distortions and output in ENGLISH ONLY.

Cognitive Distortion Types:
1. All-or-nothing thinking
2. Overgeneralization  
3. Mental filter
4. Discounting the positives
5. Mind reading
6. Fortune telling
7. Magnification or minimization
8. Emotional reasoning
9. Should statements
10. Labeling
11. Personalization and blame

Examples:
{few_shot_examples}
Use exact JSON format below:

{{
    "question": "English translation here",
    "distortions": ["Distortion"]
}}
Please follow the sample dataset.
If no distortions found, use empty list: "distortions": []"""

        return system_prompt

    def analyze_batch(self, questions: List[str], samples_path: str,
                      batch_size: int = 20, delay: float = 1) -> List[Dict]:

        samples = self.load_samples(samples_path)

        results = []
        no_distortion_count = 0
        total_batches = (len(questions) + batch_size - 1) // batch_size

        for i in range(0, len(questions), batch_size):
            batch = questions[i:i + batch_size]
            batch_num = i // batch_size + 1

            system_prompt = self.build_system_prompt(samples)

            batch_distortion_count = 0

            print(f"Processing batch {batch_num}/{total_batches}, {len(batch)} questions in this batch")

            for question in batch:
                try:
                    result = self._analyze_single(question, system_prompt)
                    if result.get("distortions") and len(result["distortions"]) > 0:
                        results.append(result)
                        batch_distortion_count += len(result["distortions"])
                    else:
                        no_distortion_count += 1
                    time.sleep(delay)
                except Exception as e:
                    print(f"Analysis failed: {question[:50]}... Error: {e}")

            print(f"Batch {batch_num} identified {batch_distortion_count} cognitive distortions")

        if no_distortion_count > 0:
            print(f"Total {no_distortion_count} questions with no cognitive distortions")

        return results

    def _analyze_single(self, question: str, system_prompt: str) -> Dict:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        user_content = f"Analyze this question: {question}"

        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            "temperature": 0.1,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=data
        )

        if response.status_code == 200:
            result = json.loads(response.json()['choices'][0]['message']['content'])
            return {
                "question": result.get("question", ""),
                "distortions": result.get("distortions", [])
            }
        else:
            raise Exception(f"API call failed: {response.status_code} - {response.text}")


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Replace with your DeepSeek API token
    analyzer = CBTAnalyzer(api_key="your_deepseek_api")
    psyqa_path = os.path.join(current_dir, "questions.xlsx")
    psyqa_df = pd.read_excel(psyqa_path)
    questions = psyqa_df['Question'].tolist()

    print(f"Loaded {len(questions)} questions to analyze")
    # Replace with your DeepSeek API token
    samples_path = os.path.join(current_dir,"..","data","CBT_Cognitive_Triplet_Dataset.xlsx")
    results = analyzer.analyze_batch(
        questions=questions,
        samples_path=samples_path,
        batch_size=15,
        delay=1.5
    )

    print(f"Successfully analyzed {len(results)} questions with cognitive distortions")
    if len(results) == 0:
        print("No questions with cognitive distortions identified")
        return pd.DataFrame()

    results_df = pd.DataFrame(results)

    results_df['distortion'] = results_df['distortions'].apply(
        lambda x: x[0] if isinstance(x, list) and len(x) > 0 else 'No distortion'
    )

    final_df = pd.DataFrame({
        'Thought': results_df['question'],
        'Cognitive Distortion': results_df['distortion']
    })

    output_path = os.path.join(current_dir, "distortion.xlsx")

    final_df.to_excel(output_path, index=False, engine='openpyxl')
    print(f"Analysis complete! Results saved to: {output_path}")
    print(f"Exported {len(final_df)} questions with cognitive distortions")

    print(f"Analysis complete! Results saved to: {output_path}")
    print(f"Exported {len(final_df)} questions with cognitive distortions")

    return final_df


if __name__ == "__main__":
    final_results = main()
