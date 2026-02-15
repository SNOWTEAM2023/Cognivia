import pandas as pd
import json
import time
import random
from typing import List, Dict
from openai import OpenAI
import os

class CBTProfessionalResponder:
    def __init__(self, model: str = "gpt-5-mini", random_seed: int = None):
        self.model = model
        self.random_seed = random_seed
        self.client = OpenAI(
            # Replace with your OpenAI API token
            api_key=os.environ.get("your_openai_api"),
            base_url="https://api.openai.com/v1"
        )
        self.samples = None

    def load_samples(self, excel_path: str, sample_size: int = 10, random_seed: int = None) -> List[Dict]:
        df = pd.read_excel(excel_path)

        if len(df) < sample_size:
            raise ValueError(f"Insufficient data: requested {sample_size} samples, but data file only has {len(df)} rows")

        if random_seed is not None:
            random.seed(random_seed)
        elif self.random_seed is not None:
            random.seed(self.random_seed)

        indices = random.sample(range(len(df)), sample_size)
        samples = []

        for i in indices:
            row = df.iloc[i]
            sample = {
                "question": row['Thought/Statement'],
                "cognitive_distortions": row['Cognitive Distortion'],
                "rational_response": row['Rational Response']
            }
            samples.append(sample)

        self.samples = samples
        print(f"Randomly loaded {len(samples)} sample examples")
        print(f"Randomly selected sample indices: {sorted(indices)}")

        return samples

    def build_prompt(self, question: str, distortion: str) -> str:

        examples_text = ""
        if self.samples:
            examples_text = "Reference examples:\n"
            for i, sample in enumerate(self.samples, 1):
                examples_text += f"\nExample {i}:\n"
                examples_text += f"Question: {sample['question']}\n"
                examples_text += f"Cognitive Distortion: {sample['cognitive_distortions']}\n"
                examples_text += f"Professional Response: {sample['rational_response']}\n"

        prompt = f"""You are a cognitive behavioral therapy (CBT) psychologist. Based on the patient's type of cognitive distortion and specific situation, please provide a professional and compassionate response. Your primary goal is to establish a safe atmosphere of trust and understanding, ensuring your reply includes the following CBT components, appropriately segmented, and connecting each part in an organized and fluid manner.

{examples_text}

1. Validation and Empathy: Acknowledge and express understanding and sympathy for the user's emotional experience and the issues raised. Respond to their emotions with warm, empathetic language, like a close friend, to build trust and a sense of security.

2. Identifying Cognitive Distortions: Briefly explain, using both professional and everyday language, how this thinking pattern might be affecting the user, based on the types of cognitive distortions provided in the Excel sheet and the specific situation.

3. Proposing Gentle Cognitive Challenges: Use open-ended reflective questions to gently and non-confrontationally help the user reconsider this thinking pattern.

4. Providing CBT Strategies: Offer practical CBT techniques directly targeting the identified cognitive distortions, including both professional terminology and detailed, easy-to-understand explanations.

5. Encouragement and Closing Remarks: Encourage the user and remind them that changes in cognition and emotions are a gradual and ongoing process.

Input Format:
question:{question}
distortion:{distortion}

Output Format (JSON):
{{
    "question": "original question",
    "distortion": "cognitive distortion type", 
    "cbt_response": "your integrated CBT response paragraph"
}}
"""

        return prompt

    def generate_response(self, question: str, distortion: str) -> Dict:

        prompt = self.build_prompt(question, distortion)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a cognitive behavioral therapy (CBT) psychologist."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=800
            )

            response_content = response.choices[0].message.content

            try:
                parsed_response = json.loads(response_content)
                return parsed_response
            except json.JSONDecodeError:
                print(f"JSON parsing failed, returning raw content: {response_content[:100]}...")
                return {
                    "question": question,
                    "distortion": distortion,
                    "cbt_response": response_content
                }

        except Exception as e:
            print(f"Error generating response: {e}")
            return {
                "question": question,
                "distortion": distortion,
                "cbt_response": f"Generation failed: {str(e)}"
            }

    def process_data_file(self,
                          sample_file: str,
                          data_file: str,
                          output_file: str,
                          delay: float = 1.0,
                          sample_size: int = 10) -> pd.DataFrame:

        print("=" * 60)
        print("Starting processing workflow")
        print("=" * 60)

        print(f"Loading data from {data_file}...")
        data_df = pd.read_excel(data_file)
        print(f"Loaded {len(data_df)} records to process")

        results = []
        total = len(data_df)

        for idx, row in data_df.iterrows():
            try:
                question = row['Thought']
                distortion = row['Cognitive Distortion']

                print(f"Processing [{idx + 1}/{total}]: {question[:50]}...")

                print(f"Randomly loading {sample_size} sample examples from {sample_file}...")
                self.load_samples(sample_file, sample_size=sample_size)

                response = self.generate_response(question, distortion)
                cbt_response = response.get('cbt_response',
                                                 response.get('CBT_response',
                                                                   str(response)))

                result = {
                    "Thought": question,
                    "Cognitive Distortion": distortion,
                    "Rational Response": cbt_response,
                }
                results.append(result)

                time.sleep(delay)

            except Exception as e:
                print(f"Error processing record {idx + 1}: {e}")
                results.append({
                    "Thought": str(row.get('Thought', '')),
                    "Cognitive Distortion": str(row.get('Cognitive Distortion', '')),
                    "Rational Response": f"Processing failed: {str(e)}",
                })

        results_df = pd.DataFrame(results)
        results_df.to_excel(output_file, index=False, engine='openpyxl')

        print(f"\nProcessing complete! Results saved to: {output_file}")

        return results_df

def main():

    responder = CBTProfessionalResponder(model="gpt-5-mini", random_seed=42)

    current_dir = os.path.dirname(os.path.abspath(__file__))

    sample_file= os.path.join(current_dir, "..","data","CBT_Cognitive_Triplet_Dataset.xlsx")
    data_file = os.path.join(current_dir, "distortion.xlsx")
    output_file = os.path.join(current_dir, "response.xlsx")

    results = responder.process_data_file(
        sample_file=sample_file,
        data_file=data_file,
        output_file=output_file,
        delay=1.5,
        sample_size=10
    )

    return results

if __name__ == "__main__":
    print("Cognitive Behavioral Therapy (CBT) Professional Response Generator")
    print("=" * 60)

    results = main()

    if results is not None:
        print("\n✅ Processing complete!")
