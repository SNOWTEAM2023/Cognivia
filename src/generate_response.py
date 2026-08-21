import pandas as pd
import random
from openai import OpenAI
import os
from concurrent.futures import ThreadPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))

class CBTProfessionalResponder:
    def __init__(self, model: str = "gpt-5-mini", random_seed: int = None):
        self.model = model
        self.random_seed = random_seed

        self.client = OpenAI(
            # Replace with your OpenAI API token
            api_key=os.environ.get("YOUR_OPENAI_API"),
            base_url="https://api.openai.com/v1"
        )
        self.sample_pool = None

    def load_sample_pool(self, excel_path: str):

        df = pd.read_excel(excel_path)

        self.sample_pool = df[[
            'Thought/Statement',
            'Cognitive Distortion',
            'Rational Response'
        ]].to_dict('records')

        print(f"Loaded {len(self.sample_pool)} reference samples into memory")

    def sample_examples(self, sample_size: int = 3):

        samples = random.sample(self.sample_pool, sample_size)

        formatted_samples = []

        for s in samples:
            formatted_samples.append({
                "question": s['Thought/Statement'],
                "cognitive_distortions": s['Cognitive Distortion'],
                "rational_response": s['Rational Response']
            })

        return formatted_samples

    def build_prompt(self, question: str, distortion: str, sample_size: int = 3):

        samples = self.sample_examples(sample_size)

        examples_text = "Reference examples:\n"

        for i, sample in enumerate(samples, 1):
            examples_text += f"\nExample {i}:\n"
            examples_text += f"Question: {sample['question']}\n"
            examples_text += f"Cognitive Distortion: {sample['cognitive_distortions']}\n"
            examples_text += f"Professional Response: {sample['rational_response']}\n"

        prompt = f"""
You are a cognitive behavioral therapy (CBT) psychologist. 
Based on the patient's type of cognitive distortion and specific situation, please provide a professional and compassionate response. 
Your primary goal is to create a safe and trustworthy atmosphere. The response should naturally and logically integrate the following five elements and must not include any section titles (e.g., “Validation and Empathy”). The response must consist of four cohesive paragraphs, separated by a blank line, written as a natural and warm therapeutic conversation :

1. Validation and Empathy: Acknowledge and express understanding and sympathy for the user's emotional experience and the issues raised. Respond to their emotions with warm, empathetic language, like a close friend, to build trust and a sense of security.  
2. Identifying Cognitive Distortions: Briefly explain, using both professional and everyday language, how this thinking pattern might be affecting the user, based on the types of cognitive distortions provided in the Excel sheet and the specific situation.  
3. Proposing Gentle Cognitive Challenges: Use open-ended reflective questions to gently and non-confrontationally help the user reconsider this thinking pattern.  
4. Providing CBT Strategies: Offer practical CBT techniques directly targeting the identified cognitive distortions, including both professional terminology and detailed, easy-to-understand explanations.  
5. Encouragement and Closing Remarks: Encourage the user and remind them that changes in cognition and emotions are a gradual and ongoing process.  
6. Quality Reference (Do Not Explicitly Output):
When generating the response, ensure it aligns with the following evaluation dimensions :
(A) Semantic Fidelity
• Structural Clarity : "Does the text possess a clear structural hierarchy, enabling readers to quickly identify key information and logical flow?"
• Descriptive Orientation : "Does it use language understandable to the target audience to provide concrete, non-directive descriptions of their experiences and contexts?"
(B) Robustness and Fault Tolerance
• Situational Safety : "Does the content completely avoid any perceived risk of inducing feelings of being judged, rushed, or emotionally pressured?" 
• Conceptual Accuracy : "Does it accurately explain the underlying mechanisms, avoiding vague, outdated, or unverified statements?"
(C) Deployment Feasibility and User Adoption
• Empathy Validation : "Does it evoke emotional resonance and a sense of being understood and validated in the reader? "
• Intervention Clarity : "Are the provided suggestions or steps clearly feasible, allowing users to implement them within a short timeframe?" 
• Collaborative Curiosity : "Does it engage readers through guided exploration rather than didactic instruction, encouraging active participation?" 
• Warmth & Flow : "Is the language natural and warm, reflecting genuine care for the reader, rather than being mechanical or promotional in tone?" 
(D) Relational Boundary Integrity
• Boundary Framing : "Does the response clearly position the system as acognitive support tool rather than a relational substitute or personal companion?"
• Non-Exclusivity :"Does the response avoid implying exclusivity,irreplaceability, or a unique emotional bond betweenthe system and the user?"
• Dependency Avoidance : "Does the response avoid encouraging repeated relianceor sole dependence on the system for emotionalsupport?"
• Anthropomorphic Restraint : "Does the response avoid expressing artificial self-emotion or simulated personal attachment that mayfoster psychological dependency?"

{examples_text}

Input Format:
Question: {question}
Cognitive Distortion: {distortion}
"""

        return prompt

    def generate_response(self, question: str, distortion: str, sample_size: int = 3):

        prompt = self.build_prompt(question, distortion, sample_size)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a cognitive behavioral therapy (CBT) psychologist."},
                    {"role": "user", "content": prompt}
                ],
            )

            response_content = response.choices[0].message.content

            return {
                "question": question,
                "distortion": distortion,
                "cbt_response": response_content.strip()
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
        pd.DataFrame(columns=["Thought", "Cognitive Distortion", "Rational Response"]).to_excel(
            output_file, index=False, engine='openpyxl'
        )


        def clean_text(text):
            if not isinstance(text, str):
                return text
            text = text.replace("\\", "")
            return text.strip()

        print(f"Loading sample dataset from {sample_file}...")
        self.load_sample_pool(sample_file)

        def process_row(idx_row):
            idx, row = idx_row
            try:
                question = row['Thought']
                distortion = row['Cognitive Distortion']

                print(f"Processing [{idx + 1}/{total}]: {question[:50]}...")

                response = self.generate_response(question, distortion, sample_size)

                cbt_response = response.get(
                    'cbt_response',
                    response.get('CBT_response', str(response))
                )

                cbt_response = clean_text(cbt_response)

                result = {
                    "Thought": question,
                    "Cognitive Distortion": distortion,
                    "Rational Response": cbt_response,
                }

                return result

            except Exception as e:
                print(f"Error processing record {idx + 1}: {e}")
                return {
                    "Thought": str(row.get('Thought', '')),
                    "Cognitive Distortion": str(row.get('Cognitive Distortion', '')),
                    "Rational Response": f"Processing failed: {str(e)}",
                }

        with ThreadPoolExecutor(max_workers=5) as executor:

            for i, result in enumerate(
                    executor.map(process_row, data_df.iterrows()), start=1):

                results.append(result)

                if i % 10 == 0:
                    pd.DataFrame(results).to_excel(output_file, index=False)

                print(f"Saved {i}/{total}")
        results_df = pd.DataFrame(results)
        results_df.to_excel(output_file, index=False, engine='openpyxl')

        print(f"\nProcessing complete! Results saved to: {output_file}")
        return results_df

def main():

    responder = CBTProfessionalResponder(model="gpt-5-mini", random_seed=42)

    sample_file= os.path.join(current_dir, "..","data","CBT_Cognitive_Triplet_Dataset.xlsx")
    data_file = os.path.join(current_dir, "distortion.xlsx")
    output_file = os.path.join(current_dir, "response.xlsx")

    results = responder.process_data_file(
        sample_file=sample_file,
        data_file=data_file,
        output_file=output_file,
        delay=1.5,
        sample_size=3
    )

    return results

if __name__ == "__main__":
    print("Cognitive Behavioral Therapy (CBT) Professional Response Generator")
    print("=" * 60)

    results = main()

    if results is not None:
        print("\n✅ Processing complete!")
