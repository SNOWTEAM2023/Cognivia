import pandas as pd
from openai import OpenAI
import time

# Replace with your SiliconFlow API token
api_key = "your_siliconflow_api"

INPUT_FILE = "question.xlsx"
OUTPUT_FILE = "Cognivia_response.xlsx"
STATEMENT_COLUMN = "user1"
client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1"
)

FINE_TUNED_MODEL_ID = "ft:LoRA/Qwen/Qwen2.5-7B-Instruct:d50jhbk50mis73di8n5g:gpt5_mini:udjarjexxlodpjueztat-ckpt_step_625"


def analyze_batch():
    df = pd.read_excel(INPUT_FILE)

    if 'assistant1' not in df.columns:
        df['assistant1'] = ""

    for i in range(len(df)):
        if df.at[i, 'assistant1']:
            continue

        user_input = str(df.at[i, STATEMENT_COLUMN])
        print(f"Processing {i + 1}/{len(df)}")

        try:
            response = client.chat.completions.create(
                model=FINE_TUNED_MODEL_ID,
                messages=[
                    {
                        "role": "system",
                        "content": """You are a cognitive behavioral therapy (CBT) psychologist. First, identify the type of cognitive distortion exhibited in the statement, and then provide a response containing the following five paragraphs, separated by blank lines:
                        1.Empathy and Validation
                        2.Cognitive Distortion Analysis 
                        3.Reflective Questions 
                        4.CBT Exercise Recommendation 
                        5.Encouragement and Next Steps.
                        If it does not contain a cognitive distortion (e.g., casual conversation, general questions, or statements without distortions), switch to a natural, supportive conversation mode. Respond in a warm, counselor-like tone without analyzing distortions or following the five-paragraph structure.
"""
                    },
                    {
                        "role": "user",
                        "content": user_input
                    }
                ],
                temperature=0.2,
                max_tokens=500
            )

            df.at[i, 'assistant1'] = response.choices[0].message.content
            time.sleep(1)

        except Exception as e:
            df.at[i, 'assistant1'] = f"Error: {str(e)}"

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Completed! Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    analyze_batch()
