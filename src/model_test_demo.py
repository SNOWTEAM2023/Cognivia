from openai import OpenAI
import os
api_key = os.getenv("YOUR_API_KEY") 

client = OpenAI(
    api_key=api_key,
    base_url="https://api.siliconflow.cn/v1"
)

FINE_TUNED_MODEL_ID = "ft:LoRA/Qwen/Qwen2.5-7B-Instruct:d50jhbk50mis73di8n5g:gpt5_mini:udjarjexxlodpjueztat-ckpt_step_625"

test_user_input = ("Is my career short-lived? Feeling lost about the future.")

response = client.chat.completions.create(
    model=FINE_TUNED_MODEL_ID,
    messages=[
        {
            "role": "system",
            "content": """You are a cognitive behavioral therapy (CBT) psychologist. First, identify the type of cognitive distortion exhibited in the statement, and then provide a response containing the following five paragraphs, separated by blank lines:1.Empathy and Validation
                          2.Cognitive Distortion Analysis
                          3.Reflective Questions 
                          4.CBT Exercise Recommendation 
                          5.Encouragement and Next Steps.
                          If it does not contain a cognitive distortion (e.g., casual conversation, general questions, or statements without distortions), switch to a natural, supportive conversation mode. Respond in a warm, counselor-like tone without analyzing distortions or following the five-paragraph structure.
                       """
        },
        {
            "role": "user",
            "content": test_user_input
        }
    ],
    temperature=0.2,
    max_tokens=300
)

print("\n--- Model Analysis Results ---\n")
print(response.choices[0].message.content)