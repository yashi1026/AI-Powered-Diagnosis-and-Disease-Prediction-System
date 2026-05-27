from groq import Groq
import os
from dotenv import load_dotenv

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

chat_history = []

def get_chat_response(user_message):
    global chat_history

    try:
        chat_history.append({"role": "user", "content": user_message})

        completion = client.chat.completions.create(
            model="openai/gpt-oss-120b",  
            messages=[
                {
                    "role": "system",
                    "content": """
                You are a medical assistant for a disease prediction app.
                
                STRICT RULES:
                - Do NOT give long paragraphs
                - Keep answers SHORT and CLEAR
                - Always respond in this format:
                
                Possible Condition:
                - (1 or 2 diseases based on symptoms)
                
                Recommended Actions:
                - (2–4 simple actionable steps)
                - Suggest using relevant prediction tool (diabetes, heart, parkinsons)
                
                - Do NOT explain in detail
                - Do NOT write paragraphs
                - Use bullet points only
                """
                },
                *chat_history[-6:]
            ],
            temperature=0.7,
            max_completion_tokens=1024
        )

        reply = completion.choices[0].message.content

        chat_history.append({"role": "assistant", "content": reply})

        return reply

    except Exception as e:
        print("ERROR:", e)
        return "⚠️ AI is currently unavailable. Please try again."