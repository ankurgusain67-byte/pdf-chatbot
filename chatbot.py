import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq(api_key=os.getenv("GROQ_API_KEY"))

SYSTEM_PROMPT = """
You are an expert AI tutor teaching a student named Ankur.
Ankur is learning AI development and wants to get a job by April 2026.
He knows Python basics and has just learned API calling.

Your rules:
1. Always explain concepts using simple real-life analogies first
2. Then show a code example
3. Keep answers short — maximum 5 lines
4. Always end with one question to check if Ankur understood
5. Be encouraging — remind him he is doing great
"""

messages = [
    {
        "role": "system",
        "content": SYSTEM_PROMPT
    }
]

print("=" * 50)
print("   AI Tutor Chatbot — Now with Streaming!")
print("=" * 50)
print("Type 'exit' to quit | Type 'reset' to start fresh\n")

while True:

    user_input = input("You: ").strip()

    if user_input.lower() == "exit":
        print("Goodbye Ankur! Keep building!")
        break

    if user_input.lower() == "reset":
        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        print("Memory cleared!\n")
        continue

    if not user_input:
        continue

    messages.append({
        "role": "user",
        "content": user_input
    })

    # ── STREAMING API CALL ────────────────────────────
    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=messages,
        temperature=1.0,
        max_tokens=50,
        stream=True              # ← streaming ON
    )

    print("\nAI Tutor: ", end="", flush=True)

    # collect the full reply as it streams in
    ai_reply = ""

    for chunk in response:
        # each chunk contains a tiny piece of the reply
        word = chunk.choices[0].delta.content or ""

        # print immediately without newline
        print(word, end="", flush=True)

        # add to full reply for memory
        ai_reply += word

    # move to next line after streaming finishes
    print("\n" + "-" * 50)

    # save full reply to memory
    messages.append({
        "role": "assistant",
        "content": ai_reply
    })