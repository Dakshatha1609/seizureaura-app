# backend/test_groq.py
from langchain_groq import ChatGroq
import os

# --- Paste your actual API key here ---
my_groq_api_key = os.getenv("GROQ_API_KEY")

try:
    print("Attempting to connect to Groq...")
    llm = ChatGroq(groq_api_key=my_groq_api_key, model_name='llama3-8b-8192')
    response = llm.invoke("Hello, Groq!")

    print("\n--- SUCCESS! ---")
    print("API Key is valid and connection was successful.")
    print("Response from Groq:", response.content)

except Exception as e:
    print("\n--- FAILED! ---")
    print("There is an issue with your API key or the connection.")
    print("Error details:", e)