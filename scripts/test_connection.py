import os
import sys
from openai import OpenAI

# Configuration matching USAGE.md and local default
API_BASE = os.getenv('OPENAI_API_BASE', 'http://127.0.0.1:5000/v1')
API_KEY = os.getenv('OPENAI_API_KEY', 'local')

def test_local_connection():
    print(f"Testing connection to: {API_BASE}")
    
    # Configure OpenAI
    try:
        client = OpenAI(
            base_url=API_BASE,
            api_key=API_KEY,
        )

        # Simple completion test
        print("Sending request...")
        resp = client.chat.completions.create(
            model='local-model', # Oobabooga often ignores this or uses currently loaded
            messages=[{'role':'user','content':'Are you online? Answer with YES.'}],
            max_tokens=10
        )
        
        content = resp.choices[0].message.content
        print(f"\n[SUCCESS] Response received:\n{content}")
        print("\nYour local LLM and API are working correctly.")
        
    except Exception as e:
        print(f"\n[ERROR] Failed to connect: {e}")
        print("\nTroubleshooting:")
        print("1. Is Oobabooga running?")
        print("2. Did you launch with the '--api' flag?")
        print(f"3. Is the URL '{API_BASE}' correct?")

if __name__ == "__main__":
    test_local_connection()
