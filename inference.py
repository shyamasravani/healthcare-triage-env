import sys
from transformers import pipeline

def run_inference():
    try:
        # Use a small, guaranteed model from Hugging Face Hub
        generator = pipeline("text-generation", model="distilgpt2")

        # Example triage prompt
        prompt = "Patient age 45, symptoms: chest pain, dizziness. Suggested triage action:"

        # Generate output
        result = generator(prompt, max_length=60, num_return_sequences=1)

        # Print the generated text
        print("[END] Response:", result[0]["generated_text"])

    except Exception as e:
        print("[ERROR] Inference failed:", str(e))
        sys.exit(1)

if __name__ == "__main__":
    print("[START] Running inference...")
    run_inference()
