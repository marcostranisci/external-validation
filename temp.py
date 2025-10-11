from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the instruction-tuned model and tokenizer
model_name = "Qwen/Qwen3-4B-Instruct-2507"  # Example of a small, instruction-tuned model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.to("mps")  # Use "cuda" for GPU, "mps" for Apple Silicon, or "cpu" for CPU
# Function to generate text
def generate_text(prompt, max_length=100, temperature=0.7, top_k=50):
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k,
        output_scores=True,
        pad_token_id=tokenizer.eos_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    prompt = "Explain the concept of causal inference in simple terms."
    generated_text = generate_text(prompt)
    print(generated_text)