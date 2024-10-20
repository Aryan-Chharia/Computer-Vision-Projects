from transformers import pipeline

# Using the latest LLaMA2 model for text generation
generator = pipeline("text-generation", model="meta-llama/LLaMA2-7b")

# Text Generation example
text = generator("Hugging Face Transformers make NLP easy", max_length=50)
