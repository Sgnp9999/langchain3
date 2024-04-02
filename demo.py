from transformers import AutoModel, AutoTokenizer
from langchain.llms import LLM

# Replace with your desired model ID from Hugging Face
model_id = "ProsusAI/finbert"

# Specify the directory to download the model to
download_directory = "./downloaded_model"

# Download the model and tokenizer
model = AutoModel.from_pretrained(model_id, cache_dir=download_directory)
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=download_directory)

# Create an LLM instance from the downloaded model
llm = LLM(model, tokenizer)
while True:
    # Use the LLM to generate text
    prompt = input("ask: ")
    response = llm.generate_text(prompt)

    print(response)