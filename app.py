import gradio as gr
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch
import os

hf_token = os.environ.get("HF_TOKEN")
model_name = "s2w-ai/DarkBERT"

# Authenticate and load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForMaskedLM.from_pretrained(model_name, use_auth_token=hf_token)

# Create fill-mask pipeline with tokenizer + model
unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Inference function
def predict(text):
    if "[MASK]" not in text:
        return "Please include [MASK] in your input text."
    result = unmasker(text)
    predictions = [r["sequence"] for r in result]
    return "\n".join(predictions)

# Launch UI
gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="DarkBERT Fill-Mask Demo",
    description="Enter a sentence with [MASK] to get predictions using s2w-ai/DarkBERT"
).launch()
