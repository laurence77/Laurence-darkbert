import gradio as gr
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import os

# Load model from gated repo with your Hugging Face token
model_name = "s2w-ai/DarkBERT"
hf_token = os.environ.get("HF_TOKEN")  # Get the secret token from Hugging Face Space secrets

# Authenticate and load model/tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=hf_token)
model = AutoModelForMaskedLM.from_pretrained(model_name, use_auth_token=hf_token)

# Prediction function
def predict(text):
    if "[MASK]" not in text:
        return "Please include [MASK] in your input text."
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    mask_index = (inputs.input_ids == tokenizer.mask_token_id).nonzero(as_tuple=True)[1].item()
    probs = torch.softmax(logits[0, mask_index], dim=0)
    topk = torch.topk(probs, 5)
    predictions = [tokenizer.decode([idx]) for idx in topk.indices]
    return ", ".join(predictions)

# Gradio interface
gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="DarkBERT Fill-Mask Demo",
    description="Enter a sentence with [MASK] to get predictions using s2w-ai/DarkBERT"
).launch()
