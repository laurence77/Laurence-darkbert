import gradio as gr
from transformers import pipeline
import os

hf_token = os.environ.get("HF_TOKEN")

# Optional debug print
print("HF_TOKEN loaded:", bool(hf_token))

try:
    unmasker = pipeline("fill-mask", model="s2w-ai/DarkBERT", use_auth_token=hf_token)
except Exception as e:
    raise RuntimeError(f"Model failed to load: {str(e)}")

def predict(text):
    if "[MASK]" not in text:
        return "Please include [MASK] in your text."
    return unmasker(text)

gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="DarkBERT Token Test"
).launch()
