import gradio as gr
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import os

# Optional: use Hugging Face token if model is gated
hf_token = os.getenv("HF_TOKEN")
model_id = "s2w-ai/DarkBERT"

# Load model with auth
tokenizer = AutoTokenizer.from_pretrained(model_id, use_auth_token=hf_token)
model = AutoModelForMaskedLM.from_pretrained(model_id, use_auth_token=hf_token)
unmasker = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Predict function
def predict(text):
    if "[MASK]" not in text and "<mask>" not in text:
        return "❗ Please include [MASK] or <mask> in your sentence."
    text = text.replace("[MASK]", "<mask>")
    try:
        result = unmasker(text)
        predictions = [r["sequence"] for r in result]
        return "\n".join(predictions)
    except Exception as e:
        return f"⚠️ Model error: {str(e)}"

# Gradio app
demo = gr.Interface(
    fn=predict,
    inputs="text",
    outputs="text",
    title="DarkBERT Fill-Mask Demo",
    description="Enter a sentence with [MASK] to get predictions using s2w-ai/DarkBERT"
)

demo.launch()
