def predict(text):
    # Make sure the sentence includes a mask
    if "[MASK]" not in text and "<mask>" not in text:
        return "❗ Please include [MASK] or <mask> in your sentence."

    # Replace [MASK] with <mask> for tokenizer compatibility
    text = text.replace("[MASK]", "<mask>")

    try:
        result = unmasker(text)
        predictions = [r["sequence"] for r in result]
        return "\n".join(predictions)
    except Exception as e:
        return f"⚠️ Model error: {str(e)}"
