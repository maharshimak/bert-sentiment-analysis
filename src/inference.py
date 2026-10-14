"""Explicit local inference only; importing this module does not load a model."""
from .model import MODEL_NAME, get_model, get_tokenizer

def predict(text, *, allow_inference=False, allow_download=False, checkpoint=MODEL_NAME, device="cpu"):
    if not allow_inference:
        raise RuntimeError("Inference requires explicit allow_inference=True")
    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
        raise ValueError("Use nonempty text of at most 10000 characters")
    if device not in ("cpu", "cuda"):
        raise ValueError("Select cpu or cuda explicitly")
    import torch
    tokenizer = get_tokenizer(checkpoint, allow_download=allow_download)
    model = get_model(checkpoint, allow_download=allow_download)
    model.to(device)
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class, logits

def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text")
    parser.add_argument("--checkpoint", default=MODEL_NAME)
    parser.add_argument("--allow-inference", action="store_true")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()
    if not args.allow_inference:
        parser.error("--allow-inference is required; no model was loaded")
    label, _ = predict(args.text, allow_inference=True, allow_download=args.allow_download, checkpoint=args.checkpoint)
    print("Predicted class index:", label)
    print("Base checkpoint classification heads are not fine-tuned sentiment models.")

if __name__ == "__main__":
    main()
