"""Lazy, offline-by-default model factories. No model is loaded during import."""
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2

def get_tokenizer(model_name=MODEL_NAME, *, allow_download=False):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(model_name, local_files_only=not allow_download, trust_remote_code=False)

def get_model(model_name=MODEL_NAME, num_labels=NUM_LABELS, *, allow_download=False):
    if type(num_labels) is not int or num_labels < 2:
        raise ValueError("At least two labels are required")
    from transformers import AutoModelForSequenceClassification
    return AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels, local_files_only=not allow_download, trust_remote_code=False, use_safetensors=True)
