"""Optional cached model tests; no network downloads are enabled."""
import os
import pytest
if os.environ.get("BERT_TEST_CACHED_MODELS") != "1":
    pytest.skip("Cached-model execution requires BERT_TEST_CACHED_MODELS=1", allow_module_level=True)
import torch
from src.model import get_model, get_tokenizer, MODEL_NAME, NUM_LABELS

def test_cached_model_forward_shape():
    tokenizer = get_tokenizer()
    model = get_model()
    with torch.no_grad():
        result = model(**tokenizer(["synthetic good", "synthetic bad"], return_tensors="pt", padding=True))
    assert result.logits.shape == (2, NUM_LABELS)

