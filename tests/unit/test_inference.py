import pytest
from src.inference import predict

def test_inference_requires_explicit_opt_in():
    with pytest.raises(RuntimeError):
        predict("synthetic input")
