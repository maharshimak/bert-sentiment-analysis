from src.metrics import compute_accuracy, compute_f1_binary

def test_compute_accuracy():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    assert abs(compute_accuracy(y_true, y_pred) - 0.75) < 1e-9

def test_compute_f1_binary_range():
    y_true = [1, 1, 0, 1, 0]
    y_pred = [1, 0, 1, 1, 0]
    f1 = compute_f1_binary(y_true, y_pred)
    assert 0.0 <= f1 <= 1.0
