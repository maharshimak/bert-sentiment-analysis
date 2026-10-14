from src.metrics import compute_accuracy, compute_f1_binary

def test_compute_accuracy():
    y_true = [0, 1, 1, 0]
    y_pred = [0, 1, 0, 0]
    acc = compute_accuracy(y_true, y_pred)
    assert abs(acc - 0.75) < 1e-6

def test_compute_f1_binary():
    y_true = [1, 1, 0, 1, 0]
    y_pred = [1, 0, 1, 1, 0]
    f1 = compute_f1_binary(y_true, y_pred)
    # around 0.666...
    assert 0.65 < f1 < 0.70

