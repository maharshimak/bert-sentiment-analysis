"""Binary metrics with explicit shape and label validation."""
def _pairs(y_true, y_pred):
    truth, predicted = list(y_true), list(y_pred)
    if len(truth) != len(predicted):
        raise ValueError("Truth and prediction lengths must match")
    if any(value not in (0, 1) for value in truth + predicted):
        raise ValueError("Binary labels must be zero or one")
    return list(zip(truth, predicted))

def compute_accuracy(y_true, y_pred):
    pairs = _pairs(y_true, y_pred)
    return sum(t == p for t, p in pairs) / len(pairs) if pairs else 0.0

def compute_f1_binary(y_true, y_pred):
    pairs = _pairs(y_true, y_pred)
    true_positive = sum(t == 1 and p == 1 for t, p in pairs)
    false_positive = sum(t == 0 and p == 1 for t, p in pairs)
    false_negative = sum(t == 1 and p == 0 for t, p in pairs)
    denominator = 2 * true_positive + false_positive + false_negative
    return 2 * true_positive / denominator if denominator else 0.0
