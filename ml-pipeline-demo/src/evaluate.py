def evaluate(model, test_rows, predict_fn):
    """Compute accuracy and other basic metrics."""
    correct = 0
    true_pos = 0
    false_pos = 0
    false_neg = 0

    for row in test_rows:
        pred = predict_fn(model, row)
        actual = int(row["churned"])
        if pred == actual:
            correct += 1
        if pred == 1 and actual == 1:
            true_pos += 1
        if pred == 1 and actual == 0:
            false_pos += 1
        if pred == 0 and actual == 1:
            false_neg += 1

    accuracy = correct / len(test_rows)
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) > 0 else 0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) > 0 else 0

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "test_size": len(test_rows)
    }
