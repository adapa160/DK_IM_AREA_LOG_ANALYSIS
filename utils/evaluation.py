from sklearn.metrics import r2_score

def evaluate_model(y_test, y_pred):
    """Evaluate the performance of the model"""
    r2 = r2_score(y_test, y_pred)
    print("R^2:", r2)
