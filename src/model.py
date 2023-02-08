from sklearn.linear_model import LinearRegression

def train_model(X_train, y_train):
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    return regressor

def predict(regressor, X_test):
    return regressor.predict(X_test)
