import utils.preprocessing as preprocessing
import model
import utils.evaluation as evaluation

if __name__ == "__main__":
    preprocessing.merge_lc_data_from_db()
    preprocessing.merge_point_machine_data_from_db()
    preprocessing.merge_rbc_data_from_db()

#    data = preprocessing.get_data()
#    X_train, X_test, y_train, y_test = preprocessing.split_data(data)
#    regressor = model.train_model(X_train, y_train)
#    y_pred = model.predict(regressor, X_test)
#    r2 = evaluation.evaluate_model(y_test, y_pred)
#    print("R^2:", r2)
