import joblib


def predict(data):
    lr = joblib.load("svr_model.sav")
    return lr.predict(data)
