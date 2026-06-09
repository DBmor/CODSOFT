import pickle

class Predictor:

    def __init__(self):
        self.model = pickle.load(
            open("models/churn_model.pkl", "rb")
        )

    def predict(self, data):
        return self.model.predict([data])