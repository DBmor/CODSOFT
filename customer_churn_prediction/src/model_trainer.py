from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

class ModelTrainer:

    def train(self, X, y):

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestClassifier()

        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        print("Accuracy:", accuracy)

        pickle.dump(
            model,
            open("models/churn_model.pkl", "wb")
        )