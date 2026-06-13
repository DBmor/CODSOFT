from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def train_model(data):

    X = data["Description"]
    y = data["Genre"]

    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=10000
    )

    X = tfidf.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = LogisticRegression(max_iter=1000)

    model.fit(X_train, y_train)

    return model, tfidf, X_test, y_test