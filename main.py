from data_loader import load_dataset
from model_trainer import train_model
from predictor import predict_genre

data = load_dataset()

model, tfidf, X_test, y_test = train_model(data)

while True:

    movie_plot = input(
        "\nEnter movie description (or exit): "
    )

    if movie_plot.lower() == "exit":
        break

    genre = predict_genre(
        model,
        tfidf,
        movie_plot
    )

    print("Predicted Genre:", genre)