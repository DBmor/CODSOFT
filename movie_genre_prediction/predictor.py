def predict_genre(model, tfidf, text):

    transformed_text = tfidf.transform([text])

    result = model.predict(transformed_text)

    return result[0]