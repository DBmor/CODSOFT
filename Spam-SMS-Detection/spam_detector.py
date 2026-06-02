import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Load dataset
archive = pd.read_csv("sms_data.csv", encoding="latin1")

# Keep useful columns only
archive = archive[['v1', 'v2']]

# Rename columns
archive.columns = ['sms_flag', 'message_text']

# Convert labels to numbers
archive['sms_flag'] = archive['sms_flag'].map({
    'ham': 0,
    'spam': 1
})

# Separate messages and labels
sms_messages = archive['message_text']
sms_labels = archive['sms_flag']

# Convert text into numerical features
text_encoder = TfidfVectorizer()

encoded_messages = text_encoder.fit_transform(sms_messages)

# Split data into training and testing
train_messages, test_messages, train_labels, test_labels = train_test_split(
    encoded_messages,
    sms_labels,
    test_size=0.2,
    random_state=42
)

# Train model
spam_identifier = MultinomialNB()

spam_identifier.fit(train_messages, train_labels)

# Test model
predicted_labels = spam_identifier.predict(test_messages)

accuracy = accuracy_score(test_labels, predicted_labels)

print("Accuracy:", round(accuracy * 100, 2), "%")

# Test custom messages
while True:

    user_sms = input("\nEnter SMS (or type exit): ")

    if user_sms.lower() == "exit":
        break

    sms_vector = text_encoder.transform([user_sms])

    result = spam_identifier.predict(sms_vector)

    if result[0] == 1:
        print("Prediction: SPAM")
    else:
        print("Prediction: HAM")