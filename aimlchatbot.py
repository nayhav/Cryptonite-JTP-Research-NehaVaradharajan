from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Training data
X_train = ["hello", "hi", "how are you", "good morning", "bye", "see you"]
y_train = ["greeting", "greeting", "greeting", "greeting", "farewell", "farewell"]

# Train model
vectorizer = CountVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
model = MultinomialNB()
model.fit(X_train_vectors, y_train)

# Chatbot function
def chatbot_response(user_input):
    input_vector = vectorizer.transform([user_input])
    category = model.predict(input_vector)[0]
    responses = {
        "greeting": "Hello! How can I assist you?",
        "farewell": "Goodbye! Have a great day!",
    }
    return responses.get(category, "I'm not sure how to respond to that.")

# Chat loop
while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        break
    print("Bot:", chatbot_response(user_input))
