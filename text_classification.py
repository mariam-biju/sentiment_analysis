import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

class TextClassifier:
    def __init__(self, data_file):
        # Load the dataset
        self.data = pd.read_csv(data_file)
        self.vectorizer = TfidfVectorizer()  # For converting text to numerical features
        self.model = SVC(kernel='linear')  # Support Vector Machine with linear kernel

    def preprocess_data(self):
        """
        Preprocess the dataset by splitting it into training and testing sets.
        """
        X = self.data['text']  # Text data
        y = self.data['sentiment']  # Labels

        # Split the data into training and testing sets (80% train, 20% test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

    def train_model(self):
        """
        Train the text classification model using SVM.
        """
        # Convert text data to numerical features using TF-IDF
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        X_test_tfidf = self.vectorizer.transform(self.X_test)

        # Train the SVM classifier
        self.model.fit(X_train_tfidf, self.y_train)

        # Evaluate the model
        y_pred = self.model.predict(X_test_tfidf)
        print("Accuracy:", accuracy_score(self.y_test, y_pred))
        print("\nClassification Report:\n", classification_report(self.y_test, y_pred))

    def classify_text(self, text):
        """
        Classify new text input using the trained SVM model.
        """
        # Convert the input text to numerical features
        text_tfidf = self.vectorizer.transform([text])
        # Predict the label
        prediction = self.model.predict(text_tfidf)
        return prediction[0]

def main():
    print("Welcome to the SVM-Based Text Classifier!")
   
    # Path to the dataset file
    data_file = "sentiment_analysis.csv"  # Replace with your file path
   
    # Initialize and preprocess the classifier
    classifier = TextClassifier(data_file)
    classifier.preprocess_data()
   
    # Train the model
    print("Training the model...")
    classifier.train_model()
   
    # Test the classifier with user input
    while True:
        user_input = input("\nEnter a sentence (type 'exit' to quit): ")
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        prediction = classifier.classify_text(user_input)
        print(f"Predicted Sentiment: {prediction}")

if __name__ == "__main__":
    main()
