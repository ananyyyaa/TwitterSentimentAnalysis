import pandas as pd
import numpy as np
import re
import nltk
import string
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK resources
nltk.download('stopwords')

# Load data
data = pd.read_csv("twitter.csv")

# Map labels to categorical values
data["labels"] = data["class"].map({0: "Hate Speech", 1: "Offensive Language", 2: "No Hate and Offensive"})

# Define preprocessing function
stemmer = nltk.SnowballStemmer("english")
stopword = set(stopwords.words('english'))

def preprocess_text(text):
    text = re.sub(r'https?://\S+|www\.\S+', '', text)  # Remove URLs
    text = re.sub(r'<.*?>+', '', text)  # Remove HTML tags
    text = re.sub(r'\[.*?\]', '', text)  # Remove square brackets and their contents
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)  # Remove punctuation
    text = re.sub(r'\n', '', text)  # Remove newline characters
    text = re.sub(r'\w*\d\w*', '', text)  # Remove words containing numbers
    text = text.lower()  # Convert text to lowercase
    return text

data["tweet"] = data["tweet"].apply(preprocess_text)

# Prepare features and labels
X = data["tweet"]
y = data["labels"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Vectorize the text data
cv = CountVectorizer()
X_train_vectorized = cv.fit_transform(X_train)
X_test_vectorized = cv.transform(X_test)

# Train the Naive Bayes classifier
clf = MultinomialNB()
clf.fit(X_train_vectorized, y_train)

# Example prediction
sample = "Let's get all the bad bitches "
sample_vectorized = cv.transform([sample])
prediction = clf.predict(sample_vectorized)
print("Predicted Label:", prediction[0])

# Generate predictions for the test set
y_pred = clf.predict(X_test_vectorized)

# Create and display the classification report
from sklearn.metrics import classification_report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Plot a pie chart for the predicted labels distribution
plt.figure(figsize=(8, 6))
labels, counts = np.unique(y_pred, return_counts=True)
plt.pie(counts, labels=labels, autopct='%1.1f%%', colors=['skyblue', 'salmon', 'lightgreen'])
plt.title('Predicted Labels Distribution')
plt.axis('equal')
plt.show()

# Create and display the confusion matrix
from sklearn.metrics import confusion_matrix
print("Confusion Matrix:")
conf_mat = confusion_matrix(y_test, y_pred)
print(conf_mat)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='g')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()
