import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv('fake_or_real_news.csv')  # Change to your dataset path
print("Sample data:\n", df.head())

# Separate labels and features
X = df['text']
y = df['label']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)

# Initialize TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)

# Initialize PassiveAggressiveClassifier
model = PassiveAggressiveClassifier(max_iter=50)
model.fit(tfidf_train, y_train)

# Predict on test set
y_pred = model.predict(tfidf_test)

# Accuracy and Confusion Matrix
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score * 100, 2)}%")

cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)
