import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle

# Load the dataset
data = pd.read_csv("/content/drive/MyDrive/news/news.csv")

# Prepare the data
X = data['text']
Y = data['label']
x_train, x_test, y_train, y_test = train_test_split(X['text'], Y, test_size=0.2, random_state=42)

# Initialize a TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Fit and transform train set, transform test set
tfidf_train = tfidf_vectorizer.fit_transform(x_train)
tfidf_test = tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
pac = PassiveAggressiveClassifier(max_iter=6000, n_jobs=-1, shuffle=True)
pac.fit(tfidf_train, y_train)

# Predict on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {round(accuracy * 100, 2)}%')

# Build confusion matrix
cm = confusion_matrix(y_test, y_pred, labels=['FAKE', 'REAL'])

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['FAKE', 'REAL'])
disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix')
plt.show()

# Save the model
with open("/content/drive/MyDrive/news/PassiveAggressiveClassifier_model.pkl", 'wb') as file:
    pickle.dump(pac, file)

# Make predictions
while True:
    sample = input("Enter the news (type 'exit' to end the loop): ")
    if sample.lower() == 'exit':
        break
    news = tfidf_vectorizer.transform([sample])
    prediction = pac.predict(news)
    print("Predicted class:", prediction)

