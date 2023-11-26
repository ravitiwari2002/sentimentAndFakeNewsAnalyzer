import tkinter as tk
from tkinter import messagebox
import pandas as pd
from newspaper import Article
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from textblob import TextBlob

# Declare text_entry, result_label, and summary_label as global variables
text_entry = None
result_label = None
summary_label = None

# Load the data and train the Fake News Detector
data = pd.read_csv("fake_or_real_news.csv")
data['fake'] = data['label'].apply(lambda x: 0 if x == "REAL" else 1)
data = data.drop("label", axis=1)
X, y = data["text"], data["fake"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vectorized = vectorizer.fit_transform(X_train)
clf = PassiveAggressiveClassifier(
    max_iter=50)  # Passive Aggressive Classifier is often used for text classification problems
clf.fit(X_train_vectorized, y_train)


def get_sentiment():
    # Use the global keyword to indicate that we are using the global text_entry, result_label, and summary_label variables
    global text_entry, result_label, summary_label
    input_text = text_entry.get("1.0", "end-1c")
    summary = ""
    if input_text.startswith('http://') or input_text.startswith('https://'):
        try:
            article = Article(input_text)
            article.download()
            article.parse()
            article.nlp()
            input_text = article.summary
            summary = article.summary  # Get the complete summary
        except Exception as e:
            messagebox.showerror("Error", "Failed to download or parse the article.")
            return
    try:
        blob = TextBlob(input_text)
        sentiment = blob.sentiment.polarity
        sentiment = round(sentiment, 2)
        result_label.config(text="Sentiment: " + str(sentiment))

        # Provide a response based on the sentiment score
        if sentiment > 0:
            response = "The overall sentiment of the text is positive."
        elif sentiment < 0:
            response = "The overall sentiment of the text is negative."
        else:
            response = "The overall sentiment of the text is neutral."
        summary_label.config(
            text=f"Summary: {summary}\n\n{response}")
    except Exception as e:
        messagebox.showerror("Error", "Failed to analyze the sentiment of the text.")


def detect_fake_news():
    # Use the global keyword to indicate that we are using the global text_entry variable
    global text_entry
    input_text = text_entry.get("1.0", "end-1c")
    vectorized_text = vectorizer.transform([input_text])
    prediction = clf.predict(vectorized_text)
    prediction_proba = clf.decision_function(vectorized_text)
    if abs(prediction_proba) < 0.3:  # You can adjust this threshold to whatever you find appropriate
        result_label.config(text="Unable to verify the news.")
    elif prediction[0] == 0:
        result_label.config(text="The news is real.")
    else:
        result_label.config(text="The news is fake.")

def open_sentiment_analyzer():
    # Use the global keyword to indicate that we are using the global text_entry, result_label, and summary_label variables
    global text_entry, result_label, summary_label
    # Clear the window
    for widget in root.winfo_children():
        widget.destroy()

    # Use a larger, nicer font for the labels and button
    font = ("Helvetica", 16)

    label = tk.Label(root, text="Enter URL or text:", font=font, bg='light blue')
    label.pack(pady=10)  # Add vertical padding

    text_entry = tk.Text(root, height=10, width=50)
    text_entry.pack(pady=10)  # Add vertical padding

    result_label = tk.Label(root, text="", font=font, bg='light blue')
    result_label.pack(pady=10)  # Add vertical padding

    summary_label = tk.Label(root, text="", font=font, bg='light blue')
    summary_label.pack(pady=20)  # Add more vertical padding

    sentiment_button = tk.Button(root, text="Get Sentiment", command=get_sentiment, font=font)
    sentiment_button.pack(pady=10)  # Add vertical padding

    back_button = tk.Button(root, text="Back", command=open_main_menu, font=font)  # Add a Back button
    back_button.pack(pady=10)  # Add vertical padding


def open_fake_news_detector():
    # Use the global keyword to indicate that we are using the global text_entry and result_label variables
    global text_entry, result_label
    # Clear the window
    for widget in root.winfo_children():
        widget.destroy()

    # Use a larger, nicer font for the labels and button
    font = ("Helvetica", 16)

    label = tk.Label(root, text="Enter the news text:", font=font, bg='light blue')
    label.pack(pady=10)  # Add vertical padding

    text_entry = tk.Text(root, height=10, width=50)
    text_entry.pack(pady=10)  # Add vertical padding

    result_label = tk.Label(root, text="", font=font, bg='light blue')
    result_label.pack(pady=10)  # Add vertical padding

    detect_button = tk.Button(root, text="Detect Fake News", command=detect_fake_news, font=font)
    detect_button.pack(pady=10)  # Add vertical padding

    back_button = tk.Button(root, text="Back", command=open_main_menu, font=font)  # Add a Back button
    back_button.pack(pady=10)  # Add vertical padding


def open_main_menu():
    # Clear the window
    for widget in root.winfo_children():
        widget.destroy()

    # Use a larger, nicer font for the labels and button
    font = ("Helvetica", 16)

    label = tk.Label(root, text="Choose an option:", font=font)
    label.pack(pady=10)  # Add vertical padding

    sentiment_button = tk.Button(root, text="Sentiment Analyzer", command=open_sentiment_analyzer, font=font)
    sentiment_button.pack(pady=10)  # Add vertical padding

    fake_news_button = tk.Button(root, text="Fake News Detector", command=open_fake_news_detector, font=font)
    fake_news_button.pack(pady=10)  # Add vertical padding


root = tk.Tk()
root.state('zoomed')  # Make the window open in maximized mode
root.title("Choose an option")  # Set the title of the window
root.configure(bg='light blue')  # Set the background color of the window

# Create a frame in the center of the window
frame = tk.Frame(root)
frame.place(relx=0.5, rely=0.5, anchor='center')

open_main_menu()

root.mainloop()
