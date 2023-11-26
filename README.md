# Project Description

This project is a **Graphical User Interface (GUI)** application built using **Python's tkinter library**. It provides two main features: a **Sentiment Analyzer** and a **Fake News Detector**.

## Features

### Sentiment Analyzer

The Sentiment Analyzer uses the **TextBlob library** to analyze the sentiment of a given text. TextBlob is a Python library for processing textual data that uses **Natural Language Processing (NLP)** techniques. The sentiment function of TextBlob returns two properties: polarity and subjectivity. Polarity is a float value within the range [-1.0, 1.0] where -1 means negative sentiment and 1 means a positive sentiment. We're using only the polarity in this code.

If the input text starts with 'http://' or 'https://', the code treats it as a URL and uses the **newspaper library** to extract the text from the article at that URL. The newspaper library is an advanced news extraction, article extraction, and content curation library in Python.

### Fake News Detector

The Fake News Detector uses a **Passive Aggressive Classifier** to classify a given text as real or fake news. The Passive Aggressive Classifier is an online learning algorithm that is well suited for large scale learning and text classification.

The model is trained on a dataset loaded from a CSV file named "fake_or_real_news.csv". The dataset should contain news texts and their labels (either "REAL" or "FAKE"). The texts are transformed into a numerical representation using the **TfidfVectorizer** from the **scikit-learn library**. TfidfVectorizer converts a collection of raw documents to a matrix of TF-IDF features. It's equivalent to CountVectorizer followed by TfidfTransformer.

### GUI

The GUI is created using the tkinter library. It starts with a main menu that lets the user choose between the Sentiment Analyzer and the Fake News Detector. Depending on the user's choice, it then opens the corresponding tool in the same window. Each tool has a "Back" button that takes the user back to the main menu.

The Sentiment Analyzer tool lets the user enter a URL or text, and then it displays the sentiment of the text when the "Get Sentiment" button is clicked.

The Fake News Detector tool lets the user enter a news text, and then it displays whether the news is real or fake when the "Detect Fake News" button is clicked. If the model is not confident in its prediction, it displays "Unable to verify the news."
