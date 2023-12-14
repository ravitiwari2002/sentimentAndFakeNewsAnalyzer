# Sentiment Analyzer and Fake News Detector

This project is a Graphical User Interface (GUI) application built using Python's tkinter library. It provides two main functionalities: a Sentiment Analyzer and a Fake News Detector.

## Features Implemented

### Sentiment Analyzer

The Sentiment Analyzer uses the TextBlob library to analyze the sentiment of a given text. If the input text starts with 'http://' or 'https://', the code treats it as a URL and uses the newspaper library to extract the text from the article at that URL.

### Fake News Detector

The Fake News Detector uses a Passive Aggressive Classifier to classify a given text as real or fake news. The model is trained on a dataset loaded from a CSV file named "fake_or_real_news.csv". The texts are transformed into a numerical representation using the TfidfVectorizer from the scikit-learn library.

### GUI

The GUI is created using the tkinter library. It starts with a main menu that lets the user choose between the Sentiment Analyzer and the Fake News Detector. Depending on the user's choice, it then opens the corresponding tool in the same window. Each tool has a "Back" button that takes the user back to the main menu.
