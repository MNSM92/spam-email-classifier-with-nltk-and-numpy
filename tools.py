import nltk
from nltk import word_tokenize, NaiveBayesClassifier, classify
from nltk.text import Text
import os
import codecs
# nltk.download('punkt')

def tokenize(input):
    word_list = []
    for word in word_tokenize(input):
        word_list.append(word)
    return word_list

def get_features(text):
    features = {}
    word_list = [word for word in word_tokenize(text.lower())]
    for word in word_list:
        features[word] = True
    return features

def read_in(folder):
    files = os.listdir(folder)
    a_list = []
    for a_file in files:
        if not a_file.startswith("."):
            file_path = os.path.join(folder, a_file)
            with codecs.open(file_path, "r", encoding="ISO-8859-1", errors="ignore") as f:
                a_list.append(f.read())
    return a_list


def train(features, proportion):
    train_size = int(len(features) * proportion)
    train_set, test_set = features[:train_size], features[train_size:]
    print(f"Training set size = {str(len(train_set))} emails")
    print(f"Test set size = {str(len(test_set))} emails")
    classifier = NaiveBayesClassifier.train(train_set)
    return train_set, test_set, classifier


def evaluate(train_set, test_set, classifier):
    print(f"Accuracy on the training set = {str(classify.accuracy(classifier, train_set))}")
    print(f"Accuracy on the test set = {str(classify.accuracy(classifier, test_set))}")
    classifier.show_most_informative_features(50)

def concordance(data_list, search_word):
    all_words = [word.lower() for email in data_list for word in word_tokenize(email)]
    text_list = Text(all_words)
    text_list.concordance(search_word)