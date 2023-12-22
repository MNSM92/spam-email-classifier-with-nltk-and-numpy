from tools import get_features, read_in, train, evaluate, concordance
import random

class SpamClassifier:
    def __init__(self, spam_dir="enron1/spam/", ham_dir="enron1/ham/"):
        self.spam_list = read_in(spam_dir)
        self.ham_list = read_in(ham_dir)
        self.all_emails = [(email_content, "spam") for email_content in self.spam_list]
        self.all_emails += [(email_content, "ham") for email_content in self.ham_list]
        random.seed(42)
        random.shuffle(self.all_emails)
        self.all_features = [(get_features(email), label) for (email, label) in self.all_emails]
        self.train_set, self.test_set, self.classifier = train(self.all_features, 0.8)

    def train_and_evaluate(self):
        evaluate(self.train_set, self.test_set, self.classifier)

    def test_specific(self, test_spam_list, test_ham_list):
        test_emails = [(email_content, "spam") for email_content in test_spam_list]
        test_emails += [(email_content, "ham") for email_content in test_ham_list]
        new_test_set = [(get_features(email), label) for (email, label) in test_emails]
        evaluate(self.train_set, new_test_set, self.classifier)

    def classify_email(self, email):
        print(self.classifier.classify(get_features(email)))

    def search_word(self, word="stocks"):
        concordance(self.ham_list, word)
        concordance(self.spam_list, word)
