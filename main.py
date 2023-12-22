from train import SpamClassifier


def train_and_classify():
    spam_classifier = SpamClassifier()
    spam_classifier.train_and_evaluate()

    spam_classifier.test_specific(("Participate in our new lottery!", "Try out this new medicine"),
                                  ("See the minutes from the last meeting attached",
                                   "Investors are coming to our office on Monday"))

    spam_classifier.classify_email("Participate in our new lottery!")
    spam_classifier.search_word("stocks")


if __name__ == "__main__":
    train_and_classify()
