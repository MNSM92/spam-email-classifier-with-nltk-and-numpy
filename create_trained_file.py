import joblib
from train import SpamClassifier


def save_model(classifier, filename="spam_classifier_model.joblib"):
    joblib.dump(classifier, filename)


spam_classifier = SpamClassifier()
spam_classifier.train_and_evaluate()
save_model(spam_classifier.classifier, "spam_classifier_model.joblib")