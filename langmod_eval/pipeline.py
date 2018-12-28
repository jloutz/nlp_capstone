import langmod_eval.data_preprocessing
import re

class Pipeline:
    def __init__(self):
        self.labels = None
        self.rawtext = None
        self.preprocessed_text = None

    def load(self):
        labeled_text = langmod_eval.data_preprocessing.load_labeled_text()
        self.labels = labeled_text["labels"]
        self.rawtext = labeled_text["texts"]

    def preprocess_text(self,text):
        return " ".join(re.split("(?u)[^a-zA-Z0-9\-_]+", text)).strip().lower()

    def preprocess_raw_text(self):
        if self.rawtext is None:
            self.load()
        self.preprocessed_text = []
        for doc_list in self.rawtext:
            self.preprocessed_text.append([self.preprocess_text(text) for text in doc_list])

    def prepare_data_sample(self,size=200):
        if self.preprocessed_text is None:
            self.preprocess_raw_text()








