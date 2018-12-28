import os
import gzip
from sklearn.externals import joblib
import pathlib

class DataPreprocessor():
    """
    brazenly taken from or let's say "inspired by")
    the DataProcessor implementation in bert-master.
    Those google guys are smart.
    """
    def get_train_dataset(self):
        """
        return a tuple of arrays (training data, training labels)
        :return: tuple of arrays (training data, training labels)
        """
        pass

    def get_test_dataset(self):
        """
        return a tuple of arrays (test data, test labels)
        :return: tuple of arrays (test data, test labels)
        """
        pass



json_dir = pathlib.Path("../data/json")
raw_dir = pathlib.Path("../data/raw")

persist_dir = pathlib.Path("../data")
persist_name = "labeled_text.pkl"
persist_path = persist_dir / persist_name

def unpack():
    if not os.path.exists(json_dir):
        os.mkdir(str(json_dir))
    for gz_file in os.listdir(raw_dir):
        with gzip.open(raw_dir / gz_file, mode="rt") as gz_f:
            f_content = gz_f.read()
            out_name = gz_file.split(".gz")[0]
            out_path = json_dir / out_name
            if os.path.exists(out_path):
                os.remove(str(out_path))
            with open(out_path, mode="wt") as out:
                out.write(f_content)

def extract_text_and_labels():
    import ast
    import re
    labels = []
    texts = []
    for category_file_name in os.listdir(json_dir):
        print("Loading text from: ", category_file_name)
        label =  re.split("qa_(.*)\\.json",category_file_name)[1].lower()
        labels.append(label)
        category_texts = []
        with open(json_dir / category_file_name,"rt") as f:
            for line in f.readlines():
                obj = ast.literal_eval(line)
                category_texts.append(obj['question'])
                category_texts.append(obj['answer'])
        texts.append(category_texts)
    return labels, texts


def persist_labeled_text():
    if not persist_dir.exists():
        raise("persist dir doesn't exist..")
    labels, texts = extract_text_and_labels()
    print("persisting labels and texts.. ")
    joblib.dump({"labels":labels,"texts":texts},persist_path)


def load_labeled_text():
    labeled_text = joblib.load(persist_path)
    return labeled_text

