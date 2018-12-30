import os
import gzip
import random
import pathlib
## sklearn
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
# bert-master
import tokenization
from run_classifier import DataProcessor, InputExample


class DataLoader():
    """
    data loader ifc
    """
    def get_data(self):
        pass

    def load(self, lazy=True, persist=True):
        pass



class AmazonQADataLoaderConfig:
    """ config for loader """
    def __init__(self):
        self.json_dir = "../data/amazon_qa/json"
        self.raw_dir = "../data/amazon_qa_/raw"
        self.persist_dir = "../data/amazon_qa"
        self.persist_name = "labeled_text.pkl"


class AmazonQADataLoader(DataLoader):
    """
    extract compressed gz files with data and process the results.
    persists processed results as a dictionary.
    load this data lazily or eagerly if lazy = False is passed to load method.
    """
    def __init__(self,conf:AmazonQADataLoaderConfig):
        self.json_dir = pathlib.Path(conf.json_dir)
        self.raw_dir = pathlib.Path(conf.raw_dir)
        self.persist_dir = pathlib.Path(conf.persist_dir)
        self.persist_name = conf.persist_name
        self.persist_path = self.persist_dir / self.persist_name
        self.data = None

    def get_data(self):
        if self.data is None:
            self.load()
        return self.data

    def load(self,lazy=True, persist=True):
        if lazy and self.persist_path.exists():
            print("loading from ",self.persist_path)
            self.data = joblib.load(self.persist_path)
            print("done!")
        else:
            labels, texts = self._extract_text_and_labels(lazy)
            self.data = {"y": labels, "X": texts}
            if persist:
                print("persisting labels and texts to ",self.persist_path)
                joblib.dump(self.data, self.persist_path)
                print("Done!")

    def _extract_text_and_labels(self,lazy=True):
        ## parse data in json files preserving only labels (category name) and texts
        ## both questions and answers found in json are treated as example texts for each category
        if not lazy or not self.json_dir.exists() or len(os.listdir(self.json_dir))==0:
            self._unpack()
        import ast
        import re
        labels = []
        texts = []
        print("Extracting...")
        for category_file_name in os.listdir(self.json_dir):
            print("Loading text from: ", category_file_name)
            label = re.split("qa_(.*)\\.json", category_file_name)[1].lower()
            labels.append(label)
            category_texts = []
            with open(self.json_dir / category_file_name, "rt") as f:
                for line in f.readlines():
                    obj = ast.literal_eval(line)
                    category_texts.append(obj['question'])
                    category_texts.append(obj['answer'])
            texts.append(category_texts)
        print("Done!")
        return labels, texts


    def _unpack(self):
        ## extract raw data from .gz files and persist as json
        if not self.raw_dir.exists() or len(os.listdir(self.raw_dir)) == 0:
            raise Exception("No raw data to extract...")
        if not os.path.exists(self.json_dir):
            pathlib.Path.mkdir(self.json_dir,parents=True)
        print("Unpacking..")
        for gz_file in os.listdir(self.raw_dir):
            with gzip.open(self.raw_dir / gz_file, mode="rt") as gz_f:
                f_content = gz_f.read()
                out_name = gz_file.split(".gz")[0]
                out_path = self.json_dir / out_name
                if os.path.exists(out_path):
                    os.remove(str(out_path))
                with open(out_path, mode="wt") as out:
                    out.write(f_content)
        print("Done!")


class DataProvider:
    """
    similar to DataProcessor in bert-base for preparing amazon qa data.
    provides data in a form expected from bert, ulmfit as well as sklearn classifiers.
    """
    def __init__(self,name,train_size=200,eval_size=100, test_size=20):
        self.name = name
        self.train_size = train_size
        self.eval_size = eval_size
        self.test_size = test_size
        self.x_train = None
        self.x_eval = None
        self.y_train = None
        self.y_eval = None
        self.x_test = None
        self.y_test = None
        self.train_examples = None
        self.dev_examples = None
        self.test_examples = None
        self.labels = None

    def sample_from_data(self,data:dict,balance_classes = True):
        ## take train and eval samples from data
        X_ = data["X"]
        y_ = data["y"]
        ## prepare for train test split
        X = []
        y = []

        ## if balance classes, downsample to smallest class size
        downsampleto = min([len(x) for x in X_])

        for (i,label) in enumerate(y_):
            vals = X_[i]
            class_sample_size = downsampleto if balance_classes else len(vals)
            y.extend([label for i in range(class_sample_size)])
            X.extend(vals[:class_sample_size])

        if len(X) < self.train_size + self.eval_size:
            raise Exception("not enough data...")
        self.labels = set(y)
        ## train test split with stratify true if balance classes
        dostrat = y if balance_classes else None
        self.x_train, self.x_eval, self.y_train, self.y_eval = \
            train_test_split(X,y,train_size=self.train_size,test_size=self.eval_size,stratify=dostrat)
        ## now make test samples.
        idc = random.sample(range(len(X)), self.test_size)
        self.x_test = [X[i] for i in idc]
        self.y_test = [y[i] for i in idc]


    def _make_examples(self,samples,set_type):
        ## samples should be list of tuples (text,label)
        examples = []
        for (i,sample) in enumerate(samples):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(sample[0])
            if set_type == "test":
                label = "0"
            else:
                label = tokenization.convert_to_unicode(sample[1])
            examples.append(
                InputExample(guid=guid, text_a=text, label=label))
        return examples


    def get_train_examples(self):
        ## for bert
        if self.train_examples is None:
            self.train_examples = self._make_examples(zip(self.x_train, self.y_train), "train")
        return self.train_examples

    def get_dev_examples(self):
        ## for bert
        if self.dev_examples is None:
            self.dev_examples = self._make_examples(zip(self.x_eval, self.y_eval), "eval")
        return self.dev_examples

    def get_test_examples(self):
        ## for bert
        if self.test_examples is None:
            self.test_examples = self._make_examples(zip(self.x_test, self.y_test), "test")
        return self.test_examples

    def get_labels(self):
        ## for bert
        return self.labels

    def __str__(self):
        return  "provider to string...TODO"

def load_amazon_qa_data():
    ## put it all together
    conf = AmazonQADataLoaderConfig()
    loader = AmazonQADataLoader(conf=conf)
    loader.load()
    provider = DataProvider("testProvider")
    provider.sample_from_data(loader.data)
    print(provider)
    return provider





