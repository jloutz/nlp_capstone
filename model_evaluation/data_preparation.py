import os
import gzip
import random
import uuid
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


LOCAL_PROJECT_DIR = "C:/Projects/udacity-capstone"
GCP_PROJECT_DIR = "/home/jloutz67/nlp_capstone"

class AmazonQADataLoaderConfig:
    """ config for loader """
    def __init__(self,project_dir):
        if not os.path.exists(project_dir):
            raise Exception("project path incorrect or doesnt exist...")
        self.json_dir = os.path.join(project_dir, "data/amazon_qa/json")
        self.raw_dir = os.path.join(project_dir, "data/amazon_qa/raw")
        self.persist_dir = os.path.join(project_dir, "data/amazon_qa")
        persist_name = "labeled_text.pkl"
        self.persist_path = os.path.join(project_dir, persist_name)


class AmazonQADataLoader(DataLoader):
    """
    extract compressed gz files with data and process the results.
    persists processed results as a dictionary.
    load this data lazily or eagerly if lazy = False is passed to load method.
    """
    def __init__(self,conf:AmazonQADataLoaderConfig):
        self.json_dir = conf.json_dir
        self.raw_dir = conf.raw_dir
        self.persist_path = conf.persist_path
        self.data = None

    def get_data(self):
        if self.data is None:
            self.load()
        return self.data

    def load(self,lazy=True, persist=True):
        if lazy and os.path.exists(self.persist_path):
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
        if not lazy or not os.path.exists(self.json_dir) or len(os.listdir(self.json_dir))==0:
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
            with open(os.path.join(self.json_dir,category_file_name), "rt") as f:
                for line in f.readlines():
                    obj = ast.literal_eval(line)
                    category_texts.append(obj['question'])
                    category_texts.append(obj['answer'])
            texts.append(category_texts)
        print("Done!")
        return labels, texts


    def _unpack(self):
        ## extract raw data from .gz files and persist as json
        if not os.path.exists(self.raw_dir) or len(os.listdir(self.raw_dir)) == 0:
            raise Exception("No raw data to extract...")
        if not os.path.exists(self.json_dir):
            os.makedirs(self.json_dir,exist_ok=True)
        print("Unpacking..")
        for gz_file in os.listdir(self.raw_dir):
            with gzip.open(os.path.join(self.raw_dir,gz_file), mode="rt") as gz_f:
                f_content = gz_f.read()
                out_name = gz_file.split(".gz")[0]
                out_path = os.path.join(self.json_dir,out_name)
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
    def __init__(self,data:dict,train_size=0,eval_size=0, test_size=0):
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
        self.data = data
        self._sample_from_data(self.data)
        self.id = uuid.uuid4().hex

    def _sample_from_data(self,data:dict,balance_classes = True):
        if self.train_size <= 0 and self.eval_size <= 0 and self.test_size <= 0:
            raise Exception("Please select size of train, eval, or test sets (at least one of them)")
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
        self.labels = list(set(y))

        ## make sizes absolute if percent
        if 0 < self.train_size < 1:
            self.train_size = int(self.train_size * len(X))
            self.eval_size = int(len(X)-self.train_size)
        elif 0 < self.eval_size < 1:
            self.eval_size = int(self.eval_size * len(X))
            self.train_size = int(len(X) - self.eval_size)

        ## balance sizes if one of them is zero (needed for input to train test split)
        train_size = self.train_size if self.train_size > 0 else self.eval_size
        eval_size = self.eval_size if self.eval_size > 0 else self.train_size

        if train_size > 0 or eval_size > 0:
            ## train test split with stratify true if balance classes
            dostrat = y if balance_classes else None
            self.x_train, self.x_eval, self.y_train, self.y_eval = \
                train_test_split(X,y,train_size=train_size,test_size=eval_size,stratify=dostrat)

        ## now erase train if no training size. (flag for do not train)
        if self.train_size <= 0:
            self.x_train = None
            self.y_train = None

        ## now erase eval if no eval size. (flag for do not eval)
        if self.train_size <= 0:
            self.x_train = None
            self.y_train = None

        ## now make test samples.
        if self.test_size > 0:
            idc = random.sample(range(len(X)), self.test_size)
            self.x_test = [X[i] for i in idc]
            self.y_test = [y[i] for i in idc]


    def _make_examples(self,samples,set_type):
        ## samples should be list of tuples (text,label)
        examples = []
        for (i,sample) in enumerate(samples):
            guid = "%s-%s" % (set_type, i)
            text = tokenization.convert_to_unicode(sample[0])
            #if set_type == "test":
             #   label = "0"
            #else:
            label = tokenization.convert_to_unicode(sample[1])
            examples.append(
                InputExample(guid=guid, text_a=text, label=label))
        return examples


    def get_train_examples(self):
        ## for bert
        if self.x_train == None:
            return []
        if self.train_examples is None:
            self.train_examples = self._make_examples(zip(self.x_train, self.y_train), "train")
        return self.train_examples

    def get_dev_examples(self):
        ## for bert
        if self.x_eval == None:
            return []
        if self.dev_examples is None:
            self.dev_examples = self._make_examples(zip(self.x_eval, self.y_eval), "eval")
        return self.dev_examples

    def get_test_examples(self):
        ## for bert
        if self.x_test == None:
            return []
        if self.test_examples is None:
            self.test_examples = self._make_examples(zip(self.x_test, self.y_test), "test")
        return self.test_examples

    def get_labels(self):
        ## for bert
        return self.labels

    def __str__(self):
        mystr = "DataProvider-{0}-Train{1}-Eval{2}-Test{3}".format(self.id,
                                                                   str(self.train_size),
                                                                   str(self.eval_size),
                                                                   str(self.test_size))
        return mystr



def load_amazon_qa_data(local=True):
    if local:
        proj_dir = LOCAL_PROJECT_DIR
    else:
        proj_dir = GCP_PROJECT_DIR
    ## put it all together
    conf = AmazonQADataLoaderConfig(proj_dir)
    loader = AmazonQADataLoader(conf=conf)
    loader.load(lazy=False)
    provider = DataProvider(loader.data,500,100,20)
    print(provider)
    return provider



if __name__ == "__main__":
    load_amazon_qa_data()
