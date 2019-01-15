import os
import gzip
import random
import uuid
import numpy as np
## sklearn
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split

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
    def __init__(self,data:dict=None,train_size=0,eval_size=0, test_size=0):
        self.train_size = train_size
        self.eval_size = eval_size
        self.test_size = test_size
        self.x_train = None
        self.x_eval = None
        self.y_train = None
        self.y_eval = None
        self.x_test = None
        self.y_test = None
        self.make_examples_fn = None
        self.train_examples = None
        self.dev_examples = None
        self.test_examples = None
        self.labels = None
        self.data = data
        self.data_stats = None
        if data is not None:
            self._sample_from_data(self.data)
        self.id = uuid.uuid4().hex


    def _data_len_stats(self, len_arr, quantile=90):
        from scipy import stats as st
        stats = {}
        nd_arr = np.array(len_arr)
        stats['len_arr']=len_arr
        stats['doc_count'] = len(len_arr)
        stats['mean'] = nd_arr.mean()
        stats['mode'] = st.mode(nd_arr)
        stats['quantile_' + str(quantile)] = np.percentile(nd_arr, quantile)
        stats['max'] = nd_arr.max()
        stats['min'] = nd_arr.min()
        stats['num_empty'] = len(nd_arr[nd_arr == 0])
        return stats

    def _normalize_lengths(self, data, quantile=90):
        ### data is data['X'] of a loader
        ### remove empty texts and texts of len > quantile (percentile)

        def lengths(dat):
            ## get lengths
            lengths = []
            for texts in dat:
                lengths.extend([len(text.split()) for text in texts])
            return lengths

        ## calculate stats
        len_arr = np.array(lengths(data))
        quantile_count = np.percentile(len_arr, quantile)
        orig_stats = self._data_len_stats(len_arr,quantile)
        newdata = []

        def _check_len(text, max_incl):
            x = len(text.split())
            return x > 0 and x <= max_incl

        for texts in data:
            newtexts = [text for text in texts if _check_len(text, quantile_count)]
            newdata.append(newtexts)

        ## calculate stats
        new_len_arr = np.array(lengths(newdata))
        new_stats = self._data_len_stats(new_len_arr, quantile)

        return newdata,orig_stats,new_stats

    def _sample_from_data(self,data:dict,balance_classes = True, normalize_lengths=True):
        ## works on full data resulting from data loader. Takes samples based on sample sizes
        ## normalizes text lengths, balances classes, and creates lists of data samples and labels
        if self.train_size <= 0 and self.eval_size <= 0 and self.test_size <= 0:
            raise Exception("Please select size of train, eval, or test sets (at least one of them)")

        ## X is list of list of samples,
        ## y is list of category labels
        X_ = data["X"]
        y_ = data["y"]
        ## new arrays for storing processed, normalized data
        X = []
        y = []

        if normalize_lengths:
            res = self._normalize_lengths(X_)
            X_ = res[0]
            self.data_stats =(res[1],res[2])
            print("Before normalize length:")
            print(self.data_stats[0])
            print("After normalize length: ")
            print(self.data_stats[1])

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

        ## now erase train if no training size. (downstream flag for do not train)
        if self.train_size <= 0:
            self.x_train = None
            self.y_train = None

        ## now erase eval if no eval size. (downstream flag for do not eval)
        if self.eval_size <= 0:
            self.x_eval = None
            self.y_eval = None

        ## now make test samples.
        if self.test_size > 0:
            idc = random.sample(range(len(X)), self.test_size)
            self.x_test = [X[i] for i in idc]
            self.y_test = [y[i] for i in idc]


    def _make_examples(self,samples,set_type):
        if self.make_examples_fn is None:
            raise NotImplementedError()
        return self.make_examples_fn(samples,set_type)


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




######### data exploration ##########
def load_provider_local():
    proj_dir = LOCAL_PROJECT_DIR
    ## put it all together
    conf = AmazonQADataLoaderConfig(proj_dir)
    loader = AmazonQADataLoader(conf=conf)
    loader.load()
    provider = DataProvider(loader.data,500,100,20)
    print(provider)
    return provider


def text_len_hist(len_arr):
    import matplotlib.pyplot as plt
    plt.hist(len_arr, bins=1000)
    plt.xlabel("text length (words)")
    plt.show()
    return plt

def explore_data():
    provider = load_provider_local()
    stats = provider.data_stats
    print("original data stats")
    print(stats[0])
    plt = text_len_hist(stats[0]['len_arr'])
    print("trimmed data stats")
    print(stats[1])
    plt2 = text_len_hist(stats[1]['len_arr'])
    return (plt,plt2)

### dataset preparation
def prepare_datasets_for_eval():
    datasets = [("full",.67,.33,100),
                ("lrg-30k",30000,10000,100),
                ("lrg-12k",12000,4000,100),
                ("lrg-3000",3000,1000,100),
                ("med-1500",1500,500,100),
                ("med-900",900,300,100),
                ("small-600",600,200,100),
                ("small-450",450,150,100),
                ("small-300",300,100,100),
                ("small-150",150,50,100)]

    conf = AmazonQADataLoaderConfig(LOCAL_PROJECT_DIR)
    loader = AmazonQADataLoader(conf=conf)
    loader.load()
    providers = {}
    for ds in datasets:
        dp = DataProvider(loader.data, ds[1],ds[2],ds[3])
        providers[ds[0]]=dp
    return providers




