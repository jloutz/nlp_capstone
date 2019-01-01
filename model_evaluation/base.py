import time
import data_preparation as data_preparation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

class Estimator:
    """
    inspired by tensorflow and sklearn
    """
    def train(self, X, y):
        pass

    def evaluate(self, X, y):
        pass

    def predict(self, X, y=None):
        pass


class BaselineEstimator(Estimator):
    """
    Estimator for baseline classifier
    """
    def __init__(self):
        featurizer = TfidfVectorizer(ngram_range=(1,2))
        mnb = MultinomialNB()
        pipeline = Pipeline([("featurizer", featurizer),("clf",mnb)])
        ##TODO param grid
        #params = {"featurizer_ngram_range":[(1,1),(1,2)]}
        self.clf = GridSearchCV(pipeline,{})


    def train(self, X, y):
        print("START training (fit)..")
        t0 = time.time()
        self.clf.fit(X, y)
        print("DONE in ",time.time()-t0)

    def evaluate(self, X, y):
        print("START evaluating")
        t0 = time.time()
        pred = self.clf.predict(X)
        print("DONE in ", time.time() - t0)
        acc = accuracy_score(y,pred)
        return (y,pred,acc)

    def predict(self, X):
        print("START predictions")
        t0 = time.time()
        preds = self.clf.predict(X)
        print("DONE in ", time.time() - t0)
        return list(zip(preds,X))


class Session():
    """
    encapsulates a run through a processing pipeline with a sampled subset of data
    """
    def __init__(self,name, train_size, eval_size, test_size,
                 data_loader: data_preparation.DataLoader,
                 estimator:Estimator):
        provider = data_preparation.DataProvider(name,
                                                 train_size=train_size,
                                                 eval_size=eval_size,
                                                 test_size=test_size)
        provider.sample_from_data(data_loader.get_data())
        self.data_provider = provider
        self.estimator = estimator
        self.name = name

    def train(self):
        X = self.data_provider.x_train
        y = self.data_provider.y_train
        if X is None:
            print("train called although no training data exists in provider (was train_size 0?)")
            return
        self.estimator.train(X,y)

    def evaluate(self):
        X = self.data_provider.x_eval
        y = self.data_provider.y_eval
        if X is None:
            print("evaluate called although no eval data exists in provider (was eval_size 0?)")
            return None
        res = self.estimator.evaluate(X,y)
        print("Evaluation result: ",res[2])##TODO
        return res

    def predict(self, X = None):
        if X is None:
            X = self.data_provider.x_test
            if X is None:
                print("predict called although no predict data exists in provider (was test_size 0?)")
                return None
        res = self.estimator.predict(X)
        print("Predictions: ")
        for pred in res:
            print("{:<30}{:100}".format(pred[0],pred[1]))
        return res

    def __str__(self):
        mystr = "Session {} with train: {} eval: {} test: {}".format(
            self.name,
            self.data_provider.train_size,
            self.data_provider.eval_size,
            self.data_provider.test_size
        )
        return mystr


def run_baseline():
    loader_conf = data_preparation.AmazonQADataLoaderConfig(data_preparation.LOCAL_PROJECT_DIR)
    loader = data_preparation.AmazonQADataLoader(conf=loader_conf)
    loader.load()
    estimator = BaselineEstimator()
    very_small = Session("very_small_baseline",500, 100, 20, loader, estimator)
    #small = Session(3000, 100, 20, loader, estimator)
    #notso_small = Session(30000, 10000, 20, loader, estimator)
    #full = Session(0.7, 0.3, 100, loader, estimator)
    print(very_small)
    very_small.train()
    very_small.evaluate()
    preds = very_small.predict()
    print()
    eval_500 = Session("very_small_baseline", 0, 500, 10, loader, estimator)
    eval_500.evaluate()
    eval_500.predict()
    print()
    eval_500 = Session("very_small_baseline", 0, 500, 10, loader, estimator)
    eval_500.evaluate()
    eval_500.predict()












