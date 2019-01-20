import uuid
import numpy
import time
import pandas

from data_preparation import DataProvider
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib


class Estimator:
    """
    Simple base interface for an estimator (classifier, learner etc)
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
        ## tfidf with uni and bi-grams
        featurizer = TfidfVectorizer(ngram_range=(1,2))
        ## naive bayes (all hail Reverend Bayes!)
        mnb = MultinomialNB()
        pipeline = Pipeline([("featurizer", featurizer),("clf",mnb)])
        ## using GridSearchCV for cross-validation
        self.clf = GridSearchCV(pipeline,{})
        self.id = uuid.uuid5(uuid.NAMESPACE_OID,str(self.clf)).hex

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

    def predict(self, X, y=None):
        print("START predictions")
        t0 = time.time()
        predprobs = self.clf.predict_proba(X)
        classes = self.clf.classes_

        fulldata = []
        for i,predprob in enumerate(predprobs):
            data = []
            data.append(y[i])
            data.append(classes[numpy.argsort(predprob)[::-1][0]])
            data.append(X[i])
            data.extend(predprob)
            fulldata.append(data)

        cols = ["true", "pred", "text"]
        cols.extend(classes)
        df = pandas.DataFrame(data=fulldata, columns=cols)
        print("DONE in ", time.time() - t0)
        return df

    def __str__(self):
        return "Baseline_Estimator_{}".format(self.id)



class Session():
    """
    encapsulates a run through a processing pipeline with a sampled subset of data
    Requires a passed-in data provider and estimator in constructor
    Records evaluation and prediction results
    """
    def __init__(self,
                 data_provider:DataProvider,
                 estimator:Estimator,
                 name=""):
        self.data_provider = data_provider
        self.estimator = estimator
        self.name = name
        self.evaluation_results = None
        self.prediction_results = None


    def train(self):
        ## gets training data from data provider and calls estimator.train
        X = self.data_provider.x_train
        y = self.data_provider.y_train
        if X is None:
            print("train called although no training data exists in provider (was train_size 0?)")
            return
        self.estimator.train(X,y)

    def evaluate(self):
        ## gets eval data from data provider and calls estimator.evaluate
        X = self.data_provider.x_eval
        y = self.data_provider.y_eval
        if X is None:
            print("evaluate called although no eval data exists in provider (was eval_size 0?)")
            return None
        self.evaluation_results = self.estimator.evaluate(X,y)


    def predict(self, X = None,y=None):
        ## gets prediction data from data provider and calls estimator.predict
        if X is None:
            X = self.data_provider.x_test
            y = self.data_provider.y_test
            if X is None:
                print("predict called although no predict data exists in provider (was test_size 0?)")
                return None
        self.prediction_results = self.estimator.predict(X,y)


    def show(self):
        print()
        if self.evaluation_results:
            print("Evaluation results:")
            print(self.evaluation_results)
        if self.prediction_results:
            print("Predictions: ")
            for pred in self.prediction_results:
                print("{:<30}{:100}".format(pred[0],pred[1]))

    def persist(self,output_dir):
        ## persists a dictionary of data and eval and prediction results
        import os
        os.makedirs(output_dir,exist_ok=True)
        output_path = os.path.join(output_dir,self.persist_name())
        obj = {}
        if self.data_provider.x_train:
            obj["x_train"]=self.data_provider.x_train
            obj["y_train"] = self.data_provider.y_train
        if self.data_provider.x_eval:
            obj["x_eval"] = self.data_provider.x_eval
            obj["y_eval"] = self.data_provider.y_eval
        if self.evaluation_results:
            obj["evaluation_results"] = self.evaluation_results
        if self.data_provider.x_test:
            obj["x_test"] = self.data_provider.x_test
            obj["y_test"] = self.data_provider.y_test
        if self.prediction_results is not None:
            obj["prediction_results"] = self.prediction_results
        print("Dumping a big fat pickle to {}...".format(output_path))
        joblib.dump(obj, output_path)
        print("Done!")

    def persist_name(self):
        ## my name for persisting
        persist_name = self.__str__()
        persist_name+= ".pkl"
        return persist_name


    def __str__(self):
        ## returns a string identifying this session
        mystr = "Session-"
        if self.name:
            mystr += self.name+"-"
        mystr += "{0}-{1}".format(str(self.estimator),str(self.data_provider))
        return mystr

################## run evaluation baseline ###################
import config
def _load_datasets_for_evaluation(dir=config.LOCAL_DATASETS_DIR,name="datasets_for_eval.pkl"):
    import os
    loadpath = os.path.join(dir,name)
    print("Loading {}...".format(loadpath))
    datasets = joblib.load(loadpath)
    print("Done!")
    return datasets


################## entry point for baseline evaluation #######
def run_evaluation_baseline(datasets_dir=config.LOCAL_DATASETS_DIR,
                            output_dir = config.LOCAL_SESSIONS_DIR,
                            datasets_name = "datasets_for_eval.pkl",
                            suffix="_1"):
    ## call this method to run an evaluation on a dataset using baseline evaluaton
    ## datasets_dir is the directory where the datasets prepared with data_preparation.prepare_datasets_for_eval
    ## are saved.
    ## output_dir is dir where sessions (results) are saved
    ## datasets_name is name of dataset pkl
    ## suffix will be appended to session name - good for multiple runs with same dataset to avoid name collision.
    datasets = _load_datasets_for_evaluation(dir=datasets_dir,name=datasets_name)
    for key,dataset in datasets.items():
        print(key)
        estimator = BaselineEstimator()
        session = Session(dataset,estimator,key+suffix)
        session.train()
        session.evaluate()
        print(session.evaluation_results[2])
        session.predict()
        session.persist(output_dir=output_dir)

