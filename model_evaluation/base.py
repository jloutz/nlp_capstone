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

    def predict(self, X, y=None):
        print("START predictions")
        t0 = time.time()
        preds = self.clf.predict(X)
        print("DONE in ", time.time() - t0)
        return list(zip(preds,X))

    def __str__(self):
        return "Baseline_Estimator_"+str(self.__hash__())



class Session():
    """
    encapsulates a run through a processing pipeline with a sampled subset of data
    """
    def __init__(self,
                 data_provider:data_preparation.DataProvider,
                 estimator:Estimator,
                 name=""):
        self.data_provider = data_provider
        self.estimator = estimator
        self.name = name
        self.evaluation_results = None
        self.prediction_results = None


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
        self.evaluation_results = self.estimator.evaluate(X,y)


    def predict(self, X = None):
        if X is None:
            X = self.data_provider.x_test
            if X is None:
                print("predict called although no predict data exists in provider (was test_size 0?)")
                return None
        self.prediction_results = self.estimator.predict(X)


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
        import os
        import pickle
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
        if self.prediction_results:
            obj["prediction_results"] = self.prediction_results

        with open(output_path,'wb') as f:
            print("Dumping a big fat pickle to {}...".format(output_path))
            pickle.dump(obj,f)
            print("Done!")


    def persist_name(self):
        persist_name = self.__str__()
        persist_name+= ".pkl"
        return persist_name


    def __str__(self):
        mystr = ""
        if self.name:
            mystr += self.name
        mystr += "-{}".format(str(self.estimator))
        if self.data_provider.x_train:
            s = "-train" + str(len(self.data_provider.x_train))
            mystr += s
        if self.data_provider.x_eval:
            s = "-eval" + str(len(self.data_provider.x_eval))
            mystr += s
        if self.data_provider.x_test:
            s = "-test" + str(len(self.data_provider.x_test))
            mystr += s
        return mystr


def run_baseline():
    loader_conf = data_preparation.AmazonQADataLoaderConfig(data_preparation.LOCAL_PROJECT_DIR)
    loader = data_preparation.AmazonQADataLoader(conf=loader_conf)
    loader.load()
    very_small_data = data_preparation.DataProvider(500, 100, 20,loader.data)
    estimator = BaselineEstimator()
    very_small = Session(very_small_data, estimator,"very_small")
    #small = Session(3000, 100, 20, loader, estimator)
    #notso_small = Session(30000, 10000, 20, loader, estimator)
    #full = Session(0.7, 0.3, 100, loader, estimator)
    print(very_small)
    very_small.train()
    very_small.evaluate()
    preds = very_small.predict()
    print(preds)
    print()
    eval_500 = data_preparation.DataProvider(0, 500, 10, loader.data)
    eval_500_session = Session(eval_500, estimator)
    print(eval_500_session)
    eval_500_session.evaluate()
    eval_500_session.predict()
    print()
    predict_40 = data_preparation.DataProvider(0, 0, 40, loader.data)
    predict_40_session = Session( predict_40, estimator)
    print(predict_40_session)
    predict_40_session.predict()

#run_baseline()










