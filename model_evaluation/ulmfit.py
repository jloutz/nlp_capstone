import uuid

from fastai.metrics import accuracy
from fastai.text import TextLMDataBunch, TextClasDataBunch, \
    language_model_learner, URLs, text_classifier_learner, DatasetType
from fastai import train
from fastai.basic_train import Learner
import data_preparation as data_preparation
import evaluation
from base import Estimator, Session
import pandas as pd
import numpy as np


class ULMFiTEstimator(Estimator):
    def __init__(self):
        self.pretrained_model=URLs.WT103
        self.drop_mult=0.7
        self.lm_learner = None
        self.clf_learner = None
        self.id = uuid.uuid5(uuid.NAMESPACE_OID, str([self.pretrained_model,self.drop_mult])).hex
        pass

    def train(self, lmdata, clfdata):
        self.lm_learn = language_model_learner(lmdata,
                                               pretrained_model=self.pretrained_model,
                                               drop_mult=self.drop_mult)
        self.lm_learn.fit_one_cycle(1)
        self.lm_learn.save_encoder('ft_enc')
        self.clf_learn = text_classifier_learner(clfdata, drop_mult=0.7)
        self.clf_learn.load_encoder('ft_enc')
        self.clf_learn.metrics = [accuracy]
        self.clf_learn.fit_one_cycle(1)

    def evaluate(self, **kwargs):
        preds, targets = self.clf_learn.get_preds()
        predictions = np.argmax(preds, axis=1)
        return (predictions, targets)

    def predict(self, **kwargs):
        preds, targets = self.clf_learn.get_preds(ds_type=DatasetType.TEST)
        predictions = np.argmax(preds, axis=1)
        return (predictions, targets)
    def __str__(self):
        return "ULMFiT_Estimator_{}".format(self.id)



class ULMFiTSession(Session):
    def __init__(self, data_provider: data_preparation.DataProvider, estimator: Estimator, name=""):
        super().__init__(data_provider, estimator, name)
        # make df out of X and y
        _trn = {"label": self.data_provider.y_train, "text": self.data_provider.x_train}
        _val = {"label": self.data_provider.y_eval, "text": self.data_provider.x_eval}
        _test = {"label": self.data_provider.y_test, "text": self.data_provider.x_test}

        df_trn = pd.DataFrame.from_dict(_trn)
        df_val = pd.DataFrame.from_dict(_val)
        df_test = pd.DataFrame.from_dict(_test)
        # Language model data
        self.lm_databunch = TextLMDataBunch.from_df(train_df=df_trn, valid_df=df_val, test_df=df_test, path="")
        # Classifier model data
        self.clf_databunch = TextClasDataBunch.from_df(path="",
                                                       train_df=df_trn, valid_df=df_val, test_df=df_test,
                                                       vocab=self.lm_databunch.train_ds.vocab,
                                                       bs=32)

    def train(self):
        print("start train")
        self.estimator.train(self.lm_databunch)

    def evaluate(self):
        print("start eval")
        self.evaluation_results= self.estimator.evaluate()

    def predict(self, X=None, y=None):
        print("start predict")
        self.prediction_results = self.estimator.predict()

def run_evaluation_ulmfit(datasets_dir=evaluation.GCP_DATASETS_DIR,output_dir = evaluation.GCP_SESSIONS_DIR, suffix="_1"):
    datasets = evaluation.load_datasets_for_evaluation(dir=datasets_dir)
    for key,dataset in datasets.items():
        print(key)
        estimator = ULMFiTEstimator()
        session = ULMFiTSession(dataset,estimator,key+suffix)
        session.train()
        session.evaluate()
        #print(session.evaluation_results[2])
        session.predict()
        session.persist(output_dir=output_dir)
