import uuid
from fastai.metrics import accuracy
from fastai.text import *
from fastai import *
from fastai.basic_train import Learner
import data_preparation as data_preparation
import evaluation
from base import Estimator, Session
import pandas as pd
import numpy as np


class ULMFiTEstimator(Estimator):
    def __init__(self,epoch1=1,epoch2=5):
        self.pretrained_model=URLs.WT103_1
        self.drop_mult=0.5
        self.lm_learner = None
        self.clf_learner = None
        self.epoch1=epoch1
        self.epoch2=epoch2
        self.id = uuid.uuid5(uuid.NAMESPACE_OID, str([self.pretrained_model,self.drop_mult])).hex
        pass

    def init_learners(self,lmdata,clfdata):
        self.lm_learn = language_model_learner(lmdata,
                                               pretrained_model=self.pretrained_model,
                                               drop_mult=self.drop_mult)
        self.clf_learn = text_classifier_learner(clfdata, drop_mult=self.drop_mult)


    def train(self, lmdata, clfdata):
        self.lm_learn = language_model_learner(lmdata,
                                               pretrained_model=self.pretrained_model,
                                               drop_mult=self.drop_mult)

        ## learn using discriminative training and gradual unfreezing
        numlayers = len(self.lm_learn.layer_groups)
        print("lm learner has {} layers ",numlayers)
        stop = 0-(numlayers)
        ## 1e-1 previously found as acceptable learning rate with learn.lr_find
        ## lr_range creates learning rates for each layer (discriminative training)
        lrr = self.lm_learn.lr_range(slice(1e-1, 1e-3))
        for i in range(self.epoch1):
            print("Training for Epoch ",i+1)
            self.lm_learn.freeze()
            ## now start layer unfreezing
            ##only last layer trainable
            self.lm_learn.fit_one_cycle(1,lrr)
            layer_num_inverted = -2 ## not second layer, but second from last
            ## train next layers
            while layer_num_inverted > -numlayers:
                print("freeze to ",layer_num_inverted)
                self.lm_learn.freeze_to(layer_num_inverted)
                self.lm_learn.fit_one_cycle(1, lrr)
                layer_num_inverted = layer_num_inverted-1


        print(self.lm_learn.predict("what size is ", n_words=5))
        self.lm_learn.save_encoder('ft_enc')

        ## now train classifier using gradual unfreezing and discriminative fine-tuning
        print("####### Fine-tuning classifier")
        self.clf_learn = text_classifier_learner(clfdata, drop_mult=self.drop_mult)
        numlayers = len(self.clf_learn.layer_groups)
        print("clf learner has {} layers ", numlayers)
        self.clf_learn.load_encoder('ft_enc')
        self.clf_learn.metrics = [accuracy]
        ## 1e-3 previously found as acceptable learning rate with learn.lr_find
        ## lr_range creates learning rates for each layer (discriminative training)
        lrr = self.clf_learn.lr_range(slice(1e-3, 1e-4))
        for i in range(self.epoch2):
            print("Training for Epoch ",i+1)
            self.clf_learn.freeze()
            ## now start layer unfreezing
            ##only last layer trainable
            self.clf_learn.fit_one_cycle(1, lrr)
            layer_num_inverted = -2  ## not second layer, but second from last
            ## train next layers
            while layer_num_inverted > -numlayers:
                print("freeze to ", layer_num_inverted)
                self.clf_learn.freeze_to(layer_num_inverted)
                self.clf_learn.fit_one_cycle(1, lrr)
                layer_num_inverted = layer_num_inverted - 1


    def evaluate(self, **kwargs):
        preds, targets = self.clf_learn.get_preds()
        predictions = np.argmax(preds, axis=1)
        #print((predictions,targets))
        return (predictions, targets)

    def predict(self, **kwargs):
        preds, targets = self.clf_learn.get_preds(ds_type=DatasetType.Test)
        predictions = np.argmax(preds, axis=1)
        return (predictions, targets)

    def __str__(self):
        return "ULMFiT_Estimator_{}".format(self.id)



class ULMFiTSession(Session):
    def __init__(self, data_provider: data_preparation.DataProvider, estimator: ULMFiTEstimator, name=""):
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
        estimator.init_learners(self.lm_databunch,self.clf_databunch)

    def train(self):
        print("start train")
        self.estimator.train(self.lm_databunch,self.clf_databunch)

    def evaluate(self):
        print("start eval")
        self.evaluation_results= self.estimator.evaluate()

    def predict(self, X=None, y=None):
        print("start predict")
        self.prediction_results = self.estimator.predict()

DATASETS_DIR = "/home/jloutz67/nlp_capstone/data/suites"
SESSIONS_DIR = "/home/jloutz67/nlp_capstone/results/sessions"

def init_sessions(white_list='med-900'):
    sessions = []
    datasets = evaluation.load_datasets_for_evaluation(dir=DATASETS_DIR)
    for key, dataset in datasets.items():
        if white_list is not None and not key in white_list:
            continue
        print(key)
        estimator = ULMFiTEstimator()
        session = ULMFiTSession(dataset, estimator, key)
        sessions.append(session)
    return sessions


def run_evaluation_ulmfit(datasets_dir=DATASETS_DIR,output_dir = SESSIONS_DIR, suffix="_1",white_list=None,
                          epoch1=7,epoch2=7):
    datasets = evaluation.load_datasets_for_evaluation(dir=datasets_dir)
    for key,dataset in datasets.items():
        if white_list is not None and not key in white_list:
            continue
        print(key)
        estimator = ULMFiTEstimator(epoch1=epoch1, epoch2=epoch2)
        session = ULMFiTSession(dataset,estimator,key+suffix)
        session.train()
        session.evaluate()
        #print(session.evaluation_results[2])
        session.predict()
        session.persist(output_dir=output_dir)

def ulmfit_results():
    import evaluation
    r = evaluation.Results()
    return r