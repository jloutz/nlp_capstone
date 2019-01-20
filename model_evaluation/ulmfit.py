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

"""
Code here depends on the fastai api. and pytorch.  
Add those dependencies to the project before using. 
see https://docs.fast.ai/install.html
I recommend a different venv for the ulmfit eval than that of the bert eval.  
"""

class ULMFiTEstimator(Estimator):
    ### estimator for ulmfit
    def __init__(self,epoch1=7,epoch2=7):
        ### epoch1 is for model fine-tuning,
        ### epoch2 is for classifier fine-tuning
        ### recommended is 7,7 for this evaluation.
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
        ## fine-tune language model and classifier
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
    ## Session gets data from data provider, massages it a bit, and runs train, eval, predict.
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




####################### run ulmfit evaluaton #######################################
import config
def _load_datasets_for_evaluation(dir=config.GCP_LOCAL_DATASETS_DIR,name="datasets_for_eval.pkl"):
    ## here using open and pickle to avoid tensorflow and sklearn dependencies.
    ## might not work with gcp bucket storage
    import os
    import pickle
    loadpath = os.path.join(dir,name)
    print("Loading {}...".format(loadpath))
    with open(loadpath,'rb') as f:
        datasets = pickle.load(loadpath)
    print("Done!")
    return datasets

def init_sessions(white_list='med-900'):
    ## method to get sessions without running eval if needed for debug/exploration
    sessions = []
    datasets = _load_datasets_for_evaluation()
    for key, dataset in datasets.items():
        if white_list is not None and not key in white_list:
            continue
        print(key)
        estimator = ULMFiTEstimator()
        session = ULMFiTSession(dataset, estimator, key)
        sessions.append(session)
    return sessions

######################## entry point for ulmfit eval #####################################################

def run_evaluation_ulmfit(datasets_dir=config.GCP_LOCAL_DATASETS_DIR,
                          datasets_name = "datasets_for_eval.pkl",
                          output_dir = config.LOCAL_SESSIONS_DIR,
                          suffix="_1",white_list=None,
                          epoch1=7,epoch2=7):
    ### epochs of 7,7 were found empirically to converge
    datasets = _load_datasets_for_evaluation(dir=datasets_dir,name=datasets_name)
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

