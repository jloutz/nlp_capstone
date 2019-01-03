import os
import uuid

import data_preparation
from sklearn.externals import joblib

from base import BaselineEstimator, Session
from bert import BertEstimatorConfig, BERT_BASE_MODEL, BERT_LARGE_MODEL, BertEstimator, BertSession


def loader_fn(basedir):
    loader_conf = data_preparation.AmazonQADataLoaderConfig(basedir)
    loader = data_preparation.AmazonQADataLoader(conf=loader_conf)
    loader.load()
    return loader

def data_suite(loader):
    suite = {
        "mini":data_preparation.DataProvider(loader.data, 100, 500, 50),
        "very_small":data_preparation.DataProvider(loader.data, 200, 500, 50),
        "small":data_preparation.DataProvider(loader.data, 500, 500, 50),
        "med":data_preparation.DataProvider(loader.data, 1000, 500, 50),
        "large":data_preparation.DataProvider(loader.data, 5000, 500, 50),
        "larger":data_preparation.DataProvider(loader.data, 50000, 500, 50)
    }
    return suite

def generate_and_persist_datasuite():
    loader = loader_fn(data_preparation.LOCAL_PROJECT_DIR)
    suite = data_suite(loader)
    dumppath = os.path.join(LOCAL_SUITES_DIR,"data_suite_{}.pkl".format(uuid.uuid4().hex))
    joblib.dump(suite,dumppath)


LOCAL_SUITES_DIR = "C:/Projects/udacity-capstone/data/suites"
GCP_SUITES_DIR = "gs://nlpcapstone_bucket/suites"

LOCAL_SESSIONS_DIR = "C:/Projects/udacity-capstone/results/sessions"
GCP_SESSIONS_DIR = "gs://nlpcapstone_bucket/sessions"


def persist_sessions(sessions,output_dir):
    print("Persisting sessions..")
    for session in sessions:
        session.persist(output_dir)
    print("Done!")


def run_suite_with_baseline(filename):
    suite_path = os.path.join(LOCAL_SUITES_DIR,filename)
    suite = suite = joblib.load(suite_path)
    estimator = BaselineEstimator()
    sessions = []
    for (name,data) in suite.items():
        session = Session(data, estimator, name)
        print(session)
        session.train()
        session.evaluate()
        session.predict()
        sessions.append(session)
    return sessions

def run_suite_with_bert(filename):
    import pickle
    import tensorflow as tf
    suite_path = os.path.join(GCP_SUITES_DIR,filename)
    with tf.gfile.GFile(suite_path,'rb') as f:
        suite = pickle.load(f)
    config = BertEstimatorConfig(
        bert_pretrained_dir=BERT_LARGE_MODEL,
        output_dir="gs://nlpcapstone_bucket/output/bert/",
        tpu_name=os.environ["TPU_NAME"]
    )
    estimator = BertEstimator(config)

    sessions = []
    for (name,data) in suite.items():
        session = BertSession(data, estimator, name)
        print(session)
        session.train()
        session.evaluate()
        session.predict()
        sessions.append(session)
    return sessions


