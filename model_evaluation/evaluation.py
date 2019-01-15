import glob
import re
from sklearn.externals import joblib
import pandas as pd
import os

LOCAL_SESSIONS_DIR = "C:/Projects/udacity-capstone/results/sessions"
GCP_SESSIONS_DIR = "gs://nlpcapstone_bucket/sessions"

LOCAL_DATASETS_DIR = "C:/Projects/udacity-capstone/data/suites"
GCP_DATASETS_DIR = "gs://nlpcapstone_bucket/suites"


class Results:
    def __init__(self):
        self.session_names = ("small-150", "small-300", "small-450", "small-600", "med-900",
                              "med-1500", "lrg-3000", "lrg-12k", "lrg-30k", "full",)
        self.sessions = self._load_sessions()
        self.res_df = self._results_df()

    def _load_sessions(self):
        from numpy.core import multiarray
        sessions = []
        for sessname in self.session_names:
            def load_session_fn(filename, session_name):
                estimator_type = ("baseline" if filename.find("Baseline_Estimator") > -1 else "bert" if filename.find(
                    "Bert_Estimator") > -1 else "")
                dataset_id = re.findall(".*DataProvider-([a-zA-z0-9]+)-.*", filename)[0][-4:]
                short_name="-".join((session_name, estimator_type, dataset_id))
                print("Loading with shortname: ",short_name)
                dict = joblib.load(filename)
                dict["name"]=short_name
                dict["session_name"]=session_name
                dict["estimator_type"]=estimator_type
                dict["dataset_id"]=dataset_id
                return dict
            sessions.extend([load_session_fn(fname,sessname) for fname in glob.glob(LOCAL_SESSIONS_DIR + "/*" + sessname + "*.pkl")])
        return sessions

    def _results_df(self):
        row_index = []
        data = []
        cols = ("sess_name","est_type","ds_id","eval_score","pred_score")
        for session in self.sessions:
            row_index.append(session["name"])
            datarow = [session["session_name"],session["estimator_type"],session["dataset_id"]]
            if session["estimator_type"]=="baseline":
                eval_score = session["evaluation_results"][2]
            else:
                eval_score = session["evaluation_results"]["eval_accuracy"]
            datarow.append(eval_score)
            pred_df = session["prediction_results"]
            pred_score = len(pred_df[pred_df["true"]==pred_df["pred"]])/len(pred_df)
            datarow.append(pred_score)
            data.append(datarow)
        res_df = pd.DataFrame(data,row_index,cols)
        return res_df

    def get_mean_scores(self):
        base = self.res_df[self.res_df['est_type']=='baseline']
        bert = self.res_df[self.res_df['est_type'] == 'bert']
        baseres = base.groupby(("sess_name","est_type"),sort=False).mean()
        bertres = bert.groupby(("sess_name", "est_type"), sort=False).mean()
        return (baseres,bertres)

    def show_results_hist(self,norm=False):
        import matplotlib.pyplot as plt
        import numpy as np
        scores = self.get_mean_scores()
        base_eval=scores[0].eval_score
        bert_eval=scores[1].eval_score

        if norm:
            max = np.max(bert_eval)
            base_eval = base_eval/max
            bert_eval = bert_eval/max


        ind = np.arange(len(base_eval))  # the x locations for the groups
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind - width / 2, base_eval, width,
                        color='SkyBlue', label='Baseline')
        rects2 = ax.bar(ind + width / 2, bert_eval, width,
                        color='IndianRed', label='BERT')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Scores by Dataset and Estimator')
        ax.set_xticks(ind)
        ax.set_xticklabels(self.session_names)
        ax.legend()

        plt.show()


from data_preparation import DataProvider
def load_datasets_for_evaluation(dir=LOCAL_DATASETS_DIR,name="datasets_for_eval.pkl"):
    import tensorflow as tf
    import pickle
    loadpath = os.path.join(dir,name)

    print("Loading {}...".format(loadpath))
    #with tf.gfile.GFile(loadpath,'rb') as f:
     #   datasets = pickle.load(f)
    datasets = joblib.load(loadpath)
    print("Done!")
    return datasets





