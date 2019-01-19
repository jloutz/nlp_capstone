import glob
import re
from sklearn.externals import joblib
import pandas as pd
import os

LOCAL_SESSIONS_DIR = "C:/Projects/udacity-capstone/results/sessions"
LOCAL_ULMFIT_SESSIONS_DIR = "C:/Projects/udacity-capstone/results/ulmfit_sessions"

GCP_SESSIONS_DIR = "gs://nlpcapstone_bucket/sessions"
GCP_LOCAL_SESSIONS_DIR = "/home/jloutz67/nlp_capstone/results/sessions"


LOCAL_DATASETS_DIR = "C:/Projects/udacity-capstone/data/suites"
GCP_DATASETS_DIR = "gs://nlpcapstone_bucket/suites"


class Results:
    def __init__(self,sessions_dir=LOCAL_SESSIONS_DIR):
        self.session_names = ("small-150", "small-300", "small-450", "small-600", "med-900",
                              "med-1500", "lrg-3000", "lrg-12k", "lrg-30k", "full",)
        self.sessions_dir = sessions_dir
        self.sessions = self._load_sessions()
        self.res_df = self._results_df()

    def _load_sessions(self):
        #from numpy.core import multiarray
        #from fastai import torch_core
        sessions = []
        for sessname in self.session_names:
            def load_session_fn(filename, session_name):
                estimator_type = ("baseline" if filename.find("Baseline_Estimator") > -1 else "bert" if filename.find(
                    "Bert_Estimator") > -1 else "ulmfit" if filename.find("ULMFiT")> -1 else "")
                dataset_id = re.findall(".*DataProvider-([a-zA-z0-9]+)-.*", filename)[0][-4:]
                short_name="-".join((session_name, estimator_type, dataset_id))
                print("Loading with shortname: ",short_name)
                dict = joblib.load(filename)
                dict["name"]=short_name
                dict["session_name"]=session_name
                dict["estimator_type"]=estimator_type
                dict["dataset_id"]=dataset_id
                return dict
            sessions.extend([load_session_fn(fname,sessname) for fname in glob.glob(self.sessions_dir+ "/*" + sessname + "*.pkl")])
        return sessions

    def _results_df(self):
        row_index = []
        data = []
        cols = ("sess_name","est_type","ds_id","eval_score","pred_score")
        for session in self.sessions:
            row_index.append(session["name"])
            datarow = [session["session_name"],session["estimator_type"],session["dataset_id"]]
            eval_score = -1
            if session["estimator_type"]=="baseline":
                eval_score = session["evaluation_results"][2]
            elif session["estimator_type"]=="bert":
                eval_score = session["evaluation_results"]["eval_accuracy"]
            elif session["estimator_type"] == "ulmfit":
                pred, true = session["evaluation_results"]
                eval_score =  len([pred[i] for i in range(len(pred)) if pred[i]==true[i]])/len(pred)
            datarow.append(eval_score)
            pred_df = session["prediction_results"]
            if session["estimator_type"] == "ulmfit":
                pred_score =0
            else:
                pred_score = len(pred_df[pred_df["true"]==pred_df["pred"]])/len(pred_df)
            datarow.append(pred_score)
            data.append(datarow)
        res_df = pd.DataFrame(data,row_index,cols)
        return res_df

    @classmethod
    def get_max_scores(cls,res_df):
        base = res_df[res_df['est_type']=='baseline']
        bert = res_df[res_df['est_type'] == 'bert']
        ulmfit = res_df[res_df['est_type'] == 'ulmfit']
        baseres = base.groupby(("sess_name","est_type"),sort=False).max()
        bertres = bert.groupby(("sess_name", "est_type"), sort=False).max()
        ulmres = ulmfit.groupby(("sess_name", "est_type"), sort=False).max()
        return (baseres,bertres,ulmres)

    @classmethod
    def show_results_hist(cls,df,session_names):
        import matplotlib.pyplot as plt
        import numpy as np
        scores = cls.get_max_scores(df)
        base_eval=scores[0].eval_score
        bert_eval=scores[1].eval_score
        ulmfit_eval = scores[2].eval_score

        ind = np.arange(len(base_eval))  # the x locations for the groupsnp.arange(
        width = 0.25  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind - width, ulmfit_eval, width,
                        color='lightslategray', label='ULMFiT')

        rects3 = ax.bar(ind + width, bert_eval, width,
                        color='steelblue', label='BERT')
        rects2 = ax.bar(ind, base_eval, width,
                        color='IndianRed', label='Baseline')

        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_ylabel('Accuracy')
        ax.set_title('Accuracy Scores by Dataset and Estimator')
        ax.set_xticks(ind)
        ax.set_xticklabels(session_names,rotation=45)

        ax.legend()

        plt.show()


from data_preparation import DataProvider
def load_datasets_for_evaluation(dir=LOCAL_DATASETS_DIR,name="datasets_for_eval.pkl"):
    loadpath = os.path.join(dir,name)
    print("Loading {}...".format(loadpath))
    datasets = joblib.load(loadpath)
    print("Done!")
    return datasets


def load_and_show_results(filepath):
    res_df = joblib.load(filepath)
    session_names = ("small-150", "small-300", "small-450", "small-600", "med-900",
                          "med-1500", "lrg-3000", "lrg-12k", "lrg-30k", "full",)

    Results.show_results_hist(res_df,session_names)





