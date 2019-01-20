import glob
import re
from sklearn.externals import joblib
import pandas as pd
import os
import config

"""
once evaluations have been run and sessions with results persisted to a directory, use the 
Results class here to load those sessions and create dataframes and histograms to inspect and 
view the results of the evaluation
"""

class Results:
    ### encapsulates session loading and preparation of results for inspecton
    def __init__(self,sessions_dir=config.LOCAL_SESSIONS_DIR,session_names = config.DATASET_SAMPLE_NAMES):
        ## sessions dir is dir where sessions were persisted after running estimators
        self.session_names = session_names
        self.sessions_dir = sessions_dir
        self.sessions = self._load_sessions()
        self.res_df = self._results_df()

    def persist(self,path=config.RESULTS_DF_PATH):
        print("persisting results dataframe to: ",path)
        joblib.dump(self.res_df,path)
        print("Done!")


    def _load_sessions(self):
        ## load sessions from dir.
        ## unpickling might work best by unpickling from environment where sessions were persisted
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
        ## create a dataframe from session results
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
            if "prediction_results" in session:
                if session["estimator_type"] == "ulmfit":
                    ulm_preds = session["prediction_results"]
                    pred = ulm_preds[0]
                    true = ulm_preds[1]
                    correct = 0
                    for i in range(len(pred)):
                        if pred[i]==true[i]:
                            correct+=correct
                    pred_score =correct/len(pred)
                else:
                    pred_df = session["prediction_results"]
                    pred_score = len(pred_df[pred_df["true"]==pred_df["pred"]])/len(pred_df)
                datarow.append(pred_score)
            data.append(datarow)
        res_df = pd.DataFrame(data,row_index,cols)
        return res_df

    @classmethod
    def get_max_scores(cls,res_df):
        ## if more than one session of same dataset and estimator, return session with highest eval score
        base = res_df[res_df['est_type']=='baseline']
        bert = res_df[res_df['est_type'] == 'bert']
        ulmfit = res_df[res_df['est_type'] == 'ulmfit']
        baseres = base.groupby(("sess_name","est_type"),sort=False).max()
        bertres = bert.groupby(("sess_name", "est_type"), sort=False).max()
        ulmres = ulmfit.groupby(("sess_name", "est_type"), sort=False).max()
        return (baseres,bertres,ulmres)

    @classmethod
    def get_mean_scores(cls, res_df):
        ## if more than one session of same dataset and estimator, return average of eval scores
        base = res_df[res_df['est_type'] == 'baseline']
        bert = res_df[res_df['est_type'] == 'bert']
        ulmfit = res_df[res_df['est_type'] == 'ulmfit']
        baseres = base.groupby(("sess_name", "est_type"), sort=False).mean()
        bertres = bert.groupby(("sess_name", "est_type"), sort=False).mean()
        ulmres = ulmfit.groupby(("sess_name", "est_type"), sort=False).mean()
        return (baseres, bertres, ulmres)

    @classmethod
    def show_results_hist(cls,df,session_names,mean_or_max='max',with_ulm=False):
        import matplotlib.pyplot as plt
        import numpy as np
        scores = None
        if mean_or_max=='max':
            scores = cls.get_max_scores(df)
        elif mean_or_max=='mean':
            scores = cls.get_mean_scores(df)
        if scores is None:
            raise Exception("check min or max param")
        base_eval=scores[0].eval_score
        bert_eval=scores[1].eval_score
        if with_ulm:
            ulmfit_eval = scores[2].eval_score

        ind = np.arange(len(base_eval))  # the x locations for the groupsnp.arange(
        width = 0.25  # the width of the bars

        fig, ax = plt.subplots()
        if with_ulm:
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


    @classmethod
    def show_reduced_results_hist(cls,df,mean_or_max='max',
                                  session_names=('small-150','small-300','small-450','small-600','med-900','med-1500')):
        import matplotlib.pyplot as plt
        import numpy as np
        scores = None
        if mean_or_max=='max':
            scores = cls.get_max_scores(df)
        elif mean_or_max=='mean':
            scores = cls.get_mean_scores(df)
        if scores is None:
            raise Exception("check min or max param")
        base_eval=scores[0].eval_score[:6]
        bert_eval=scores[1].eval_score[:6]
        #ulmfit_eval = scores[2].eval_score

        ind = np.arange(len(base_eval))  # the x locations for the groupsnp.arange(
        width = 0.25  # the width of the bars

        fig, ax = plt.subplots()
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


def load_and_show_results(filepath=config.RESULTS_DF_PATH):
    ## load results dataframe and view histogram
    res_df = joblib.load(filepath)
    session_names = ("small-150", "small-300", "small-450", "small-600", "med-900",
                          "med-1500", "lrg-3000", "lrg-12k", "lrg-30k", "full",)

    Results.show_results_hist(res_df,session_names)

def results_table_from_results_df(filepath=config.RESULTS_DF_PATH,mean_or_max='max'):
    ## method for producing simple tabular results from results dataframe
    res_df = joblib.load(filepath)
    base = res_df[res_df['est_type'] == 'baseline']
    bert = res_df[res_df['est_type'] == 'bert']
    ulmfit = res_df[res_df['est_type'] == 'ulmfit']
    if mean_or_max=='max':
        baseres = base.groupby(("sess_name", "est_type"), sort=False).max()
        bertres = bert.groupby(("sess_name", "est_type"), sort=False).max()
        ulmres = ulmfit.groupby(("sess_name", "est_type"), sort=False).max()
    else:
        baseres = base.groupby(("sess_name", "est_type"), sort=False).mean()
        bertres = bert.groupby(("sess_name", "est_type"), sort=False).mean()
        ulmres = ulmfit.groupby(("sess_name", "est_type"), sort=False).mean()

    res_tbl = pd.DataFrame()
    sessnames = baseres.index.get_level_values('sess_name')
    res_tbl.index=sessnames
    res_tbl['BERT'] = bertres['eval_score'].values
    res_tbl['Baseline'] = baseres['eval_score'].values
    res_tbl['ULMFiT'] = ulmres['eval_score'].values
    return res_tbl



#res = Results(sessions_dir=config.LOCAL_RESULTS_DIR+"/small_sessions",session_names=config.DATASET_SAMPLE_NAMES[:6])