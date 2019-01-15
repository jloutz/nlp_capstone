"""
cesspool of necessary evils...
one-shot hacks needed during implementaion. Kept here as a
tribute to the god of spaghetti-code
"""
import data_preparation as prep
import evaluation
from base import BaselineEstimator, Session
from run_classifier import InputExample, PaddingInputExample


## copy - constructor
## early version of data provider had bert-specific code.
## After refactoring (for use with ulmfit) I needed to get providers
## with the same datasets.
def data_provider_from_data_provider(provider):
    new_provider = prep.DataProvider()
    new_provider.x_train = provider.x_train
    new_provider.train_size = len(provider.x_train)
    new_provider.y_train = provider.y_train
    new_provider.labels = list(set(provider.y_train))
    new_provider.x_eval = provider.x_eval
    new_provider.eval_size = len(provider.x_eval)
    new_provider.y_eval = provider.y_eval
    new_provider.x_test = provider.x_test
    new_provider.test_size = len(provider.x_test)
    new_provider.y_test = provider.y_test
    new_provider.labels = provider.labels
    new_provider.data = provider.data
    new_provider.data_stats = provider.data_stats
    return new_provider

def reconstruct_datasets():
    new_ds = {}
    datasets = evaluation.load_datasets_for_evaluation()
    for key, dp in datasets.items():
        new_dp = data_provider_from_data_provider(dp)
        new_ds[key]=new_dp
    return new_ds


class UglyDataProvider(prep.DataProvider):
    """
    some ugly builder methods I needed to avoid
    generating more (expensive!) results with cloud TPUs...
    hopefully not needed anymore
    """
    @classmethod
    def provider_from_prepared_data(cls, x_train, y_train, x_eval, y_eval, x_test, y_test):
        """
        need this to rehydrate sessions for baseline using persisted bert sessions... unfortunately..
        :return: provider
        """
        provider = prep.DataProvider()
        provider.x_train = x_train
        provider.train_size = len(x_train)
        provider.y_train = y_train
        provider.labels = list(set(y_train))
        provider.x_eval = x_eval
        provider.eval_size = len(x_eval)
        provider.y_eval = y_eval
        provider.x_test = x_test
        provider.test_size = len(x_test)
        provider.y_test = y_test
        return provider

    @classmethod
    def provider_from_input_examples(cls, train_examples: [InputExample], eval_examples: [InputExample],
                                     test_examples: [InputExample]):
        """
        need this to rehydrate sessions for baseline using persisted bert sessions... unfortunately..
        :return: provider
        """
        xy_tup = [(ex.text_a, ex.label) for ex in train_examples if type(ex) is not PaddingInputExample]
        xy = list(zip(*xy_tup))
        x_train = xy[0]
        y_train = xy[1]

        xy_tup = [(ex.text_a, ex.label) for ex in eval_examples if type(ex) is not PaddingInputExample]
        xy = list(zip(*xy_tup))
        x_eval = xy[0]
        y_eval = xy[1]

        xy_tup = [(ex.text_a, ex.label) for ex in test_examples if type(ex) is not PaddingInputExample]
        xy = list(zip(*xy_tup))
        x_test = xy[0]
        y_test = xy[1]
        provider = cls.provider_from_prepared_data(x_train, y_train, x_eval, y_eval, x_test, y_test)
        # provider.train_examples=train_examples
        # provider.dev_examples=eval_examples
        # provider.test_examples=test_examples
        return provider


def run_baseline_with_providers(providers:[prep.DataProvider]):
    estimator = BaselineEstimator()
    sessions = []
    for provider in providers:
        session = Session(provider, estimator, "rehydrated")
        print(session)
        session.train()
        session.evaluate()
        session.predict()
        sessions.append(session)
    return sessions

def rehydrate_providers_from_bert_sessions():
    import glob
    import os
    from sklearn.externals import joblib
    bert_results = [(os.path.split(path)[1], joblib.load(path)) for path in glob.glob("C:/Projects/udacity-capstone/results/sessions/*Bert*")]
    providers = []
    for name,result in bert_results:
        provider = UglyDataProvider.provider_from_input_examples(
            result["train_examples"],
            result["eval_examples"],
            result["test_examples"]
        )
        providers.append(provider)
    return bert_results,providers

def run_baseline_with_bert_session_data():
    ### util function to do things exactly the opposite of the
    ### way I intended... o well...
    ### but seriously - I managed to get a handful of bert results persisted along with data.
    ### but I neglected to run the baseline with the same data, and I DO NOT want to do it again and pay for more cloud tpu/gpu time right now..
    #   In order to run baseline against the same data do
    ###     load bert session results, extract input_examples, make providers from those examples, make sessions from
    ###     those providers, and run baseline.
    ###     then compare original bert session results with baseline session results...
    bert_results,providers = rehydrate_providers_from_bert_sessions()
    baseline_sessions = run_baseline_with_providers(providers)
    # now compare bert results with baseline sessions
    for i,bert_result in enumerate(bert_results):
        bname = bert_result[0]
        bres = bert_result[1]
        baseline_session = baseline_sessions[i]
        print("Bert name")
        print(bname)
        print("Bert eval result: ",bres["evaluation_results"]["eval_accuracy"])
        print("Baseline name")
        print(baseline_session)
        print("Baseline results: ",baseline_session.evaluation_results[2])
