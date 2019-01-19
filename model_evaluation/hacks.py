"""
cesspool of necessary evils...
one-shot hacks needed during implementaion. Kept here as a
tribute to the god of spaghetti-code
"""
# from fastai.torch_core import

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
        new_ds[key] = new_dp
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


def run_baseline_with_providers(providers: [prep.DataProvider]):
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
    bert_results = [(os.path.split(path)[1], joblib.load(path)) for path in
                    glob.glob("C:/Projects/udacity-capstone/results/sessions/*Bert*")]
    providers = []
    for name, result in bert_results:
        provider = UglyDataProvider.provider_from_input_examples(
            result["train_examples"],
            result["eval_examples"],
            result["test_examples"]
        )
        providers.append(provider)
    return bert_results, providers


def run_baseline_with_bert_session_data():
    ### util function to do things exactly the opposite of the
    ### way I intended... o well...
    ### but seriously - I managed to get a handful of bert results persisted along with data.
    ### but I neglected to run the baseline with the same data, and I DO NOT want to do it again and pay for more cloud tpu/gpu time right now..
    #   In order to run baseline against the same data do
    ###     load bert session results, extract input_examples, make providers from those examples, make sessions from
    ###     those providers, and run baseline.
    ###     then compare original bert session results with baseline session results...
    bert_results, providers = rehydrate_providers_from_bert_sessions()
    baseline_sessions = run_baseline_with_providers(providers)
    # now compare bert results with baseline sessions
    for i, bert_result in enumerate(bert_results):
        bname = bert_result[0]
        bres = bert_result[1]
        baseline_session = baseline_sessions[i]
        print("Bert name")
        print(bname)
        print("Bert eval result: ", bres["evaluation_results"]["eval_accuracy"])
        print("Baseline name")
        print(baseline_session)
        print("Baseline results: ", baseline_session.evaluation_results[2])

### for dumb technical reasons that I didn't want to spend too much time to solve, I couldn't get the ULMFiT data structures
### unpickled in my local dev environment (after producing them in the cloud on GPU-machines).
### So I just copied the values here
def plot_lm_lr(skip_start: int = 10, skip_end: int = 5) -> None:
    ## plot ulmfit lrs
    lrs = [1e-07, 1.202264434617413e-07, 1.4454397707459274e-07, 1.7378008287493754e-07, 2.0892961308540395e-07,
           2.51188643150958e-07, 3.019951720402016e-07, 3.6307805477010137e-07, 4.36515832240166e-07,
           5.248074602497725e-07,
           6.309573444801933e-07, 7.585775750291837e-07, 9.120108393559096e-07, 1.096478196143185e-06,
           1.3182567385564074e-06, 1.5848931924611132e-06, 1.9054607179632473e-06, 2.2908676527677735e-06,
           2.754228703338166e-06, 3.311311214825911e-06, 3.981071705534973e-06, 4.7863009232263826e-06,
           5.754399373371569e-06, 6.918309709189365e-06, 8.317637711026708e-06, 9.999999999999999e-06,
           1.202264434617413e-05, 1.4454397707459279e-05, 1.737800828749376e-05, 2.0892961308540385e-05,
           2.5118864315095795e-05, 3.019951720402016e-05, 3.630780547701014e-05, 4.365158322401661e-05,
           5.248074602497728e-05, 6.309573444801929e-05, 7.585775750291836e-05, 9.120108393559096e-05,
           0.00010964781961431851, 0.00013182567385564074, 0.0001584893192461114, 0.00019054607179632462,
           0.00022908676527677726, 0.0002754228703338166, 0.0003311311214825911, 0.0003981071705534973,
           0.0004786300923226385, 0.0005754399373371565, 0.0006918309709189362, 0.0008317637711026709, 0.001,
           0.001202264434617413, 0.001445439770745928, 0.001737800828749376, 0.0020892961308540407,
           0.002511886431509582,
           0.0030199517204020187, 0.00363078054770101, 0.004365158322401656, 0.005248074602497722, 0.006309573444801929,
           0.007585775750291836, 0.009120108393559097, 0.01096478196143185, 0.013182567385564075, 0.01584893192461114,
           0.019054607179632484, 0.022908676527677745, 0.027542287033381692, 0.03311311214825908, 0.03981071705534969,
           0.0478630092322638, 0.05754399373371566, 0.06918309709189363, 0.08317637711026708, 0.09999999999999999,
           0.12022644346174131, 0.1445439770745928, 0.17378008287493762, 0.2089296130854041, 0.25118864315095824,
           0.3019951720402019, 0.36307805477010097, 0.43651583224016566, 0.5248074602497723, 0.630957344480193,
           0.7585775750291835, 0.9120108393559095, 1.096478196143185, 1.3182567385564072, 1.584893192461114,
           1.9054607179632481, 2.290867652767775, 2.754228703338169, 3.3113112148259076, 3.981071705534969,
           4.7863009232263805, 5.754399373371567, 6.918309709189362, 8.317637711026709]

    losses = [(5.9986), (5.8234), (5.7826), (5.7487), (5.7309), (5.7439),
              (5.7523), (5.7477), (5.7488), (5.7500), (5.7658), (5.7668), (5.7698),
              (5.7695), (5.7775), (5.7879), (5.7850), (5.7914), (5.7826),
              (5.7740), (5.7772), (5.7759), (5.7612), (5.7568), (
                  5.7589), (5.7638), (5.7659), (5.7591), (5.7555), (5.7639), (5.7730),
              (5.7672), (5.7739), (5.7704), (5.7683), (5.7673), (5.7676),
              (5.7656), (5.7615), (5.7620), (5.7613), (5.7591), (5.7586),
              (5.7586), (5.7561), (5.7589), (5.7556), (5.7601), (5.7610),
              (5.7501), (5.7503), (5.7464), (5.7423), (5.7343), (5.7315), (5.7270),
              (5.7240), (5.7136), (5.7031), (5.6887), (5.6790), (5.6632),
              (5.6480), (5.6292), (5.6040), (5.5848), (5.5578), (5.5303),
              (5.4995), (5.4701), (5.4431), (5.4048), (5.3691), (5.3271),
              (5.2842), (5.2402), (5.1920), (5.1470), (5.1034), (5.0614),
              (5.0191), (4.9757), (4.9379), (4.9059), (4.8787), (4.8556),
              (4.8376), (4.8339), (4.8369), (4.8566), (4.8934), (4.9506),
              (5.0495), (5.1914), (5.3980), (5.6893), (6.1329), (6.6997),
              (7.4941), (8.4637)]

    import matplotlib.pyplot as plt
    "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`."
    lrs = lrs[skip_start:-skip_end] if skip_end > 0 else lrs[skip_start:]
    losses = losses[skip_start:-skip_end] if skip_end > 0 else losses[skip_start:]
    _, ax = plt.subplots(1, 1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
    plt.show()


def plot_clf_lr(skip_start: int = 10, skip_end: int = 5) -> None:
    ## plot ulmfit lrs
    lrs = [1e-07, 1.202264434617413e-07, 1.4454397707459274e-07, 1.7378008287493754e-07, 2.0892961308540395e-07,
           2.51188643150958e-07, 3.019951720402016e-07, 3.6307805477010137e-07, 4.36515832240166e-07,
           5248074602497725e-07,
           6.309573444801933e-07, 7.585775750291837e-07, 9.120108393559096e-07, 1.096478196143185e-06,
           1.3182567385564074e-06,
           1.5848931924611132e-06, 1.9054607179632473e-06, 2.2908676527677735e-06, 2.754228703338166e-06,
           3.311311214825911e-06, 3.981071705534973e-06, 4.7863009232263826e-06, 5.754399373371569e-06,
           6.918309709189365e-06,
           8.317637711026708e-06, 9.999999999999999e-06, 1.202264434617413e-05, 1.4454397707459279e-05,
           1.737800828749376e-05,
           2.0892961308540385e-05, 2.5118864315095795e-05, 3.019951720402016e-05, 3.630780547701014e-05,
           4.365158322401661e-05, 5.248074602497728e-05, 6.309573444801929e-05, 7.585775750291836e-05,
           9.120108393559096e-05,
           0.00010964781961431851, 0.00013182567385564074, 0.0001584893192461114, 0.00019054607179632462,
           0.00022908676527677726, 0.0002754228703338166, 0.0003311311214825911, 0.0003981071705534973,
           0.0004786300923226385,
           0.0005754399373371565, 0.0006918309709189362, 0.0008317637711026709, 0.001, 0.001202264434617413,
           0.001445439770745928, 0.001737800828749376, 0.0020892961308540407, 0.002511886431509582,
           0.0030199517204020187,
           0.00363078054770101, 0.004365158322401656, 0.005248074602497722, 0.006309573444801929, 0.007585775750291836,
           0.009120108393559097, 0.01096478196143185, 0.013182567385564075, 0.01584893192461114, 0.019054607179632484,
           0.022908676527677745, 0.027542287033381692, 0.03311311214825908, 0.03981071705534969, 0.0478630092322638,
           0.05754399373371566, 0.06918309709189363, 0.08317637711026708, 0.09999999999999999, 0.12022644346174131,
           0.1445439770745928, 0.17378008287493762, 0.2089296130854041, 0.25118864315095824, 0.3019951720402019,
           0.36307805477010097, 0.43651583224016566, 0.5248074602497723, 0.630957344480193, 0.7585775750291835,
           0.9120108393559095, 1.096478196143185, 1.3182567385564072, 1.584893192461114, 1.9054607179632481,
           2.290867652767775, 2.754228703338169, 3.3113112148259076]

    losses = [(2.3284), (2.3777), (2.3801), (2.3562), (2.3695), (2.3662), (2.3359), (2.3698), (2.3703), (2.3687),
              (2.3487), (2.3384), (2.3362), (2.3389), (2.3534), (2.3451), (2.3463), (2.3433), (2.3505), (2.3534),
              (2.3541), (2.3527), (2.3499), (2.3551), (2.3546), (2.3504), (2.3537), (2.3550), (2.3514), (2.3519),
              (2.3514), (2.3502), (2.3554), (2.3537), (2.3521), (2.3543), (2.3550), (2.3546), (2.3606), (2.3617),
              (2.3630), (2.3649), (2.3656), (2.3592), (2.3600), (2.3599), (2.3672), (2.3651), (2.3681), (2.3683),
              (2.3678), (2.3649), (2.3637), (2.3599), (2.3584), (2.3636), (2.3692), (2.3688), (2.3726), (2.3731),
              (2.3703), (2.3710), (2.3717), (2.3696), (2.3744), (2.3804), (2.3801), (2.3762), (2.3784), (2.3765),
              (2.3802), (2.3788), (2.3777), (2.3736), (2.3787), (2.3766), (2.3832), (2.3846), (2.3892), (2.3880),
              (2.3913), (2.3868), (2.3901), (2.3984), (2.4461), (2.4755), (2.5129), (2.5545), (2.6769), (2.9804),
              (3.4491), (4.3639), (5.7965), (8.4509), (11.6205)]

    import matplotlib.pyplot as plt
    "Plot learning rate and losses, trimmed between `skip_start` and `skip_end`."
    lrs = lrs[skip_start:-skip_end] if skip_end > 0 else lrs[skip_start:]
    losses = losses[skip_start:-skip_end] if skip_end > 0 else losses[skip_start:]
    _, ax = plt.subplots(1, 1)
    ax.plot(lrs, losses)
    ax.set_ylabel("Loss")
    ax.set_xlabel("Learning Rate")
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.0e'))
    plt.show()

### couldn't unpickle ulmfit sessions..
# just making a data frame here from results copy-pasted from cloud environment.
def ulmfit_results():
    import pandas as pd
    cols = ("sess_name", "est_type", "ds_id", "eval_score", "pred_score")
    idc = [
        "small-150-ulmfit-8766",
        "small-300-ulmfit-4ef3",
        "small-450-ulmfit-b095",
        "small-600-ulmfit-c165",
        "med-900-ulmfit-1dca",
        "med-1500-ulmfit-3c7b",
        "lrg-3000-ulmfit-ob16",
        "lrg-12k-ulmfit-cc41",
        "lrg-30k-ulmfit-01af",
        "full-ulmfit-4c4f"
    ]
    data = [
        ['small-150','ulmfit','8766',0.16,0],
        ['small-300', 'ulmfit', '4ef3', 0.13, 0],
        ['small-450', 'ulmfit', 'b095', 0.1467, 0],
        ['small-600', 'ulmfit', 'c165', 0.21, 0],
        ['med-900', 'ulmfit', 'ldca', 0.263, 0],
        ['med-1500', 'ulmfit', '3c7b', 0.352, 0],
        ['lrg-3000', 'ulmfit', 'ob16', 0.46, 0],
        ['lrg-12k', 'ulmfit', 'cc41', 0.564, 0],
        ['lrg-30k', 'ulmfit', '01af', 0.638, 0],
        ['full', 'ulmfit', '4c4f', 0.703, 0],
    ]
    res = pd.DataFrame(data,idc,cols)
    return res









