import uuid
import numpy
import os
import pandas
## model-evaluation dependencies
import config
import data_preparation as data_preparation
import evaluation
from base import Estimator, Session

## bert dependencies
import modeling
import tokenization
import tensorflow as tf
from run_classifier import InputExample,\
    file_based_input_fn_builder,\
    file_based_convert_examples_to_features,\
    PaddingInputExample, \
    model_fn_builder


"""
code here is largely an object-oriented rework of the code 
in bert-master run_classifier. I can't help it, it is how my brain works. 

particularly the really important code parts in train, evaluate, and predict are lifted
directly from the bert implementation in run_classifier

IMPORTANT: this code relies on the bert-master code available at 
https://github.com/google-research/bert
clone that repo and add it to the project source path (PYTHONPATH or sys.path) before trying this code.
"""

class BertEstimatorConfig:
    ### contains config and hyperparametes for bert estimator
    def __init__(self,
                 bert_pretrained_dir,
                 output_dir ,
                 use_tpu = True,
                 tpu_name = None
                 ):
        ### much of the hyperparamters and comments here taken directly from bert-master code
        self.bert_config_file = os.path.join(bert_pretrained_dir, 'bert_config.json')
        self.vocab_file = os.path.join(bert_pretrained_dir, 'vocab.txt')
        "The output directory where the model checkpoints will be written."
        self.output_dir = output_dir
        "Initial checkpoint (usually from a pre-trained BERT model)."
        self.init_checkpoint = os.path.join(bert_pretrained_dir, 'bert_model.ckpt')
        self.do_lower_case = (bert_pretrained_dir.find("uncased")>-1)
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded."
        self.max_seq_length = 128
        self.train_batch_size = 32
        self.eval_batch_size = 8
        self.predict_batch_size = 8
        self.learning_rate = 5e-5
        self.num_train_epochs = 3.0
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10% of training."
        self.warmup_proportion = 0.1
        "How often to save the model checkpoint."
        self.save_checkpoints_steps = 1000
        "How many steps to make in each estimator call."
        self.iterations_per_loop = 1000
        self.use_tpu = use_tpu
        "The Cloud TPU to use for training. This should be either the name "
        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
        "url."
        self.tpu_name = tpu_name
        "[Optional] GCE zone where the Cloud TPU is located in. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata."
        self.tpu_zone = None
        self.master = None##TODO
        "[Optional] Project name for the Cloud TPU-enabled project. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata."
        self.gcp_project = None
        "Only used if `use_tpu` is True. Total number of TPU cores to use."
        self.num_tpu_cores = 8


class BertEstimator(Estimator):
    """
    Estimator subclass for bert
    rework of bert-master run_classifier
    """
    def __init__(self,config:BertEstimatorConfig):
        """
        iniialize and setup/validate the estimator
        :param config: config params
        """
        self.config = config
        self.id = uuid.uuid5(uuid.NAMESPACE_OID,str(config.__dict__)).hex
        tf.logging.set_verbosity(tf.logging.ERROR)

        tokenization.validate_case_matches_checkpoint(self.config.do_lower_case,
                                                      self.config.init_checkpoint)

        ## tokenizer
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.config.vocab_file, do_lower_case=self.config.do_lower_case)


    def setup_estimator(self,num_train_examples,label_list):
        """
        setup tensorflow estimator to use
        and set as self.estimator
        """
        ## clean output
        if num_train_examples > 0 and tf.gfile.Exists(self.config.output_dir):
            tf.gfile.DeleteRecursively(self.config.output_dir)
        ## make output
        tf.gfile.MakeDirs(self.config.output_dir)

        bert_config = modeling.BertConfig.from_json_file(self.config.bert_config_file)

        if self.config.max_seq_length > bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (self.config.max_seq_length, bert_config.max_position_embeddings))

        tpu_cluster_resolver = None
        if self.config.use_tpu and self.config.tpu_name:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                self.config.tpu_name, zone=self.config.tpu_zone, project=self.config.gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=self.config.master,
            model_dir=self.config.output_dir,
            save_checkpoints_steps=self.config.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.config.iterations_per_loop,
                num_shards=self.config.num_tpu_cores,
                per_host_input_for_training=is_per_host))

        num_train_steps = int(
            num_train_examples / self.config.train_batch_size * self.config.num_train_epochs)
        num_warmup_steps = int(num_train_steps * self.config.warmup_proportion)

        model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list),
            init_checkpoint=self.config.init_checkpoint,
            learning_rate=self.config.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=self.config.use_tpu,
            use_one_hot_embeddings=self.config.use_tpu)

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=self.config.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=self.config.train_batch_size,
            eval_batch_size=self.config.eval_batch_size,
            predict_batch_size=self.config.predict_batch_size)
        self.estimator = estimator
        self.num_train_steps = num_train_steps

    def train(self, X, y):
        ## X is training examples, y is label list
        train_file = os.path.join(self.config.output_dir, "train.tf_record")
        file_based_convert_examples_to_features(
            X, y, self.config.max_seq_length, self.tokenizer, train_file)
        print("***** Running training *****")
        print("  Num examples = %d", len(X))
        print("  Batch size = %d", self.config.train_batch_size)
        print("  Num steps = %d", self.num_train_steps)
        train_input_fn = file_based_input_fn_builder(
            input_file=train_file,
            seq_length=self.config.max_seq_length,
            is_training=True,
            drop_remainder=True)
        self.estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)

    def evaluate(self, X, y):
        eval_examples = X
        num_actual_eval_examples = len(eval_examples)
        if self.config.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % self.config.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(self.config.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
            eval_examples, y, self.config.max_seq_length, self.tokenizer, eval_file)

        print("***** Running evaluation *****")
        print("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        print("  Batch size = %d", self.config.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if self.config.use_tpu:
            assert len(eval_examples) % self.config.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // self.config.eval_batch_size)

        eval_drop_remainder = True if self.config.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=self.config.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = self.estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
        return result


    def predict(self, X, y=None):
        predict_examples = X
        num_actual_predict_examples = len(predict_examples)
        if self.config.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % self.config.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(self.config.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, y,
                                                self.config.max_seq_length, self.tokenizer,
                                                predict_file)

        print("***** Running prediction*****")
        print("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        print("  Batch size = %d", self.config.predict_batch_size)

        predict_drop_remainder = True if self.config.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=self.config.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = self.estimator.predict(input_fn=predict_input_fn)
        fulldata = []
        for (i, prediction) in enumerate(result):
            if i >= num_actual_predict_examples:
                break
            probs = [prob for prob in prediction["probabilities"]]
            data = []
            data.append(X[i].label)
            data.append(y[numpy.argsort(probs)[::-1][0]])
            data.append(X[i].text_a)
            data.extend(probs)
            fulldata.append(data)

        cols = ["true","pred","text"]
        cols.extend(y)
        df = pandas.DataFrame(data=fulldata,columns=cols)
        return df

    def __str__(self):
        return "Bert_Estimator_{}".format(self.id)


def make_examples_fn(samples,set_type):
    ## samples should be list of tuples (text,label)
    examples = []
    for (i,sample) in enumerate(samples):
        guid = "%s-%s" % (set_type, i)
        text = tokenization.convert_to_unicode(sample[0])
        if set_type == "test":
           label = "0"
        else:
            label = tokenization.convert_to_unicode(sample[1])
        examples.append(
            InputExample(guid=guid, text_a=text, label=label))
    return examples

class BertSession(Session):
    """
    subclass of session which provides InputExamples and label list
    to bert estimator
    """
    def __init__(self, data_provider: data_preparation.DataProvider,
                 estimator: BertEstimator, name:str = "",):
        super().__init__(data_provider, estimator, name)
        if name:
            name = name.strip()
        # bert-specific callback for data provider
        data_provider.make_examples_fn = make_examples_fn
        #patch output path
        if name and os.path.split(estimator.config.output_dir)[1] != name:
            estimator.config.output_dir = os.path.join(estimator.config.output_dir, name)
        num_train_examples = 0 if self.data_provider.x_train is None else len(self.data_provider.x_train)
        self.estimator.setup_estimator(num_train_examples, self.data_provider.get_labels())

    def train(self):
        X = self.data_provider.get_train_examples()
        y = self.data_provider.get_labels()
        if X is None:
            print("train called although no training data exists in provider (was train_size 0?)")
            return
        self.estimator.train(X,y)

    def evaluate(self):
        X = self.data_provider.get_dev_examples()
        y = self.data_provider.get_labels()
        if X is None:
            print("evaluate called although no eval data exists in provider (was eval_size 0?)")
            return
        self.evaluation_results = self.estimator.evaluate(X, y)

    def predict(self, X=None, y=None):
        if X is None:
            X = self.data_provider.get_test_examples()
            if X is None:
                print("test called although no test data exists in provider (was test_size 0?)")
                return
        y = self.data_provider.get_labels()
        self.prediction_results = self.estimator.predict(X, y)


    def show(self):
        print()
        if self.evaluation_results:
            print("Evaluation results:")
            print(self.evaluation_results)
        if self.prediction_results:
            print("Predictions: ")
            print(self.prediction_results)


    def persist(self,output_dir="gs://nlpcapstone_bucket/sessions/"):
        ## persisting here makes use of tensorflow gfile, which can interface smoothly with
        ## google cloud platform bucket storage
        import os
        import pickle
        tf.gfile.MakeDirs(output_dir)
        output_path = os.path.join(output_dir,self.persist_name())
        obj = {}
        if self.data_provider.train_examples:
            obj["train_examples"]=self.data_provider.train_examples
        if self.data_provider.dev_examples:
            obj["eval_examples"] = self.data_provider.dev_examples
        if self.evaluation_results:
            obj["evaluation_results"] = self.evaluation_results
        if self.data_provider.test_examples:
            obj["test_examples"] = self.data_provider.test_examples
        if self.prediction_results is not None:
            obj["prediction_results"] = self.prediction_results

        with tf.gfile.GFile(output_path, "w") as f:
            print("Dumping a big fat pickle to {}...".format(output_path))
            pickle.dump(obj,f)
            print("Done!")

    def persist_name(self):
        persist_name = self.__str__()
        persist_name+= ".pkl"
        return persist_name

    def __str__(self):
        return super().__str__()


BERT_BASE_MODEL = "gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12"
BERT_LARGE_MODEL = "gs://cloud-tpu-checkpoints/bert/uncased_L-24_H-1024_A-16"

def setup_estimator_test():
    ## just a test if a session can be created
    loader_conf = data_preparation.AmazonQADataLoaderConfig(data_preparation.LOCAL_PROJECT_DIR)
    loader = data_preparation.AmazonQADataLoader(conf=loader_conf)
    loader.load()
    config = BertEstimatorConfig(
        ## path to pretrained model here
        bert_pretrained_dir=BERT_BASE_MODEL,
        output_dir="C:/Projects/udacity-capstone/output/bert/",
        use_tpu=False,
        tpu_name=None
    )
    estimator = BertEstimator(config)
    provider = data_preparation.DataProvider(200, 200, 20,loader.data)
    session = BertSession(provider, estimator,name="setup_test")


################## run bert evaluation ###########################
def load_datasets_for_evaluation(dir=config.GCP_DATASETS_DIR,name="datasets_for_eval.pkl"):
    import os
    from sklearn.externals import joblib
    loadpath = os.path.join(dir, name)
    print("Loading {}...".format(loadpath))
    datasets = joblib.load(loadpath)
    print("Done!")
    return datasets


################# entry point for bert evaluation ################
def run_evaluation_bert(datasets_dir=config.GCP_DATASETS_DIR,
                        dataset_name="datasets_for_eval.pkl",
                        output_dir = config.GCP_SESSIONS_DIR,
                        suffix="_1", white_list=None):
    ## call this method to run an evaluation on a dataset using bert estimator
    ## datasets_dir is the directory where the datasets prepared with data_preparation.prepare_datasets_for_eval
    ## are saved. output_dir is dir where sessions (results) are saved
    ## datasets_name is name of dataset pkl
    ## suffix will be appended to session name - good for multiple runs with same dataset to avoid name collision.
    ## white_list names of datasets ex. ['small-450','med-1500] to run. if none, runs all.
    datasets = _load_datasets_for_evaluation(dir=datasets_dir,name=dataset_name)
    for key,dataset in datasets.items():
        if white_list is not None and not key in white_list:
            continue
        print("*********** START "+key+suffix)
        config = BertEstimatorConfig(
            bert_pretrained_dir=BERT_BASE_MODEL,
            ## output_dir is bert-internal output, not session results output
            output_dir="gs://nlpcapstone_bucket/output/bert/",
            tpu_name=os.environ["TPU_NAME"]
        )
        estimator = BertEstimator(config)
        session = BertSession(dataset,estimator,key+suffix)
        session.train()
        session.evaluate()
        print(session.evaluation_results)
        session.predict()
        session.persist(output_dir=output_dir)


#### check for words in training data
def check_training_data():
    ### see report. some examples are examined if bert might really be using transfer learning.
    ##  'high information' words are checked if they are present in training data for a few select examples
    ##  see report for details
    res = evaluation.Results()
    bert_450_sess = [sess for sess in res.sessions if sess['name'].find('small-450-bert') > -1]
    bert_450 = bert_450_sess[1]
    test_texts =["Guinea pig","gobs ruffles","under armours fly open"]
    for text in test_texts:
        for example in bert_450['train_examples']:
            for word in text.split():
                if word in example.text_a.split():
                    print(word, " ", example.label)

