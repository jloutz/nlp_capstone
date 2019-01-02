import numpy
import os

import pandas

import data_preparation as data_preparation
from base import Estimator, Session

import modeling
import tokenization
import tensorflow as tf

from run_classifier import file_based_input_fn_builder, file_based_convert_examples_to_features, PaddingInputExample, \
    model_fn_builder

"""
code here is largely an object-oriented rework of the code 
in bert-master run_classifier. I can't help it, it is how my brain works. 
"""

class BertEstimatorConfig:

    def __init__(self,
                 bert_pretrained_dir,
                 output_dir ,
                 use_tpu = True,
                 tpu_name = None
                 ):

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
    rework of bert-base run_classifier
    """
    def __init__(self,config:BertEstimatorConfig):
        """
        iniialize and setup/validate the estimator
        :param config: config params
        """
        self.config = config
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
            data.append(X[i].text_a)
            data.append(X[i].label)
            data.append(y[numpy.argsort(probs)[::-1][0]])
            data.extend(y)
            fulldata.append(data)

        cols = ["text","true","pred"]
        cols.extend(y)
        df = pandas.DataFrame(data=data,columns=cols)
        return df

    def __str__(self):
        return "Bert_Estimator_"+str(self.__hash__())



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

    def predict(self, X=None):
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


    def persist(self,output_dir="/nlpcapstone_bucket/results/"):
        import os
        import pickle
        import cloudstorage as gcs
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
        if self.prediction_results:
            obj["prediction_results"] = self.prediction_results

        with gcs.open(output_path) as f:
            print("Dumping a big fat pickle to {}...".format(output_path))
            pickle.dump(obj,f)
            print("Done!")


    def persist_name(self):
        persist_name = self.__str__()
        persist_name+= ".pkl"
        return persist_name


    def __str__(self):
        mystr = ""
        if self.name:
            mystr += self.name
        mystr += "-{}".format(str(self.estimator))
        if self.data_provider.x_train:
            s = "-train" + str(len(self.data_provider.x_train))
            mystr += s
        if self.data_provider.x_eval:
            s = "-eval" + str(len(self.data_provider.x_eval))
            mystr += s
        if self.data_provider.x_test:
            s = "-test" + str(len(self.data_provider.x_test))
            mystr += s
        return mystr


def setup_estimator_test():
    loader_conf = data_preparation.AmazonQADataLoaderConfig(data_preparation.LOCAL_PROJECT_DIR)
    loader = data_preparation.AmazonQADataLoader(conf=loader_conf)
    loader.load()
    config = BertEstimatorConfig(
        bert_pretrained_dir="C:/Projects/udacity-capstone/data/uncased_L-12_H-768_A-12",
        output_dir="C:/Projects/udacity-capstone/output/bert/",
        use_tpu=False,
        tpu_name=None
    )
    estimator = BertEstimator(config)
    provider = data_preparation.DataProvider(200, 200, 20,loader.data)
    session = BertSession(provider, estimator,name="setup_test")


def run_bert_local():
    loader_conf = data_preparation.AmazonQADataLoaderConfig("C:/Projects/udacity-capstone/")
    loader = data_preparation.AmazonQADataLoader(conf=loader_conf)
    loader.load()
    config = BertEstimatorConfig(
        bert_pretrained_dir="C:/Projects/udacity-capstone/data/uncased_L-12_H-768_A-12",
        output_dir="C:/Projects/udacity-capstone/output/bert/",
        use_tpu=False,
        tpu_name=None
    )
    estimator = BertEstimator(config)
    very_small_data = data_preparation.DataProvider(200, 200, 20, loader.data)
    small_data = data_preparation.DataProvider(3000, 100, 20, loader.data)
    notso_small_data = data_preparation.DataProvider(30000, 10000, 20, loader.data)
    full_data = data_preparation.DataProvider(0.7, 0.3, 100, loader.data)

    very_small_session = BertSession(very_small_data, estimator,name="vs")
    small_session = BertSession(small_data, estimator,name="s")
    notso_small_session = BertSession(notso_small_data, estimator,name="nss")
    full_session = BertSession(full_data, estimator,name="full")
    for session in (very_small_session,small_session,notso_small_session,full_session):
        print(session)
        session.train()
        session.evaluate()
        session.predict()


BERT_BASE_MODEL = "gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12"
BERT_LARGE_MODEL = "gs://cloud-tpu-checkpoints/bert/uncased_L-24_H-1024_A-16"

def run_bert_tpu():
    loader_conf = data_preparation.AmazonQADataLoaderConfig("/home/jloutz67/nlp_capstone")
    loader = data_preparation.AmazonQADataLoader(conf=loader_conf)
    loader.load()
    config = BertEstimatorConfig(
        bert_pretrained_dir=BERT_BASE_MODEL,
        output_dir="gs://nlpcapstone_bucket/output/bert/",
        tpu_name=os.environ["TPU_NAME"]
    )
    estimator=BertEstimator(config)
    very_small_data = data_preparation.DataProvider(500, 100, 20, loader.data)

    very_small = BertSession(very_small_data, estimator,name="very_small_bert")
    print(very_small)
    print()
    very_small.train()
    very_small.evaluate()
    very_small.predict()
    eval500_data = data_preparation.DataProvider(0, 500, 6, loader.data)
    eval_500_session = BertSession(eval500_data, estimator)
    print(eval_500_session)
    #print()
    eval_500_session.evaluate()
    eval_500_session.predict()
    return(very_small,eval_500_session)


if __name__=="__main__":
    #setup_estimator_test()
    #run_bert_local()
    run_bert_tpu()
