import os

import model_evaluation.data_preparation as data_preparation
from model_evaluation.base import Estimator, Session
import pathlib

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
                 bert_pretrained_dir = "../data/uncased_L-12_H-768_A-12",
                 output_dir = "../output/amazonqa_output",
                 use_tpu = True,
                 tpu_name = ""
                 ):

        pretr_dir = pathlib.Path(bert_pretrained_dir)
        self.bert_config_file = pretr_dir / 'bert_config.json'
        self.vocab_file = pretr_dir / 'vocab.txt'
        "The output directory where the model checkpoints will be written."
        self.output_dir = output_dir
        "Initial checkpoint (usually from a pre-trained BERT model)."
        self.init_checkpoint = pretr_dir / 'bert_model.ckpt'
        self.do_lower_case = (str(pretr_dir).find("uncased")>-1)
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
        tf.logging.set_verbosity(tf.logging.INFO)

        tokenization.validate_case_matches_checkpoint(self.config.do_lower_case,
                                                      self.config.init_checkpoint)

        ## make output
        tf.gfile.MakeDirs(self.config.output_dir)
        ## tokenizer
        self.tokenizer = tokenization.FullTokenizer(
            vocab_file=self.config.vocab_file, do_lower_case=self.config.do_lower_case)


    def setup_estimator(self,num_train_examples,label_list):
        """
        setup tensorflow estimator to use
        and set as self.estimator
        """
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
        tf.logging.info("***** Running training *****")
        tf.logging.info("  Num examples = %d", len(X))
        tf.logging.info("  Batch size = %d", self.config.train_batch_size)
        tf.logging.info("  Num steps = %d", self.num_train_steps)
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

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(eval_examples), num_actual_eval_examples,
                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("  Batch size = %d", self.config.eval_batch_size)

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

        output_eval_file = os.path.join(self.config.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            tf.logging.info("***** Eval results *****")
            for key in sorted(result.keys()):
                tf.logging.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))

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

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                        len(predict_examples), num_actual_predict_examples,
                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("  Batch size = %d", self.config.predict_batch_size)

        predict_drop_remainder = True if self.config.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=self.config.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = self.estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(self.config.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                probabilities = prediction["probabilities"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == num_actual_predict_examples


class BertSession(Session):
    """
    subclass of session which provides InputExamples and label list
    to bert estimator
    """
    def train(self):
        X = self.data_provider.get_train_examples()
        y = self.data_provider.get_labels()
        self.estimator.train(X,y)

    def evaluate(self):
        X = self.data_provider.get_dev_examples()
        y = self.data_provider.get_labels()
        self.estimator.evaluate(X, y)

    def predict(self, X=None):
        if X is None:
            X = self.data_provider.get_test_examples()
        y = self.data_provider.get_labels()
        self.estimator.predict(X, y)


def run_bert():
    loader_conf = data_preparation.AmazonQADataLoaderConfig()
    loader = data_preparation.AmazonQADataLoader(conf=loader_conf)
    loader.load()
    config = BertEstimatorConfig(
        use_tpu=False
    )
    estimator = BertEstimator(config)
    very_small = BertSession(200, 200, 20, loader, estimator)
    small = BertSession(3000, 100, 20, loader, estimator)
    notso_small = BertSession(30000, 10000, 20, loader, estimator)
    full = BertSession(0.7, 0.3, 100, loader, estimator)
    for session in (very_small,small,notso_small,full):
        print(session)
        estimator.setup_estimator(len(session.data_provider.x_train),session.data_provider.get_labels())
        session.train()
        session.evaluate()
        session.predict()
