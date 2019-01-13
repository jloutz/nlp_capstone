# Machine Learning Engineer Nanodegree
## Capstone Project
John Loutzenhiser
January 6, 2019

## Evaluating Pretrained Language Models for Short Text Classification



## I. Definition
_(approx. 1-2 pages)_

### Project Overview
Text classification is a well-studied and widely used practical application of machine learning technologies ([Giorgino 2004][] ). In a text classification task, one starts with a corpus of documents or texts which have been categorized or labeled according to some criteria. This pre-categorized corpus is used as training data to train a machine learning model which is then able to predict the category or class of new, unseen documents. 

Examples of text classification tasks that I have recently been involved with are:

- automatic quality control of vocational course descriptions uploaded from vocational schools from all over Germany into a central course description database
  - Are the documents plausible as a vocational course description? (yes/no classification)
  - Which type of qualifications does a course description offer (multiclass, or one of many classification)
- automatic recognition of university enrollment certificates scanned and uploaded by citizens applying for government benefits.
  - is an uploaded document plausibly an authentic university enrollment certificate? (yes/no classification)

For these classification tasks, there were relatively large amounts of training data available - in the range of many thousands up to 1 million examples. With this amount of training data, it was relatively easy to achieve good prediction results quickly by training classifier models using the "Bag of Words" (BOW)  technique using a Support Vector Machine, Naive Bayes, or Logistic Regression algorithms. (see [Wang and Manning 2012][] for a discussion of these strong baseline text classifiers)   

However training robust text-classifiers that can generalize well and effectively classify a wide variety of input texts  is a challenge when large training datasets are not available, which is often the case when developing a chatbot from scratch, for example. In my experience, a lack of readily available domain-specific training data is a significant hurdle to bootstrapping chatbots and getting them robust enough for production without significant data-collection an annotation effort. 

It seems that when only small datasets are available, the Bag-of-Words approach might have some limitations. This is because Bag-of-words only considers the surface lexical aspects of a text (essentially the arrangement of letters into words) and no way of encoding lexical semantics (e.g. similar words), phrase structure (for example, negation "not a good book" means the opposite of "a very good book") or context. A toy dataset can demonstrate why this could be a limitation. Given the following labeled texts as training data:

| Text                                                  | label    |
| ----------------------------------------------------- | -------- |
| dogs should be walked every day                       | pet_care |
| indoor cats need activities to avoid boredom          | pet_care |
| brush your teeth twice a day to keep your teeth clean | hygiene  |

How would a BOW classifier classify the following? 

"hamsters need a clean cage"

Most likely "hygiene", as the word "clean" is shared and "hamster" is out-of-vocabulary for the training set. However, "pet_care" seems to be the correct class. 

Although it might appear as if BOW classifiers might have limitations on small datasets, classifiers such as those in [Wang and Manning 2012][] still represent very strong benchmarks for text classification. So the question is, are more modern, sophisticated methods more powerful than these simple, well known models on small datasets?

##### Pretrained Language Models

A recent advancement in NLP has been the development of pretrained _language_ models which can be used in transfer learning for a variety of NLP tasks including text classification. ([UlmFit][], [Elmo][], [BERT][])

These models represent not only lexical features (as is the case with Bag-of-Words) but also lexical and contextual semantics as well. Because of this, the potential impact of these language models for NLP is being compared to the impact pretrained image-recognition models have had for the field of computer vision ([NLPs Imagenet Moment has Arrived](https://thegradient.pub/nlp-imagenet/))

The promise of pretrained language models is that they make it possible to fine-tune pretrained models with relatively small training sets, creating classifiers that might generalize well by leveraging features and context they learned through pre-training on huge datasets.  

### Problem Statement

I would like to determine how pretrained language models fine-tuned on small datasets measure up for text classification on short texts. The focus on relatively small datasets and rather short texts should simulate the real-world problem of bootstrapping a dialog system/chatbot when large training datasets are unavailable.

Can pretrained language models result in classifiers that can generalize well when fine-tuned on small datasets? Can they:

- classify texts with features that were not explicitly seen in the (fine-tuning) training set? 
- "understand" out-of-vocabulary words?
- classify well taking polysemy and synonyms into account?

Can these classifiers perform better than well-known benchmark classifiers?

In order to answer these questions, an evaluation is proposed where two modern pretrained language model-based classifiers will be compared to a common benchmark text classifier. In order to perform this evaluation, the following tasks were undertaken:

*  Identify and acquire an  appropriate dataset for training and evaluation. The dataset should consist of short texts taken from a real-world conversational or "chat" scenario. Each text must have a label or class assigned so that it can be used for supervised learning 

* Using a supervised learning paradigm, train 3 text classifiers using training data in the dataset. The 3 classifiers are based on the following implementations:

  * tfidf weighted word-grams fed into a Naive Bayes Classifier. This implementation is based on scikit-learn, and represents a widely-used and strong baseline implementation for comparison
  * BERT - implementation based on the open-source release available at https://github.com/google-research/bert
  * ULMFiT - implementation based on the open source release found at https://github.com/fastai/fastai

* Each classifier is trained (or fine-tuned) on samples out of the total dataset of various sizes. The focus is on small samples, to simulate the situation of bootstrapping a chatbot from scratch, and to test the hypothesis that pretrained language models might indeed require less training (fine-tuning) than other classifiers to achieve similar or better results.   

* Evaluate the performance of each classifier on training or holdout data in each dataset sample using the accuracy metric (percent of correctly classified texts).

* Examine a confusion matrix of classification results for each dataset sample in order to determine to what extent pretrained language models were indeed able to "infer" the correct class in cases where training features were not present in the input sample (out-of-vocabulary)  


### Metrics
All models are evaluated on test data according to the accuracy metric. As **balanced datasets** are used in the training samples, the accuracy metric can be used to reliably predict the overall performance of the classifiers. 

Accuracy is defined by the following:
$$
\frac{tp+tn}{tp+tn+fp+fn}
$$
where:

- tp = true positives - number of instances with correctly predicted class label
- tn = true negatives  - for a particular class, number of instances correctly predicted as not belonging to that class
- fp = false positives - for a class, number of instances incorrectly predicted as belonging to that class
- fn = false negatives - for a particular class, number of instances actually belonging to the class which were incorrectly predicted as not belonging to that class

As this is a multiclass classification problem, a [confusion matrix](https://en.wikipedia.org/wiki/Confusion_matrix)  will also be useful to allow for more detailed analysis of tp, tn, fp, and fn.  



## II. Analysis
### Data Exploration

For this evaluation I used the [amazon question/answer dataset][]. ([Wan and McAuley 2016][]), ([McAuley and Yan 2016][]) This dataset contains around 1.4 million relatively short question/answer pairs taken from amazon product reviews. This dataset is an excellent dataset for this evaluation, because:

- they texts are short in a question/answer format -  similar to a single "turn" in a conversational chatbot
- they are real-world transcribed data, full of all of the wonderful noisy ambiguities and complexities of natural language

For the classification task, I have downloaded a total of 227106 question/answer pairs belonging to 10 different categories. The following is an example of one question/answer pair taken from the "baby" category

```
{'questionType': 'open-ended',
'asin': '177036417X',
'answerTime': 'Apr 16, 2015',
'unixTime': 1429167600,
'question': "Does this book contain any vaccination/immunization pages? Or pages about school? (Most do, yet I don't vax and I homeschool). Thx so much! :)",
'answer': 'Immunization page, yes. School, no.'}
```

As this is not a question answering evaluation but a simple text classification, I split the question/answer pairs into two separate short texts, for a total of 454212 short texts.  

The category names, the amount of texts per category, and the relative size of the category are:

| Category                   | Count | Percent |
| -------------------------- | ----- | ------- |
| appliances                 | 18022 | 3.97    |
| arts_crafts_and_sewing     | 42524 | 9.36    |
| baby                       | 57866 | 12.74   |
| beauty                     | 84844 | 18.68   |
| clothing_shoes_and_jewelry | 44136 | 9.72    |
| grocery_and_gourmet_food   | 39076 | 8.60    |
| musical_instruments        | 46644 | 10.27   |
| pet_supplies               | 73214 | 16.12   |
| software                   | 21272 | 4.68    |
| video_games                | 26614 | 5.86    |

 As we can see, the sizes of each category range from about ~4% to about 19% of the total dataset size. This unbalanced category distribution will be corrected in order to produce balanced classes. This step is particularly important, as the accuracy metric will be used to evaluate classifier performance. 

### Exploratory Visualization

Since this evaluation is concerned with short text classification, it is interesting to see just how short the texts in the total dataset actually are. The following histogram shows the distribution of text lengths:

![](C:\Projects\udacity-capstone\img\data_len.png) 

```
'doc_count': 454212, 'mean': 22.59106320396643, 
'mode': ModeResult(mode=array([7]), count=array([23396])),
'quantile_90': 47.0, 'max': 2660, 'min': 0, 'num_empty': 11
```

As we can see by the graphic and the stats above, the vast majority of the texts are somewhere between 1 and 100 words long, with an average length of 22.6 words. However there are small numbers of much longer texts, up to 2660 words long (not visible as bars in the histogram, but apparent on the scale of the x-axes). As well, there are 11 empty texts in the dataset. Because we are interested in short text classification, we can discard the small amount of very long texts. Additionally, empty texts should be discarded. 

The following shows the distribution of text lengths after discarding all texts longer than 47 words (the 90th percentile of all text lengths) and empty texts. 

![](C:\Projects\udacity-capstone\img\trimmed_data_len.png) 

 

```
'doc_count': 409768, 'mean': 15.86339343238125, 
'mode': ModeResult(mode=array([7]), count=array([23396])), 
'quantile_90': 31.0, 'max': 47, 'min': 1, 'num_empty': 0
```

After discarding those texts, ~90% of the dataset remains, and consists of more appropriate texts for a short text classification task. Discarding long and empty texts is one step taken in data preprocessing for this reason. 

The new distribution of text categories after discarding long and empty texts is:

| Category                   | Count | Percent |
| -------------------------- | ----- | ------- |
| appliances                 | 16349 | 3.99    |
| arts_crafts_and_sewing     | 38639 | 9.43    |
| baby                       | 52163 | 12.73   |
| beauty                     | 76992 | 18.79   |
| clothing_shoes_and_jewelry | 40775 | 9.95    |
| grocery_and_gourmet_food   | 35539 | 8.67    |
| musical_instruments        | 42084 | 10.27   |
| pet_supplies               | 64127 | 15.65   |
| software                   | 18441 | 4.50    |
| video_games                | 24659 | 6.02    |

The distribution did not change significantly compared to the dataset before trimming, so balancing the dataset is still a requirement. 

### Algorithms and Techniques

##### Transfer Learning

Transfer Learning is a technique for improving the performance of task-specific machine learning models by incorporating existing models pretrained on very large, general datasets into those task-specific models. During so-called *fine-tuning*, task-specific data pertaining to the specific problem to be solved is used to further train an existing model, adjusting or fine-tuning the model to better fit the task at hand. 

 Transfer learning has been used extensively in recent years in the field of computer vision and image classification ([Donahue et al. 2013]()). In this application, deep neural convolutional networks are trained on a large dataset, such as the popular [imagenet](http://www.image-net.org/). Through training, the layers in these networks come to represent a hierarchy of low-level (lines, edges) to higher-level (shapes, object contours, scenes) features of the images. The intuition of transfer learning is that these layers of learned feature representations can be re-used for a wide variety of novel tasks where the amount of training data might be insufficient for training a deep model from scratch.    

##### Language Models

Transfer learning in NLP (Natural Language Processing) works by the same principle as transfer learning in Computer Vision , but applied to language tasks instead of image-related tasks. First, a *language model* is trained on a large dataset of text which thereby learns representations of the features of the language/dataset which can then be re-used for specific, novel NLP tasks. The models evaluated here (BERT, ULMFiT) are examples of such pretrained models. 

A *language model* in the most exact sense refers to a model trained on the prediction task of *predicting the next word in a sequence of words*. This task is difficult because accurate prediction of the next word in a sequence must involve some representation of syntactic, semantic, and real-world knowledge as is demonstrated by the following sentences:

*Have you ever really fallen in ___*

*Have you ever fallen down the ___*

A distinguishing feature of the language models evaluated here as opposed to word embeddings such as [word2vec](Mikolov et. al 2013), [GloVe](Pennington et. al), or [Fasttext](Joulin et al. 2016), is that these models learn *context-sensitive representations* of words, whereas word embeddings will learn one embedding for each word in the training vocabulary. In contrast to word embeddings, these models can be used for all but the output layer in transfer learning, whereas word embeddings can only provide the first layer of a task-specific network in transfer learning.   

Learning context-sensitive representations presumably entails that these models are learning and representing abstract and higher-level features of language. The exact nature of these higher-level features must remain speculative, as they are represented in the weights of very large neural networks and as such, not transparently interpretable, however there is evidence that language models trained on word prediction tasks learn useful abstraction of language such as long-distance dependencies ([]()), hierarchical relations [](), and sentiment [](). TODO

The abstractions learned by these models are useful because they generalize well. Such models can be applied to a wide variety of language related tasks such as question-answering, entailment and sentiment analysis, grammaticality judgments and more, as evidenced by competitive performance of these models on difficult benchmarks such as [GLUE](). 

Language modeling of this type has the additional advantage of being an *unsupervised learning task*. This is fundamentally important, as in NLP, the amount of unlabeled training data is practically unlimited (raw text), whereas labeled training data for a specific task is usually hard to come by and involves significant human effort to collect and annotate. The ability to use freely available and abundant text for pretraining is key in the usefulness of pretrained language models 

##### ULMFiT

ULMFiT itself is less of a language model implementation as a general framework for transfer learning and fine-tuning language models. Indeed, the suggestion that word-prediction-from-context language models are the right task to train for NLP transfer learning is a proposal of the ULMFiT framework. The reference implementation uses the  [AWD LSTM language model](Merity et al. 2017) under the hood which has the following properties:

* implements innovations in regularization and hyperparameter tuning, but otherwise:
* is a "Vanilla" LSTM with no custom modifications to the LSTM architecture such as attention or short-cut connections
* It is a 3-layer architecture with 1150 units in the hidden layer and an embedding size of 400

ULMFiT claims to be a general framework in that the same language model and hyperparameter tune can be used for transfer learning in a wide variety of tasks.  

ULMFiT proposes a 3-Step process for training and fine-tuning

1. **LM pretraining** 

   The model is pretrained on a large corpus. The reference implementation is pretrained on the [Wikitext 103 dataset](Merity et al. 2017b)

2. **LM fine-tuning**

   During LM fine-tuning, the weights of all layers of the network are updated using task-specific training data. LM fine-tuning uses Discriminative fine-tuning and Slanted triangular learning rates (see below)

3. **Classification fine-tuning** (for classification) - for classification fine-tuning, an intermediate layer is added to the model with ReLU activation, and the classifier output layer is added with softmax activation outputting the probability distribution of predictions for each class in the target variable. 

ULMFiT proposes 3 innovations for fine-tuning:

1. **Discriminative fine-tuning** assigns a *unique, dedicated learning rate parameter to each individual layer in the model*. The reasoning is that because each layer encodes different information, they should be fined tuned differently or *discriminatively*. In contrast to normal Gradient Decent, where all model parameters at a time-step are updated at once, 
   $$
   \theta_t = \theta_{t-1}-\mu \nabla_{\theta_t} J(\theta)
   $$
   where  $\theta=parameters,$ $ \mu=learning rate$ and $ \nabla=gradient$

   discriminative fine-tuning splits the model parameters into one set per layer, and updates them according to their own learning rate as follows:
   $$
   \theta_t^l = \theta_{t-1}^l-\mu^l \nabla_{\theta_t} J(\theta)
   $$
   where *l* refers to each layer. 

2. **Slanted Triangular learning rates** - the authors want the learning at each layer to converge quickly to the approximate correct region in feature space, and then iteratively refine the parameters. To accomplish this, they propose a slanted triangular learning rate which increases rapidly, and then declines slowly.

    ![1547309151408](C:\Users\loutzenj\AppData\Roaming\Typora\typora-user-images\1547309151408.png) (taken from [Howard and Ruder 2018](), page 4)

3. **Gradual Unfreezing**   is a technique applied at the classification fine-tuning step. Rather than "unfreeze" or unlock all layers in the model at once during fine-tuning, making them all available for updating, the authors propose unfreezing the top layer, fine-tuning for one epoch, and then unfreezing the next layer, and repeating the process until convergence. Gradual unfreezing is claimed to help the model resist "catastrophic forgetting" - that is, forgetting much of what it has learned in steps 1 and 2 and making transfer learning pointless. The rational for starting at the top layer is that this layer contains the least general information, or in other words the information containing features which are nearest to the final classification step. 

These techniques are claimed to help ULMFiT generalize well to a broad range of specific tasks, while at the same time preserving as much pretrained knowledge as possible.   



##### BERT

BERT (Bidirectional Encoder Representations from Transformers) differs from other language model implementations by incorporating **bi-directional context**. The idea is intuitive: train a language model which can predict a word from context, but use context from both directions, both before and after the word:

*{She went to the} ____ {and bought groceries}*

While ULMFiT and other pretrained LMs can be trained forwards (predicting next word) and backwards (predicting previous word) independently and either concatenate forward and backward context features ([Peters et al 2018](Elmo) ) or average LM classifier results (ULMFiT), BERT is unique in that it learns bi-directional contextual representations at once. 

The technique used for bi-directional learning is called the **"masked language model"** or MLM. This technique is inspired by the [Cloze Test](https://en.wikipedia.org/wiki/Cloze_test) where a paragraph is presented with a proportion of words replaced by a blank, and subjects have to try to guess the missing words. In the MLM, a small percentage of words are hidden or "masked out" during training (as in the example sentence above) and the prediction task is to predict the masked tokens. The learned hidden vectors representing the masked tokens are feed to a softmax output layer to predict the output token (in the space of all tokens in the training set vocabulary). In the reference implementation, 15% of input tokens were masked out. 

BERT is based on a multi-layer bidirectional transformer encoder architecture ([Vaswani et al 2017]()) . The details of the underlying implementation can be taken from that paper, important here is to note that it is a fundamentally different architecture than ULMFiT. The Transformer contains no recurrence (as in LSTM) nor convolutions (as in CNNs), but uses rather a unique encoder-decoder with stacked attention architecture.

### Benchmark

There is no established benchmark for classifiers using this dataset that I am aware of, however to provide a baseline for the performance of the language models on this data, a baseline Bag-of-Words classifier is used. This baseline tokenizes input text creating normalized word-unigrams and bigrams which are then weighted with tf-idf. The tf-idf feature vectors are  used as input to a Multinomial naive Bayes classifier. This simple baseline represents a strong benchmark for short text classification, as already mentioned above and discussed in [Wang and Manning 2012](). 

## III. Methodology
### Data Preprocessing

The training data was downloaded from the amazon question/answer dataset site. The downloaded files were tarballs (*.tar.gz), one tarball per category. The extracted data consists of one *.json file for each category. These files consist of question-answer data and metadata, one line for each question/answer pair. 

```
{'questionType': 'open-ended',
'asin': '177036417X',
'answerTime': 'Apr 16, 2015',
'unixTime': 1429167600,
'question': "Does this book contain any vaccination/immunization pages? Or pages about school? (Most do, yet I don't vax and I homeschool). Thx so much! :)",
'answer': 'Immunization page, yes. School, no.'}
```

The first real step of preprocessing is to extract the question and answers from these text lines, discarding the other metadata. For each question/answer pair in each category, the pair is extracted and appended to an array of texts. This array represents all texts in a category, for example "baby". Additionally, an array of category names (labels) is built. 

```
{'labels': ['baby', '.....,
'texts': [['"Does this book contain any vaccination/immunization pages? Or pages about school? (Most do, yet I don't vax and I homeschool). Thx so much! :)","Immunization page, yes. School, no.", .....],[....]] 
```

The class `AmazonQADataLoader` takes over the above tasks of tarball extraction, sentence pair extraction, and label and text array construction. The method for extracting texts and building label array is:

```python
    def _extract_text_and_labels(self,lazy=True):
        ## parse data in json files preserving only labels (category name) and texts
        ## both questions and answers found in json are 
        ## treated as example texts for   each category
        
        ....        
        
        for category_file_name in os.listdir(self.json_dir):
            print("Loading text from: ", category_file_name)
            label = re.split("qa_(.*)\\.json", category_file_name)[1].lower()
            labels.append(label)
            category_texts = []
            with open(os.path.join(self.json_dir,category_file_name), "rt") as f:
                for line in f.readlines():
                    obj = ast.literal_eval(line)
                    category_texts.append(obj['question'])
                    category_texts.append(obj['answer'])
            texts.append(category_texts)
        print("Done!")
        return labels, texts
    ...
self.data = {"y": labels, "X": texts}   ## data loader "data" attribute 

```

The extracted text is saved in the data loader "data" attribute. This data is then used to initialize instances of *`DataProvider`* class as needed. The `DataProvider` works on the underlying total extracted dataset, and does the following:

* **normalizes text lengths** - as discussed above, long and empty texts are removed from data from which samples are later taken

  ```python
      def _normalize_lengths(self, data, quantile=90):
          ### remove empty texts and texts of len > quantile (percentile)
  
          def lengths(dat):
              ## get lengths
              lengths = []
              for texts in dat:
                  lengths.extend([len(text.split()) for text in texts])
              return lengths
  
          newdata = []
  
          def _check_len(text, max_incl):
              x = len(text.split())
              return x > 0 and x <= max_incl
  
          for texts in data:
              newtexts = [text for text in texts if _check_len(text, quantile_count)]
              newdata.append(newtexts)
  
          return newdata,orig_stats,new_stats
  
  ```


- **provides samples** of the data corresponding to the size of the passed-in train, eval, and test size parameters. This is important as this evaluation is carried out on samples of various sizes of the underlying dataset. As the focus of this evaluation is on fine-tuning pretrained models with small datasets, small samples were taken from the total dataset to create the train and test datasets.  The sample sizes used for the evaluation were:

| Sample name | training set size | evaluation set size | test set size |
| ----------- | ----------------- | ------------------- | ------------- |
| small-150   | 150               | 50                  | 100           |
| small-300   | 300               | 100                 | 100           |
| small-450   | 450               | 150                 | 100           |
| small-600   | 600               | 200                 | 100           |
| med-900     | 900               | 300                 | 100           |
| med-1500    | 1500              | 500                 | 100           |
| lrg-3000    | 3000              | 1000                | 100           |
| lrg-12k     | 12000             | 4000                | 100           |
| lrg-30k     | 30000             | 10000               | 100           |
| full        | ~150k             | ~50k                | 100           |



- ensures that the **provided samples are balanced** across the target classes. Balanced classes are important so as to not skew the accuracy score of the classifier.  Balanced classes have the effect of giving the classifiers no a-priori reason for preferring one class over the other. As there is much more data available in the dataset than needed for training/fine-tuning, creating balanced sample datasets is relatively easy. 

  ```python
      def _sample_from_data(self,data:dict,balance_classes = True):
          
  		... 
          
          ## if balance classes, downsample to smallest class size
          downsampleto = min([len(x) for x in X_])
  
          for (i,label) in enumerate(y_):
              vals = X_[i]
              class_sample_size = downsampleto if balance_classes else len(vals)
              y.extend([label for i in range(class_sample_size)])
              X.extend(vals[:class_sample_size])
         
          ....
  
          ## train test split with stratify true if balance classes
          dostrat = y if balance_classes else None
          self.x_train, self.x_eval, self.y_train, self.y_eval = \        					train_test_split(X,y,train_size=train_size,                                                        test_size=eval_size,stratify=dostrat)
  ```

- **Exposes the samples** in a format appropriate for the different classifiers in the evaluation. For example, BERT requires a list of Instances of `TrainingExample` objects, whereas the scikit-learn based baseline classifier requires two arrays, one "X" array of training texts, and a "y" array of the corresponding labels. 

### Implementation
A particular coding challenge for this evaluation was in analyzing and understanding the open source BERT code, and getting it to work with this dataset. Because of the complexity of this code, I decided to do some object oriented analysis and design as a method of understanding the code and incorporating it into this evaluation. 

The BERT code used as a starting point for the BERT implementation here is available at: 

https://github.com/google-research/bert

The script `run_classifier.py`provides the boilerplate for implementing a custom classifier with BERT.  This code was cloned and added as a source dependency to the project.  

The goal of the analysis and design was to produce a suite of classes, interfaces, and implementations that would provide the functionality of evaluating all three classifiers (baseline, ULMFiT, BERT) on the same set of dataset samples, and record the results. I wanted to do this in a way that provided a consistent API, was manageable and readable, and had a minimum of "spaghetti code". 

A simplified UML of the primary classes and collaborators is as follows:

![](C:\Projects\udacity-capstone\doc\oo.png) 

As already covered above, the `DataLoader` and `DataProvider` handles loading, extraction, sampling, balancing and length normalization. 

For the evaluation, one `DataProvider` was initialized for each sample size (see above). 

```python
## initialize data provider for dataset with 900 train, 300 eval, and 100 test samples
dp = DataProvider(data, "med-900",900,300,100)
```

The `Estimator` class encapsulates the configuration and logic for training, evaluating, and testing for each of the classifiers being evaluated. The train, evaluate, and test methods take data prepared by the `DataProvider` as parameters, and return evaluation or prediction results as needed. 

```python
## create an estimator for bert 
bert_estimator=BertEstimator(config)            
```

The `Session` class has a `DataProvider` and an `Estimator` as collaborators, and encapsulates the process of running an `Estimator` with a specific dataset sample prepared from the `DataProvider`, and then recording the results in its `evaluation_results` and `prediction_results` variables respectively. After a session is run, it can be persisted so that the data and the results can be retrieved for final evaluation.  

```python
bert_session = BertSession(dp, bert_estimator,name="bertsession")
bert_session.run()
## run runs train, evaluate, and predict. These can also be called separately
#bert_session.train()
#bert_session.evaluate()
#bert_session.predict()
print(bert_session.evaluation_results)
print(bert_session.prediction_results)
bert_session.persist()

```



##### A note on text preprocessing

The text data provided by the `DataProvider` class are not yet fully preprocessed for the estimators. Much text preprocessing happens "under the hood" and is initiated by the calls to train, evaluate, predict respectively. 

The baseline estimator uses the default tokenization and normalization provided by `scikit-learn` `TfIdfVectorizer` with the exception that unigrams and bigrams are used (`ngram_range(1,2)`). The relevant part of that configuration is reproduced here:

```python
  encoding='utf-8',
  strip_accents=None, lowercase=True,
  analyzer='word',
  stop_words=None, token_pattern=r"(?u)\b\w\w+\b"  
```

The BERT estimator uses the BERT-internal `FullTokenizer`. This tokenizer converts to unicode and then to lower case, removes punctuation and normalizes whitespace, just as the baseline does. It also segments the input into "word-pieces" with the BERT `WordpieceTokenizer`This tokenizer is unique to BERT and creates tokens based on word segments. The documentation for this is provided here:

```python
  def tokenize(self, text):
    """Tokenizes a piece of text into its word pieces.

    This uses a greedy longest-match-first algorithm to perform tokenization
    using the given vocabulary.

    For example:
      input = "unaffable"
      output = ["un", "##aff", "##able"]

    Args:
      text: A single token or whitespace separated tokens. This should have
        already been passed through `BasicTokenizer.

    Returns:
      A list of wordpiece tokens.
    """

```

TODO ULMFiT

##### Estimator Hyperparameters

The baseline estimator uses an alpha value of 1 for `MultinomialNB`. 

The hyperparameters for BERT are extensive. It was prohibitive to do an extensive parameter tune due to the time and expense of running BERT on cloud TPU machines, so the defaults recommended by the BERT team were used. The relevant hyperparameters used  are as shown below:

```python
class BertEstimatorConfig:

    def __init__(self,
                 bert_pretrained_dir,
                 output_dir ,
                 use_tpu = True,
                 tpu_name = None
                 ):
        ... 
        
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
        .... 
```

The BERT Base model was used for this evaluation

```python
BERT_BASE_MODEL = "gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12"
```

The BERT Base model utilizes a Transformer architecture with 12-layers of size 768 each, 12 attention heads, and 110 million trainable parameters. It is pretrained on Wikipedia text.  

TODO ULMFiT



### Refinement

Initially, the evaluation was run with the BERT large model. 

```python
BERT_LARGE_MODEL = "gs://cloud-tpu-checkpoints/bert/uncased_L-24_H-1024_A-16"
```

This model has a deeper architecture with 340 million trainable parameters as opposed to the base model's 110 million. 

It was found that running the large model often resulted in inconsistent evaluation and training results. Specifically, often the predicted class returned would be the same for all examples, resulting in an accuracy score of 10%, or exactly the score of a naive classifier. After extensive debugging, I was convinced this error was not in my code, but coming from BERT somewhere. Switching to the base model fixed the problem. 

## IV. Results

### Model Evaluation 



## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
### Reflection
### Improvement

### References

[Giorgino 2004]: https://pdfs.semanticscholar.org/99e1/3b7ac59b3b82956b26fb5fb964b2c69f4338.pdf	"An Introduction to Text Classification"
[Wang and Manning 2012]: https://www.aclweb.org/anthology/P12-2018	" "Baselines and Bigrams: Simple, Good Sentiment and Topic Classification""
[ULMFiT]: https://arxiv.org/abs/1801.06146
[Elmo]: https://allennlp.org/elmo
[Peters et al. 2018]: https://arxiv.org/abs/1802.05365	"Deep contextualized word representations"
[BERT]: https://github.com/google-research/bert
[Howard and Ruder 2018]: https://arxiv.org/pdf/1801.06146.pdf "Universal Language Model Fine-tuning for Text Classification"
[Devlin et al. 2018]: https://arxiv.org/pdf/1810.04805.pdf	"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
[NLPs-Imagenet-Moment]: https://thegradient.pub/nlp-imagenet/	"NLPs ImageNet Moment has Arrived"
[amazon question/answer Dataset]: http://jmcauley.ucsd.edu/data/amazon/qa/
[Wan and McAuley 2016]: http://cseweb.ucsd.edu/~jmcauley/pdfs/icdm16c.pdf	"Modeling ambiguity, subjectivity, and diverging viewpoints in opinion question answering systems"
[McAuley and Yang 2016]: http://cseweb.ucsd.edu/~jmcauley/pdfs/www16b.pdf	"Addressing complex and subjective product-related queries with customer reviews"
[Donahue et. al. 2013]: https://arxiv.org/abs/1310.1531	"DeCAF: A Deep Convolutional Activation Feature for Generic Visual Recognition"
[Mikolov et. al. 2013]: https://arxiv.org/abs/1301.3781 " Efficient Estimation of Word Representations in Vector Space"
[Pennington et. al]: https://nlp.stanford.edu/pubs/glove.pdf	"GloVe: Global Vectors for Word Representation"
[Joulin et al. 2016]: https://arxiv.org/pdf/1607.01759.pdf   "Bag of Tricks for Efficient Text Classification"
[GLUE]: https://gluebenchmark.com/tasks
[Merity et al. 2017]: https://arxiv.org/abs/1708.02182 "Regularizing and Optimizing LSTM Language Models "
[Merity et al. 2017b]: https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/
[Vaswani et al.]: https://arxiv.org/abs/1706.03762 "Attention is all you need"