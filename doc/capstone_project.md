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

The promise of pretrained language models is that they make it possible to fine-tune pretrained models with relatively small training sets, creating classifiers that might generalize well by leveraging lexical semantics and context they learned through pre-training on huge datasets.  

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

  * tfidf word-bigrams fed into a Naive Bayes Classifier. This implementation is based on scikit-learn, and represents a widely-used and strong baseline implementation for comparison
  * BERT (TODO)
  * ULMFiT (TODO)

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
_(approx. 2-4 pages)_

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

 As we can see, the sizes of each category range from about ~4% to about 19% of the total dataset size. This unbalanced category distribution had to be corrected in order to produce balanced classes. This step is particularly important, as the accuracy metric will be used to evaluate classifier performance. 

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

The distribution did not change significantly compared to the dataset before trimming.



### Algorithms and Techniques

### Benchmark

## III. Methodology
_(approx. 3-5 pages)_

### Data Preprocessing

After preprocessing, the example above is reduced to two short texts in an array of texts corresponding to the "baby" category. Much of the meta-data in the original format will be discarded during preprocessing, as I am only interested in the the labels and texts.

```
{'labels': ['baby', 'beauty'.....,
'texts': [['"Does this book contain any vaccination/immunization pages? Or pages about school? (Most do, yet I don't vax and I homeschool). Thx so much! :)","Immunization page, yes. School, no.", .....],[....]] 
```

As the focus of this evaluation is on fine-tuning pretrained models with small datasets, small samples will be taken from the total dataset to create the train and test datasets.  

These samples will be taken in such a way that the data is ***balanced across the 10 classes***. Balanced classes have the effect of giving the classifiers no a-priori reason for preferring one class over the other. As there is much more data available in the dataset than needed for training/fine-tuning, creating balanced sample datasets will be easy. 

In order to understand the relative effect of increasing the training data size on the target model as well as benchmark models, increasingly larger sample datasets will be added into the mix as well. 

_"very small data"_  < 500 training examples, ~10 classes. This amount simulates the situation of bootstrapping a chatbot in which you have dreamed up a dozen or so intents and a handful of examples each.   

_"small data"_  < 5000 training examples ~ 10 classes.

_"not so small data"_ < 50000 ~ 10 classes.  

_test or hold-out data_ - a rather large proportion will be used for test, Even up to 50% is appropriate. The large relative proportion of test as well as the low absolute numbers of data instances (especially in very small data) will emphasize the challenge of this evaluation - using transfer learning to generalize well to unseen instances - as in the case of small data, it is more probable that there will be examples in the test set with lexical features not seen in instances in the training set. 



### Implementation
### Refinement

## IV. Results
_(approx. 2-3 pages)_

### Model Evaluation and Validation
### Justification

## V. Conclusion
_(approx. 1-2 pages)_

### Free-Form Visualization
### Reflection
### Improvement
