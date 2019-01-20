### Evaluation of pretrained language models for short text classification and small datasets
To read about what this is, see the written report capstone_project.pdf 

This project is structured as follows:

*model_evaluation* contains the python source code for running this evaluation. 
Start with data_preparation.py and base.py - function *run_evaluation_baseline* to get an
idea of how to get started if you want to try to reproduce this evaluation.

*data* contains the downloaded data from the amazon qa dataset. The raw tarballs are in teh *raw* folder, 
and the extracted json files are in the *json* folder. 

The *suites* subfolder in *data* contain preprocessed datasets (pickled DataProvider objects)
which were used for the evaluation. Passing this folder to the *run_evaluation_** functions in 
the source code tells the evaluation to load the datasets contained in that folder. 

The *doc* folder contains the written proposal and report and related files.

The *img* folder contains images used in the report. 

For BERT, You will need to create an output folder where bert stores intermediate results. 

The *results* folder contains persisted Session objects resulting from the evaluation, and 
the final_results_df.pkl containing the dataframe results. 

The following pythong dependencies need to be installed (pip install, lastest versions):
scikit-learn
pandas
numpy
matplotlib

For BERT, the bert source code needs to be cloned and added to the sourcepath (See bert.py) for details. 
Running BERT should take place in a google-cloud-platform tpu- accelerated machine. 
You will NOT get far on a normal laptop... These machines have tensorflow intstalled, which is a 
dependency of running bert. 

For ULMFiT, pytorch and fastai must be intstalled. See ulmfit.py for details. 

 

 
