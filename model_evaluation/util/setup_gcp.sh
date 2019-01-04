#!/usr/bin/env bash
git clone https://github.com/jloutz/nlp_capstone.git nlp_capstone
cd nlp_capstone
git clone https://github.com/google-research/bert.git bert-master
set PYTHONPATH=/home/jloutz67/nlp_capstone/bert-master