import glob
import os
from bs4 import BeautifulSoup
import bs4
import string
import flair
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings, BertEmbeddings, ELMoEmbeddings
import torch
# create a StackedEmbedding object that combines glove and forward/backward flair embeddings
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_similarity_score
#import numpy as np
from docx import Document
import sys
import numpy as np
from itertools import islice
from collections import deque
import csv
from random import shuffle
from sklearn.externals import joblib
from time import time
import pickle
import umap
from sklearn.pipeline import make_union, Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
#from sklearn.pipeline import Pipeline, make_pipeline
import eli5 
from eli5.lime import TextExplainer
from eli5 import explain_prediction
from eli5.formatters import format_as_text, format_as_html
import pandas as pd
from IPython.display import display
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.models import load_model

keras = False
increment = False

stacked_embeddings = DocumentPoolEmbeddings([
                                        WordEmbeddings('en'),
                                        WordEmbeddings('glove'),
                                        WordEmbeddings('extvec'),#ELMoEmbeddings('original'),
                                        #BertEmbeddings('bert-base-cased'),
                                        #FlairEmbeddings('news-forward-fast'),
                                        #FlairEmbeddings('news-backward-fast'),
                                        ]) #, mode='max')

def create_model(optimizer='adam', kernel_initializer='glorot_uniform', epochs = 5):
        model = Sequential()
        model.add(Dense(list_of_embeddings[1].size, activation='relu',kernel_initializer='he_uniform', use_bias = True))
        model.add(Dense(11,activation='softmax',kernel_initializer=kernel_initializer, use_bias = True))
        model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
        return model


def parse_string(a_str):
    to_ret = "".join([c.lower() for c in a_str if c in string.ascii_letters or c in string.whitespace])
    to_ret2 = to_ret.split()
    to_ret3 = " ".join(to_ret2)
    return to_ret3

class Text2Vec( BaseEstimator, TransformerMixin):
    '''
    def __init__():
        self.X = None
    '''
    def fit(self, X, y=None):
        return self 

    def transform(self, X):
        list_of_emb = []
        size_of_emb = stacked_embeddings.embedding_length
        if not isinstance(X, str):
            for doc in X:
                p_str = parse_string(doc)
                if not p_str:
                    list_of_emb.append(np.zeros((size_of_emb,), dtype=np.float32))##TODO: don't hard code vector size 
                else:
                    a_set = Sentence(p_str)
                    stacked_embeddings.embed(a_set)
                    list_of_emb.append(a_set.get_embedding().cpu().detach().numpy())
            to_ret = np.array(list_of_emb)
        else:
            try:
                p_str = parse_string(X)
                if not p_str:
                    to_ret = np.zeros((size_of_emb,), dtype=np.float32)##TODO here too
                else:
                    a_set = Sentence(p_str)
                    stacked_embeddings.embed(a_set)
                    to_ret = a_set.get_embedding().cpu().detach().numpy().reshape(1, -1)
            except:
                print(type(X))
                print(X)
        return to_ret



pipe = joblib.load('saved_card_classification.pkl')

if keras:
    pipe.named_steps['model'].model = load_model('keras_model.h5')


te = TextExplainer(random_state=42, n_samples=10000, position_dependent=True)

def explain_pred(sentence):
    te.fit(sentence, pipe.predict_proba)
    #txt = format_as_text(te.explain_prediction(target_names=["green", "neutral", "red"]))
    t_pred = te.explain_prediction(top = 20, target_names=["ANB", "CAP", "ECON", "EDU", "ENV", "EX", "FED", "HEG", "NAT", "POL", "TOP"])
    txt = format_as_text(t_pred)
    html = format_as_html(t_pred)
    html_file = open("latest_prediction.html", "a+")
    html_file.write(html)
    html_file.close()
    print(te.metrics_)


def print_misclass():
    print("misclassified examples!!!")
    print(np.where(Y_val != pipe.predict(X_val)))

    

with open('card_classification.csv', 'a') as csvfile:
    spamwriter = csv.writer(csvfile)
    done = False
    while not done:
        to_process = input("Please copy and paste a document to be classified Ctrl-shift-D or ctrl-D to exit")    
        print("MODEL PREDICTION:")
        pred = pipe.predict(str(to_process))
        print(pred)
        explain_pred(str(to_process))
        label = input("What is the ground truth label of this? Seperate labels with a space")
        if label == "":
            pass
        elif label == "f":
            break
        elif label == "stop":
            csvfile.close()
            if keras:
                pipe.named_steps['model'].model.save('keras_model.h5')
                pipe.named_steps['model'].model = None
            joblib.dump(pipe, 'saved_card_classification.pkl')
            print("Model Dumped!!!!")
            done = True
            sys.exit()
        else:
            the_labels = label.split()
            if increment == True:
                t_model = pipe.named_steps['model']
                ppset = Sentence(str(to_process))
                stacked_embeddings.embed(ppset)
                the_emb = ppset.get_embedding().cpu().detach().numpy().reshape(1, -1)
                t_model.partial_fit(the_emb, the_labels) ##INCREMENTAL LEARNING MODE ENGAGED
            the_labels.append(str(to_process))
            spamwriter.writerow(the_labels)
            csvfile.flush()
