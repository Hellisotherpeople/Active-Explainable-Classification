import string
import csv
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings, BertEmbeddings, ELMoEmbeddings, OpenAIGPTEmbeddings, RoBERTaEmbeddings, XLNetEmbeddings
import torch
from torch import tensor
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, log_loss, roc_auc_score, make_scorer, balanced_accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.neural_network import MLPClassifier
from time import time
import pickle
import umap
from sklearn.pipeline import make_union, Pipeline
from sklearn.base import TransformerMixin, BaseEstimator
#from sklearn.pipeline import Pipeline, make_pipeline
import eli5 
from eli5.lime import TextExplainer
from eli5 import explain_prediction
from eli5.formatters import format_as_text
import pandas as pd
from sklearn.externals import joblib
from keras.callbacks import ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Embedding, Reshape, Input, SimpleRNN, LSTM
import torch.nn as nn
import torch.nn.functional as F


keras = True

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
        size_of_emb = list_of_embeddings[1].size
        if not isinstance(X, str):
            for doc in X:
                #p_str = parse_string(doc)
                p_str = doc
                if not p_str:
                    list_of_emb.append(np.zeros((size_of_emb,), dtype=np.float32))##TODO: don't hard code vector size 
                else:
                    a_set = Sentence(p_str)
                    stacked_embeddings.embed(a_set)
                    list_of_emb.append(a_set.get_embedding().cpu().detach().numpy())
            to_ret = np.array(list_of_emb)
        else:
            try:
                #p_str = parse_string(X)
                p_str = X
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



stacked_embeddings = DocumentPoolEmbeddings([#WordEmbeddings('en'),
                                        #WordEmbeddings('glove'),
                                        WordEmbeddings('en-crawl')])

with open('card_classification.csv') as csvfile:
    reader = csv.reader(csvfile)
    list_of_sentences = []
    list_of_labels = []
    list_of_embeddings = []
    for row in reader:
        list_of_labels.append(row[0])
        parsed_string = parse_string(row[1])
        parsed_string = row[1]
        list_of_sentences.append(parsed_string)
        set_obj = Sentence(parsed_string)
        stacked_embeddings.embed(set_obj)
        list_of_embeddings.append(set_obj.get_embedding().cpu().detach().numpy())


X_train, X_val, Y_train, Y_val, Emb_train, Emb_val = train_test_split(np.asarray(list_of_sentences), np.asarray(list_of_labels), np.asarray(list_of_embeddings), test_size = 0.30, stratify = list_of_labels, random_state=42)

print(list_of_embeddings[1].size)

def create_model(optimizer='adam', kernel_initializer='glorot_uniform', epochs = 5):
        model = Sequential()
        #model.add(Reshape((137, 1, 400), input_shape = (137, 400)))
        #model.add(Conv1D(64, 1, activation='relu'))
        model.add(Dense(list_of_embeddings[1].size, activation='relu',kernel_initializer='he_uniform', use_bias = False))
        #model.add(LSTM(list_of_embeddings[1].size, return_sequences = True,))
        model.add(Dense(len(np.unique(Y_val)),activation='softmax',kernel_initializer=kernel_initializer, use_bias = False))
        model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
        return model



if keras:
    checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)    
    model = KerasClassifier(build_fn=create_model, batch_size = 32, epochs = 150, callbacks=[checkpointer], validation_split = 0.2)



#model = SVC(kernel = "rbf", probability = True)
#model = KNeighborsClassifier(n_neighbors=5, metric='cosine', weights = 'distance')
#model  = AdaBoostClassifier(n_estimators = 100, random_state = 42)
#model = RandomForestClassifier(n_jobs = -1, n_estimators = 100, max_features = "auto", criterion = "entropy")
#model = MLPClassifier(hidden_layer_sizes=(500,), activation = 'relu', solver = 'adam', verbose = True, max_iter = 100) #early_stopping = True, validation_fraction = 0.3, n_iter_no_change = 100)





pipe = Pipeline([('text2vec', Text2Vec()), ('model', model)])
#model.fit(Emb_train, Y_train)
pipe.fit(X_train, Y_train)

pred = pipe.predict(X_val)

print(accuracy_score(Y_val, pred))

labels = np.unique(Y_val)
conf = confusion_matrix(Y_val, pred, labels=labels)

print(pd.DataFrame(conf, index=labels, columns=labels))

probs = pipe.predict_proba(X_val)
a_df = pd.DataFrame(probs, index=Y_val, columns=labels)
a_df[a_df.eq(0)] = np.nan
print(a_df.round(2))

if keras:
    pipe.named_steps['model'].model.save('keras_model.h5')
    pipe.named_steps['model'].model = None
joblib.dump(pipe, 'saved_card_classification.pkl')
print("Model Dumped!!!!")
