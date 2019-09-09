import string
import csv
from flair.data import Sentence
from flair.models import SequenceTagger
from flair.embeddings import WordEmbeddings, FlairEmbeddings, StackedEmbeddings, DocumentPoolEmbeddings, BertEmbeddings, ELMoEmbeddings, OpenAIGPTEmbeddings, RoBERTaEmbeddings, XLNetEmbeddings, BytePairEmbeddings, XLNetEmbeddings, OpenAIGPT2Embeddings, XLMEmbeddings
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
from sklearn.preprocessing import MultiLabelBinarizer
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
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense, Conv1D, Conv2D, MaxPooling1D, Flatten, Embedding, Reshape, Input, SimpleRNN, LSTM, InputLayer, GRU, GlobalMaxPooling1D, Bidirectional
import torch.nn as nn
import torch.nn.functional as F
from keras.layers.advanced_activations import PReLU, ELU
from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from modAL.uncertainty import uncertainty_sampling
from collections import Counter
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer
from keras.layers import Embedding
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from eli5.formatters import format_as_text, format_as_html

keras = True
keras_mode = "MLP" # MLP, CNN, RNN, EMB
multi_label = True
attention = False
stacked = False
learned_emb = False

def parse_string(a_str):
    to_ret = "".join([c.lower() for c in a_str if c in string.ascii_letters or c in string.whitespace])
    to_ret2 = to_ret.split()
    to_ret3 = " ".join(to_ret2)
    return to_ret3


def get_misclass():
    return np.where(Y_val != pipe.predict(X_val))




class MultiLabelProbClassifier(BaseEstimator, ClassifierMixin):

    def __init__(self, clf):
        self.clf = clf

    def fit(self, X, y):
        self.clf.fit(X, y)
        self.model = self.clf.model

    def predict(self, X):
        ret = self.clf.predict(X)
        return ret

    def predict_proba(self, X):
        if len(X) == 1:
            self.probas_ = self.clf.predict_proba(X)[0] 
            sums_to = sum(self.probas_)
            new_probs = [x / sums_to for x in self.probas_]
            return new_probs
        else:
            self.probas_ = self.clf.predict_proba(X)
            #print(self.probas_)
            ret_list = []
            for list_of_probs in self.probas_:
                sums_to = sum(list_of_probs)
                #print(sums_to)
                new_probs = [x / sums_to for x in list_of_probs]
                ret_list.append(np.asarray(new_probs))
            return np.asarray(ret_list)


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






stacked_embeddings = DocumentPoolEmbeddings([WordEmbeddings('en'),
                                            #XLMEmbeddings()
                                            #ELMoEmbeddings('original')
                                            #XLNetEmbeddings(),
                                            #OpenAIGPT2Embeddings(),
                                        #FlairEmbeddings('news-forward-fast'),
                                        #FlairEmbeddings('news-backward-fast'),
                                        WordEmbeddings('glove'),
                                        WordEmbeddings('en-crawl'),
                                       #BytePairEmbeddings('en', 300),
                                       ], pooling = 'mean')


stacked_embeddings_max = DocumentPoolEmbeddings([WordEmbeddings('en'),
                                            #XLMEmbeddings()
                                            #ELMoEmbeddings('original')
                                            #XLNetEmbeddings(),
                                            #OpenAIGPT2Embeddings(),
                                        #FlairEmbeddings('news-forward-fast'),
                                        #FlairEmbeddings('news-backward-fast'),
                                        #WordEmbeddings('glove'),
                                        #WordEmbeddings('en-crawl'),
                                       #BytePairEmbeddings('en', 300),
                                       ], pooling = 'max')


stacked_embeddings_min = DocumentPoolEmbeddings([WordEmbeddings('en'),
                                            #XLMEmbeddings()
                                            #ELMoEmbeddings('original')
                                            #XLNetEmbeddings(),
                                            #OpenAIGPT2Embeddings(),
                                        #FlairEmbeddings('news-forward-fast'),
                                        #FlairEmbeddings('news-backward-fast'),
                                        #WordEmbeddings('glove'),
                                        #WordEmbeddings('en-crawl'),
                                       #BytePairEmbeddings('en', 300),
                                       ], pooling = 'min')


#en_embedding = WordEmbeddings('en')


with open('card_classification2.csv') as csvfile:
    reader = csv.reader(csvfile)
    list_of_sentences = []
    list_of_labels = []
    list_of_embeddings = []
    if not learned_emb:
        for row in reader:
            if multi_label:
                list_of_labels.append(row[:-1])
                parsed_string = row[-1]
            else:
                list_of_labels.append(row[0])
                parsed_string = row[1]
            list_of_sentences.append(parsed_string)
            set_obj = Sentence(parsed_string)
            stacked_embeddings.embed(set_obj)
            avg_emb = set_obj.get_embedding().cpu().detach().numpy()
            if stacked:
                stacked_embeddings_min.embed(set_obj)
                min_emb = set_obj.get_embedding().cpu().detach().numpy()
                stacked_embeddings_max.embed(set_obj)
                max_emb = set_obj.get_embedding().cpu().detach().numpy()
                concat_emb = np.concatenate((avg_emb, min_emb, max_emb), axis=None)
            #list_of_embeddings.append(set_obj.get_embedding().cpu().detach().numpy())
                list_of_embeddings.append(concat_emb)
            else:
                list_of_embeddings.append(avg_emb)
    else:
        for row in reader:
            if multi_label:
                list_of_labels.append(row[:-1])
                parsed_string = row[-1]
            else:
                list_of_labels.append(row[0])
                parsed_string = row[1]
            list_of_sentences.append(parsed_string)

if not learned_emb:
    print(list_of_embeddings[1])
#print(Counter(list_of_labels).items())

if learned_emb:
    t = Tokenizer()
# fit the tokenizer on the documents
    t.fit_on_texts(list_of_sentences)
    b_encoder = LabelBinarizer()
    new_labels = b_encoder.fit_transform(list_of_labels)
    X_train, X_val, Y_train, Y_val = train_test_split(np.asarray(list_of_sentences), new_labels, test_size = 0.2, stratify = list_of_labels, random_state=42)    
    output_size = len(set(list_of_labels))
    print(output_size)
    print(Y_val[1])
    sequences_train = t.texts_to_sequences(X_train)
    sequences_valid = t.texts_to_sequences(X_val)
    X_train = pad_sequences(sequences_train)
    X_val = pad_sequences(sequences_valid, maxlen=X_train.shape[1])
    sequence_length = X_train.shape[1]
    encoded_docs = t.texts_to_sequences(list_of_sentences)
    print("dictionary size: ", len(t.word_index))
    vocabulary_size = len(t.word_index)
    EMBEDDING_DIM=300
    vocabulary_size=len(t.word_index)+1
    embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
    for word, i in t.word_index.items():
        try:
            word_sent = Sentence(word)
            en_embedding.embed(word_sent)
            embedding_vector = word_sent[0].embedding.cpu().detach().numpy()
            embedding_matrix[i] = embedding_vector
        except KeyError:
            embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)
    embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)


else:
    if multi_label:
        mlb = MultiLabelBinarizer()
        list_of_mlb_labels = mlb.fit_transform(list_of_labels)
        X_train, X_val, Y_train, Y_val, Emb_train, Emb_val = train_test_split(np.asarray(list_of_sentences), np.asarray(list_of_mlb_labels), np.asarray(list_of_embeddings), test_size = 0.30, stratify = list_of_labels, random_state=42)
    else:
        X_train, X_val, Y_train, Y_val, Emb_train, Emb_val = train_test_split(np.asarray(list_of_sentences), np.asarray(list_of_labels), np.asarray(list_of_embeddings), test_size = 0.33, stratify = list_of_labels, random_state=42)    

#X_train, X_val, Y_train, Y_val, Emb_train, Emb_val = train_test_split(np.asarray(list_of_sentences), np.asarray(list_of_labels), np.asarray(list_of_embeddings), test_size = 0.33, stratify = list_of_labels, random_state=42)

def create_model(optimizer='adam', kernel_initializer='glorot_uniform', epochs = 5):
        model = Sequential()
        if not learned_emb:
            if keras_mode == "CNN":
                model.add(Reshape((1, list_of_embeddings[1].size), input_shape = Emb_train.shape[1:])) ##magical fucking stupid keras BS needed for RNN/CNN
                model.add(Conv1D(filters=300, kernel_size=1, strides = 5, activation='relu')) ##works now
                model.add(Flatten()) ##need this with Conv1D
                #model.add(GlobalMaxPooling1D()) ##pooling would go here instead of flattening if you're into that 
                model.add(Dense(len(np.unique(Y_val)),activation='softmax',kernel_initializer=kernel_initializer, use_bias = False))

            elif keras_mode == "RNN":
                model.add(Reshape((1, list_of_embeddings[1].size), input_shape = Emb_train.shape[1:])) 
                if attention:
                    model.add(Bidirectional(GRU(list_of_embeddings[1].size, activation = 'relu', return_sequences = True))) ##this works too - seems to be better for smaller datasets too!
                    model.add(SeqWeightedAttention())
                else:
                    model.add(Bidirectional(GRU(list_of_embeddings[1].size, activation = 'relu')))
                model.add(Dense(len(np.unique(Y_val)),activation='softmax',kernel_initializer=kernel_initializer, use_bias = False))
            else: ##for simple MLP models 
                if not multi_label:
                    model.add(Dense(list_of_embeddings[1].size, activation='relu',kernel_initializer='he_uniform', use_bias = False))
                    model.add(Dense(len(np.unique(Y_train)),activation='softmax',kernel_initializer=kernel_initializer, use_bias = False))
                else:
                    model.add(Dense(list_of_embeddings[1].size, activation='relu',kernel_initializer='he_uniform', use_bias = False))
                    model.add(Dense(Y_train.shape[1] ,activation='sigmoid',kernel_initializer=kernel_initializer, use_bias = True))
        else:
            model.add(embedding_layer)
            model.add(Bidirectional(GRU(EMBEDDING_DIM, return_sequences=False, input_shape=(sequence_length, EMBEDDING_DIM), activation = 'relu')))
            model.add(Dense(output_size, activation='softmax'))

        if multi_label:
            model.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])
        else:
            model.compile(loss='categorical_crossentropy',optimizer=optimizer, metrics=['accuracy'])
        return model



if keras:
    checkpointer = ModelCheckpoint(filepath='/tmp/weights.hdf5', verbose=1, save_best_only=True)
    stopper = EarlyStopping(patience = 50, restore_best_weights = True)    
    model = KerasClassifier(build_fn=create_model, batch_size = 32, epochs = 200, callbacks=[stopper], validation_split = 0.3)


the_model = MultiLabelProbClassifier(model)

#model = SVC(kernel = "rbf", probability = True)
#model = KNeighborsClassifier(n_neighbors=1, metric='cosine', weights = 'distance')
#model  = AdaBoostClassifier(n_estimators = 100, random_state = 42)
#model = RandomForestClassifier(n_jobs = -1, n_estimators = 100, max_features = "auto", criterion = "entropy")
#model = MLPClassifier(hidden_layer_sizes=(500,), activation = 'relu', solver = 'adam', verbose = True, max_iter = 100) #early_stopping = True, validation_fraction = 0.3, n_iter_no_change = 100)




if not learned_emb:
    pipe = Pipeline([('text2vec', Text2Vec()), ('model', the_model)])
else:
    pipe = model
#model.fit(Emb_train, Y_train)
pipe.fit(X_train, Y_train)

pred = pipe.predict(X_val)


te = TextExplainer(random_state=42, n_samples=300, position_dependent=True)

def explain_pred(sentence):
    te.fit(sentence, pipe.predict_proba)
    t_pred = te.explain_prediction()
    #t_pred = te.explain_prediction(top = 20, target_names=["ANB", "CAP", "ECON", "EDU", "ENV", "EX", "FED", "HEG", "NAT", "POL", "TOP", "ORI", "QER","COL","MIL", "ARMS", "THE", "INTHEG", "ABL", "FEM", "POST", "PHIL", "ANAR", "OTHR"])
    txt = format_as_text(t_pred)
    html = format_as_html(t_pred)
    html_file = open("latest_prediction.html", "a+")
    html_file.write(html)
    html_file.close()
    print(te.metrics_)



if not multi_label:
    print(accuracy_score(Y_val, pred))

    labels = np.unique(Y_train)
    conf = confusion_matrix(Y_val, pred, labels=labels)

    print(pd.DataFrame(conf, index=labels, columns=labels))


    predicts = pipe.predict(X_val)
    probs = pipe.predict_proba(X_val)
    a_df = pd.DataFrame(probs, index=Y_val, columns=labels)
    a_df[a_df.eq(0)] = np.nan
    print(a_df.round(2))

    misclass = get_misclass()

    print("misclassified examples!!!")
    print(get_misclass())
    print(a_df.iloc[get_misclass()].round(2))

else: 
    #print(mlb.classes_)
    predicts = pipe.predict(X_val[0])
    myvec = Text2Vec()
    #print(pipe.named_steps['one_hot_encoder'].inverse_transform(myvec.transform(X_val)))
    probs = pipe.predict_proba(X_val[0:2])
    #explain_pred(str(X_val[0]))
    #a_df = pd.DataFrame(probs, index=Y_val, columns=labels)
    #a_df[a_df.eq(0)] = np.nan
    #print(a_df.round(2))
    print(mlb.classes_)
    #print(predicts)
    print(np.around(probs, decimals = 2))
if keras:
    pipe.named_steps['model'].model.save('keras_model.h5')
    pipe.named_steps['model'].model = None
joblib.dump(pipe, 'saved_card_classification.pkl')
print("Model Dumped!!!!")






