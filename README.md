# Active-Explainable-Classification
A set of tools for leveraging active learning and model explainability for effecient document classification

Note: A webapp which implements much of the functionality of this repo (minus the active labeling part, at this time anyway!) can be found here: https://huggingface.co/spaces/Hellisotherpeople/Interpretable_Text_Classification_And_Clustering

## What is this?

One component of my vision of FULLY AUTOMATED competative debate case production. 


I want to take in massive sums of articles from a news API which will be placed in their corresponding file based on where my classifier says I should put them. I have to generate my own labeled data for this. That is a *problem*. Most people don't realize that the sample effeciency in models which utilize transfer learning is so great that AI-assisted data labeling is extremely useful and can significantly shorten what is ordinarily a painful data labeling process. 


1. We need a way to quickly create word embedding powered document classifiers which learn with a human in the loop. For some classes, an extremely limited number of examples may be all that is necessary to get results that a user would consider to be succesful for their task. 

2. I want to know what my model is learning - so I integrate the word embeddings avalible with [Flair](https://github.com/zalandoresearch/flair), combine with Classifiers in Sklearn and Tensorflow/Keras/PyTorch, and finish it off with a nice squeeze of the [LIME](https://arxiv.org/pdf/1602.04938.pdf) algorithim for model interpretability (implemented within the [ELI5](https://eli5.readthedocs.io/en/latest/index.html) Library)


TODO:
* Integrate Uncertainty measurments and only have it prompt to label those examples (self label what it's certain about) 
* Finish README - Cite relavent technologies and papers 
* Documentation/Examples/Installation Instructions 
* More examples
* Enable Cross Validation and Grid Search
* Figure out better way to store embeddings (stop moving the embeddings from GPU to CPU ineffeciently)

Changelog: 

8/12/2019 - 
* got Keras RNN/CNN working!
* Now Prints out misclassified examples in validation test set and we see probabilities. 
* Easy to switch between MLP/CNN/RNN

8/9/2019 - 
* Added model, model weights, updated dataset, and misc code updates 
* Tried to get Keras LSTM/CNN to work (failed) 
* Experimented with different settings on LIME.


8/8/2019 - 
* Added Keras model support - now utilizes by default KNN if not in Keras mode and a Neural Network if in Keras mode. 
* Added HTML exporting of model explanations. - Thank you ELI5!
* Tested the possibility of doing Multilabel classification with TextExplainer... doesn't seem to work :( 
* Added pictures

## Examples 

Toy example of a possible debate classifier seperating between 11 classes


![](https://github.com/Hellisotherpeople/Active-Explainable-Classification/blob/master/explaination2.png)

ANB = Antiblackness, CAP = Capitalism, ECON = Economy, EDU = Education, ENV = Environment, EX = Extinction, FED = Federalism, HEG = Hegemony, NAT = Natives, POL = Politics, TOP = Topicality

Top matrix is a confusion matrix of my validation set 

This classifier gets 75% accuracy (~150 examples in train set, 0.2 * 150 in val set) 

Bottom matrix is showing classification probabilities for each individual example in my validation set.

![](https://github.com/Hellisotherpeople/Active-Explainable-Classification/blob/master/conf_matrix.png)

Takes in documents from the user using Standard Input - Then the model classifies, explains why it classified the way it did, and asks the user if the predicted label is the ground truth or not. User supplies the ground truth, the model incrementally trains on the new example, and that new example (with human supplied label) is appended to my dataset and the cycle continues. This is called *active learning*

![](https://github.com/Hellisotherpeople/Active-Explainable-Classification/blob/master/explaination3.png)

![](https://github.com/Hellisotherpeople/Active-Explainable-Classification/blob/master/explaination.png)
