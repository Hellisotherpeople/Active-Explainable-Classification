# Active-Explainable-Classification
A set of tools for leveraging active learning and model explainability for effecient document classification

## What is this?

One component of my vision of FULLY AUTOMATED competative debate case production. When I take in massive sums of articles from a news API, I need a way to classify these documents into various buckets. I have to generate my own labeled data for this. That is a problem. Most people don't realize that the sample effeciency in models which utilize transfer learning is so great that AI-assisted data labeling is extremely useful and can significantly shorten what is ordinarily a painful data labeling process. 


1. We need a way to quickly create word embedding powered document classifiers which learn with a human in the loop. For some classes, an extremely limited number of examples may be all that is necessary to get results that a user would consider to be succesful for their task. 

2. I want to know what my model is learning - so I integrate the word embeddings avalible with [Flair](https://github.com/zalandoresearch/flair), combine with Classifiers in Sklearn and PyTorch, and finish it off with the [LIME](https://arxiv.org/pdf/1602.04938.pdf) algorithim for model interpretability (implemented within the [ELI5](https://eli5.readthedocs.io/en/latest/index.html) Library)





TODO: 1. Finish README - Cite relavent technologies and papers 
2. Documentation/Examples/Installation Instructions 
3. More examples


## Examples 

Toy example of a possible debate classifier seperating between 11 classes 

ANB = Antiblackness, CAP = Capitalism, ECON = Economy, EDU = Education, ENV = Environment, EX = Extinction, FED = Federalism, HEG = Hegemony, NAT = Natives, POL = Politics, TOP = Topicality

Top matrix is a confusion matrix of my validation set 

Bottom matrix is showing classification probabilities for each individual example in my validation set.

![](https://github.com/Hellisotherpeople/Active-Explainable-Classification/blob/master/conf_matrix.png)

Takes in documents from the user using Standard Input - Then the model classifies, explains why it classified the way it did, and asks the user if the predicted label is the ground truth or not. User supplies the ground truth, the model incrementally trains on the new example, and that new example (with human supplied label) is appended to my dataset and the cycle continues. This is called *active learning* 

![](https://github.com/Hellisotherpeople/Active-Explainable-Classification/blob/master/explaination.png)
