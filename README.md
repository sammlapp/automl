## AutoML vs Linear Probing for Transfer Learning on Audio Embeddings

Modern foundation models like BirdCODE and Perch V2 like to evaluate their transfer learning capabilities by generating embeddings on audio, then training shallow classifiers on new/specific classes using a small number of samples. Examples: https://www.nature.com/articles/s41598-023-49989-z and https://www.biorxiv.org/content/10.64898/2026.07.31.742086v1.full.pdf 
It’s popular to evaluate this approach on well-established benchmarks such as BEANS, BirdSET, and the newer and larger WABAD dataset
BUT, experiments typically just train a linear probe on the embeddings for each class. Meanwhile, AutoML approaches recognize the benefit of hyper parameter optimization and ensembling for shallow classifier performance. 
This suggests that transfer learning performance of these models could be improved by using AutoML frameworks rather than just linear probing, but nobody is doing this yet as the common workflow. 

Suggested approach:
Get a copy of WABAD and use a fraction of samples per class as the training set, rest as eval. Similar to the approach of the Ghani paper linked above. 
Generate embeddings with a popular foundation model, likely Perch V2
Train the typical way: linear probe; evaluate performance
Also use AutoGluon or another AutoML package to train and evaluate with the same train/eval data. Is performance better? By how much? For all classes or just some? Does it depend on how many training samples are available? 