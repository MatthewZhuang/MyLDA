# MyLDA
Author: guomianzhuang
Title: StandardLDA and SparseLDA

This project implements the StandardLDA and SparseLDA algorithm. And Compare these two Algorithms with a experiment.

The implementation is in lda-core directory.

StandardGibbs Sampling Complexity is O(MNmK)( Nm is the average document length). The standard implementation is too slow to apply to the project. So we need a more faster implementation. Yao presents an algorithm and data structure for evaluating Gibbs sampling distribution, SparseLDA. This method can reduce the time complexity to O(MNm|NoneZero(Nkt)|), if the topic num is large, the Nkt will be sparse, this method can improve the speed of model inference.
We have implemented this two LDA model with Gibbs Sampling. (See LdaModel.py and LdaModel_SparseLDA.py)

Exprimental environment:
System: Mac OS X
CPU: 2.7GHz intel Core i5
Memory: 8GB

Result:
I set the alpha=50.0/K, beta = 0.01, iteration=1000 for all cases and use a corpus including 1000 articles.
The result show in Figure 1, SparseLDA has a lower overall time than StandardLDA, and the time increases slowly as we increase the topics.
