---
layout: post
title:  "Query Optimization by using Deep Q-learning"
date:   2019-01-29 13:20:00 +8000
author: Ji Sun
tags: "database optimizer deep_learning"
categories: blog
---
<script type="text/javascript" src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
### Introduction
Join order enumeration has been studied decades, in Postgresql,
Dynamic Programming based on left deep was adopted for small queries,
while Genetic Algorithm for large queries, both methods are costly and have no
optimal guarantee. Q-learning is essentially a Dynamic Programming enumeration method for
Markov Decision Process (MDP), it can determine next actions based on the minimum values
in Q-table. According to the Bellman Equation, the q-value can be updated as following:
<center>
$$Q(S,a) = Q(S,a) + \alpha (Q(S,a) - (\gamma * \max_{a'}Q(S',a') + R(S,a)))$$
</center>
$$\alpha$$ is the learning rate, $$\gamma$$ is the discount factor.
However, Q-larning based on Q-table is still costly, we can not get the optimal trajectory for each
query online, therefore, we replaced Q-table by a Deep Neural Network Like LSTM in our work.
Deep Q-learning has the ability of generalization, we can train the model on test queries,and the model
can determine the join order of unknown queries faster and better than traditional query optimizer.  
In this blog, we first propose a effective Q-learning methed to find optimal join order for single query, and then
we generalize this method to multiple queries, we propose an Inverse Q-Learning method to help the model converge faster 
single query and use transfer learning to help the model to determine the join order for unknown queries.

### Q-learning for single query
#### Input Features
For single query, action is the chosen join condition, and the sequence of picked
join conditions is enough to depict the state, each action is represented as a one-hot vector where
each bit represents whether a join condition was picked. Sequence of selected actions is input to the model
 as State, and the output is values for different actions.
#### Model
Because the future cost is also influenced by the order of history actions and RNN model can learn the order features
of one sequence, we take LSTM as deep Q-network, the input dimention is the number of join conditions in total and
the input length is the number of chosen actions before current state, which contains two hidden layers and one fully
 connected linear layer as outputs.
#### Deep Q-learnings
#### Results
![value_convergence.png](/figures/value_convergence.png)
![reward_convergence.png](/figures/reward_convergence.png)
 