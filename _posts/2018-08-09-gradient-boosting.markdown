---
layout: post
title:  "Gradient Boosting"
date:   2018-08-09 14:21:49 -0000
author: Ji Sun
tags: "machine_learning"
categories: blog
---
<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>
### Derivation of Algorithm
As we all know, for any predictive learning problem, solution is trying to minimize the expected value of loss function \\(L(y, F(x))\\) over the joint distribution of all \\((y, \mathbf{x})\\)  
<center>
$$F^* = arg\min_FE_{y,\mathbf{x}}L(y, F(\mathbf{x})) = arg\min_FE_{\mathbf{x}}[E_y(L(y,F(\mathbf{x})))|\mathbf{x}]$$
</center>
Such function approximation \\(F(\mathbf{x})\\) can be expressed as an expansions of the form  
<center>
$$F(\mathbf{x};\{\beta_m,\mathbf{\alpha_m}\}_1^M)=\sum_{m=1}^M\beta_mh(\mathbf{x};\mathbf{\alpha_m})$$
</center>
In general, choosing a parameterized model or approximation function \\(F(\mathbf{x};\mathbf{P})\\) changes the function optimization problem to one of parameter optimization which is Gradient Descent (or Steepest Descent)
<center>
$$\mathbf{P}^*=arg\min_{\mathbf{P}}E_{y,\mathbf{x}}L(y, F(\mathbf{x};\mathbf{P}))$$
$$F(\mathbf{x}^*)=F(\mathbf{x};\mathbf{P}^*)$$
$$\mathbf{P}^*=\sum_{m=0}^M\mathbf{p}_m$$
$$\mathbf{g}_m=\{g_{jm}\}=\{[\frac{\delta E_{y,\mathbf{x}}L(y, F(\mathbf{x};\mathbf{P_{m-1}}))}{\delta P_j}]\}$$
$$\mathbf{P}_{m-1}=\sum_{i=0}^{m-1}\mathbf{p}_i$$
$$\downarrow$$
$$\mathbf{p}_{m}=-\rho_m\mathbf{g}_m$$
$$\rho_m=arg\min_\rho E_{y,\mathbf{x}}L(y, \mathbf{P}_{m-1}-\rho\mathbf{g}_m)$$
</center>
Another way to find the function is using numeric optimization in function space.
<center>
$$F^*(\mathbf{x})=arg\min_{F}E_{y,\mathbf{x}}L(y, F(\mathbf{x}))$$
$$=arg\min_{F}E_y[L(y, F(\mathbf{x})|\mathbf{x}]$$
$$F^*(\mathbf{x})=\sum_{m=0}^Mf_m(\mathbf{x})$$
$$g_m(\mathbf{x})=E_y[\frac{\delta L(y, F(\mathbf{x}))}{\delta F(\mathbf{x})}]_{F(\mathbf{x})=F_{m-1}(\mathbf{x})}$$
$$F_{m-1}(\mathbf{x})=\sum_{i=0}^{m-1}f_i(\mathbf{x})$$
$$f_m(\mathbf{x})=-\rho_mg_m(\mathbf{x})$$
$$\rho_m=arg\min_\rho E_{y,\mathbf{x}}L(y, F_{m-1}(\mathbf{x})-\rho g_m(\mathbf{x}))$$
</center>
However, for the finite data tuples, joint distribution of \\((y, \mathbf{x})\\) is hard to estimate, thus adopting the 'greedy-stagewise' approach.  
For m = 1,2,3,...,M  
<center>
$$(\beta_m, \mathbf{a}_m)=arg\min_{\beta,\mathbf{a}}\sum_{i=1}^{N}L(y_i,F_{m-1}(\mathbf{x}_i)+\beta h(\mathbf{x}_i;\mathbf{a}))$$
$$F_m(\mathbf{x})=F_{m-1}(\mathbf{x})+\beta_m h(\mathbf{x};\mathbf{a}_m)$$
</center>
Still we can formalize the data based negative gradient.
<center>
$$-g_m(\mathbf{x}_i)=-[\frac{\delta L(y_i,F_{m-1}(\mathbf{x}_i))}{\delta F_{m-1}(\mathbf{x}_i)}]$$
</center>
In order to generalize the direction to other \\(\mathbf{x}\\)-values, we should find a parameterized function \\(\beta h(\mathbf{x};\mathbf{a})\\) that close to \\(-g_m(\mathbf{x})\\)  
<center>
$$\mathbf{a}_m=arg\min_{\beta,\mathbf{a}}\sum_{i=1}^N[-g_m(\mathbf{x}_i)-\beta h(\mathbf{x}_i;\mathbf{a})]^2$$
$$\rho_m=arg\min_{\rho}\sum_{i=1}^NL(y_i,F_{m-1}(\mathbf{x}_i)+\rho h(\mathbf{x}_i;\mathbf{a}_m))$$
$$F_m(\mathbf{x})=F_{m-1}(\mathbf{x})+\rho_mh(\mathbf{x};\mathbf{a}_m)$$
</center>

### References
1. [Greedy Function Approximation: A Gradient Boosting Machine](/resource/gdbt_1.pdf)
2. [A Gentle Introduction to Gradient Boosting](/resource/gdbt_2.pdf)
3. [Boosting: Foundations and Algorithms](/resource/gdbt_3.pdf)