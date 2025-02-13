---
layout: page
permalink: /progress-report-12-02
title: Markdown file for the progress report of 12/02/2025
nav: false
nav_order: 2
description: 
---

---
author: Ahmet Akman
date: 12/02/2025
---
## Progress Report
- Implemetation of the apical multiplier to the network.
- Implementation of the gradient descent on apical multipliers.
- Implementation of the learning rule and weight offloading.
- Implementation of the multiiterative learning stages.
- Characterization of the apical learning rule.

### First pass (Phase 1):
Setting all $a=1$ and passing through the dataset portion $D_A$ all the gradients relating the apical multipliers are stored/recorded. Then taking the mean of through the all batches, for layer $k$ apical multipliers are set according to 
$$\Delta\underbar{a}_k = - \frac{1}{\#batch} \sum_{batch} \frac{\partial \mathcal{L}(\underbar{\underbar{W}}_{init}, x_{batch})}{\partial \underbar{a}_k}$$


Then the same data $D_W$ passed through the apical multiplier mapped network and change in the loss is measured. Then the following plot for different learning rate and batch sizes obtained.

![alt text](https://ahmetakman.github.io/assets/progress_reports/12_02_25/figures/lr_batch_loss_diff.png "Loss Drop")

#### Apical multiplier gradient statistics
To get an intuition about how the gradients associeted with the apical multipliers are distributed during the first pass (no update involved) over the whole data following histogram is obtained.
- Layer close to output -> higher variation.
 ![](https://ahmetakman.github.io/assets/progress_reports/12_02_25/figures/gradient_histogram.png)

### Learning rule / Weight offloading (Phase 2):
The apical learning rule is imposed onto the network via the other part of the dataset $D_W$. Further drop in loss observed. Within a single data pass, following typical batch  loss within in a single epoch are recorded and plotted.

![alt text](https://ahmetakman.github.io/assets/progress_reports/12_02_25/figures/loss_typical_phase_2.png "Single Weight Offloading")

### Multi-epoch training
To be able to get an idea which levels of loss drop we can get different set of learning rates for fixed batch size (30) is applied and loss values over 100 epochs are plotted. It is typical that around the losses (on $D_W$ or on $D_{test}$) of 0.2 it is likely for values to blow up.To be able to see if this can be prevented by loss scheduling, cosine scheduling is applied to same set of starting learning rates. That did not effect much.

- *each line represents a learning rate pair.*
![](https://ahmetakman.github.io/assets/progress_reports/12_02_25/figures/sweep_lr_a_w_no_scheduling_test_10_02_2025.png)
![](https://ahmetakman.github.io/assets/progress_reports/12_02_25/figures/sweep_lr_a_w_cosine_scheduling_test_10_02_2025.png)
To see if having smaller batch size effect: following plot suggest this allows to go lower loss values yet it has lower
![](https://ahmetakman.github.io/assets/progress_reports/12_02_25/figures/sweep_lr_a_w_no_scheduling_test_bs_1_11_02_2025.png)

Fundamentally the early stopping (higher lr pairs) runs have losses exploding earlier. **side note 1: a training is stopped if an indefinite value (NaN or inf) is observed in loss. **side note 2: a training

### Multi-epoch training ***multiple phase 2 operation before reiteration on phase 1
To see how more iterations of phase 2 before the new epoch run on phase 1 results in loss movements , a set of runs are done.
#### One phase 1 operation per six phase 2 operation

This case resulted in a continueing loss drop profile.
![](https://ahmetakman.github.io/assets/progress_reports/12_02_25/figures/multi_phase_2_12_02_2025.png)

#### Single phase 1 and many phase 2 operation
Similarly for a single phase 1 and many phase to (basically until explosion) following figure shows this case.
![](https://ahmetakman.github.io/assets/progress_reports/12_02_25/figures/multi_phase_2_100_epoch_12_02_2025.png)


### $\Delta W_{mn}$ Dynamics and selection of apical multiplier ranges
To get a glimpse of how and when $\Delta W_{mn}$ evolve to zero. A *toy* experiment is run with randomly initialized weights and inputs to a layer. Then via different random initalization of apical multipliers experiments are run. For example for a set of random apical multipliers $a \in [0, 1)$ it is observed that $\Delta W_{mn}$ always go to zero (that is not a certain statement just an observation after repetitive trials). Also for a set of random apical multipliers that are selected from standard gaussian distribution and it has been seen that in this case the convergence is not always there.

 
- Sample case where apical multiplier are *good* for weigth change 
  ![](https://ahmetakman.github.io/assets/progress_reports/12_02_25/figures/learning_rule_a_01.png)
- Sample case where apical multipliers are *bad* for weight change.
![a gaussian](https://ahmetakman.github.io/assets/progress_reports/12_02_25/figures/learning_rule_a_randn.png)


### Comparison of variance of apical multipliers per layer during multi-epoch training
To get an idea about if all apical multipliers move alltogether and generate diversity along a layer, their varience per layer is measured in two settings per layer.
- Regular training (one phase 1 and one phase 2 per epoch)
  ![varience of apical regular](https://ahmetakman.github.io/assets/progress_reports/12_02_25/figures/variance_regular_train.png)
- Other training setup (one phase 1 per six phase 2 per epoch)
  ![variance of apical multi phase 2](https://ahmetakman.github.io/assets/progress_reports/12_02_25/figures/variance_multi_phase_2_train.png)

- Even though it might not be exactly fair comparison when they are just one go runs, it can be said when we look at the variance in epoch 10, regular setup has larger variance for last layer and similar variances for other layers.
## Appendix
### Neuron Layer Model

$$r_n = a_n \phi\left(\sum Wx\right)$$
$$\mathcal{L}(W, a, x) = \sum_s l (\hat{y}, y(W, a, x))$$
Where the $\phi$ is the activation function, $l$ is the loss function, $\hat{y}$ is the predicted output, $y$ is the output of the network, $a$ is the apical input and $inp$ is the input to the network.

### Learning Rule

Using the notation from the overleaf document, for the weights between layer m and n, the learning rule is given by:
$$\Delta W_{mn} \propto r_m (\tilde{r}_n - r_n)$$
where $r_m$ is the rate of the presynaptic neuron and $\tilde{r_n}$ is the baseline rate of the postsynaptic neuron. Basically
$$a_n r_n = \tilde{r}_n$$

Using the notation from the paper under review:
$$\Delta w = \eta r_{pre} r_{post}^{baseline}(a^* - a^{baseline})$$
