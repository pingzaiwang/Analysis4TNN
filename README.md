# Analysis4TNN
This is the demo code for the paper titled 'Transformed Low-Rank Parameterization Can Help Robust Generalization for Tensor Neural Networks' presented at NeurIPS'23. We extend our deepest gratitude to Linfeng Sui and Xuyang Zhao for their indispensable support in implementing the Python code for t-NNs during the rebuttal phase.

If you have any questions or need assistance, please feel free to reach out to the first author, Andong Wang, at w.a.d@outlook.com. 
You are welcome to modify the code to suit your needs, and we kindly request that you consider citing our paper if you find it useful.
```
@inproceedings{
WLBJZZ2023AdvTNNs,
title={Transformed Low-Rank Parameterization Can Help Robust Generalization for Tensor Neural Networks},
author={Andong Wang, Chao Li, Mingyuan Bai, Zhong Jin, Guoxu Zhou, and Qibin Zhao},
booktitle={Thirty-seventh Conference on Neural Information Processing Systems (NeurIPS)},
year={2023},
url={https://openreview.net/forum?id=rih3hsSWx8}
}
```
## Experiment A.1 Effects of Exact Transformed Low-rank Weights on the Adversarial Generalization Gap
To validate the adversarial generalization bound in Theorem 6, we have conducted experiments on the MNIST dataset to explore the relationship between 
adversarial generalization gaps (AGP), weight tensor low-rankness, and training sample size. 
We consider binary classification of 3 and 7, with FSGM attacks of strength 20/255. The t-NN consists of three t-product layers and one FC layer, 
with weight tensor dimensions of 28×28×28 for $\underline{\textbf{{W}}}^{(1)}$, $\underline{\textbf{{W}}}^{(2)}$, 
and $\underline{{\textbf{W}}}^{(3)}$, and 784 for the FC weight $\textbf{{w}}$. 
Each MNIST image of size 28×28 is treated as a t-vector of 28×1×28, serving as an input example for the t-NN. 

While training a t-NN with low-tubal-rank parameterization, we have the following two points to emphasize. 
<ul>
<li> <strong>The low-tubal-rank constraint.</strong>
  To impose the low-tubal-rank constraint to a weight tensor $\underline{\textbf{{W}}}\in\mathbb{R}^{28\times 28 \times 28}$, 
we adopt the "factorization trick", that is we directly use $\underline{\textbf{{A}}}*_M\underline{\textbf{{B}}}$ instead of  $\underline{\textbf{{W}}}$ in the training process, with two tensors
$\underline{\textbf{{A}}}\in\mathbb{R}^{28\times r \times 28}$ 
and $\underline{\textbf{{B}}}\in\mathbb{R}^{r\times 28 \times 28}$. 

Thus, we train 
the proxy t-NN 
$$\textbf{w}^{\top} \sigma \bigg( \underline{\textbf{A}}^{(3)} *_M \underline{\textbf{B}}^{(3)} *_M
\sigma \big(\underline{{\textbf{A}}}^{(2)} *_M 
\underline{{\textbf{B}}}^{(2)} *_M 
\sigma(\underline{{\textbf{A}}}^{(1)} *_M 
\underline{{\textbf{B}}}^{(1)} *_M \underline{\textbf{x}})
\big) \bigg)$$
instead of the original t-NN 
$$\textbf{w}^{\top}\sigma\bigg(\underline{{\textbf{W}}}^{(3)} *_M 
\sigma\big(\underline{{\textbf{W}}}^{(2)} *_M 
\sigma(\underline{{\textbf{W}}}^{(1)} *_M \underline{\textbf{x}})
\big)\bigg).$$
</li>
<li> <strong>The upper bounds on F-norm of the weight tensors.</strong> 
  In our implementation, we add a regularization term $R(\underline{\textbf{W}})$ defined as 
$R(\underline{\textbf{W}}) = 0~\text{if}~ \|\underline{\textbf{W}}\|_F < B$, and 
$R(\underline{\textbf{W}}) = \lambda (\|\underline{\textbf{W}}\|_F - B)^2$, other wise. 
</li>
</ul>

You can run the demo code for Section A.1 "Effects of Exact Transformed Low-rank Weights on the Adversarial Generalization Gap" in the appendix as follows
```
# when the upper bound on the tubal-rank of each weight tensor is 4
python demo4ExpA1_AGPvsRank_R4.py
# when the upper bound on the tubal-rank of each weight tensor is 28, which is equivalent to the 
python demo4ExpA1_AGPvsRank_R28.py
```

## Experiment A.2 Implicit Bias of GF-based Adversarial Training to Approximately Transformed Low-rank Weight Tensors
We conducted numerical experiments to validate the following two theoretical statements pertaining to the analysis of GF-based adversarial training.

**Statement A.2.1**  According to Theorem 10, well-trained t-NNs under highly over-parameterized adversarial 
training with GF demonstrate approximately transformed low-rank parameters, given certain conditions.

**Statement A.2.2** As stated in Lemma 22 (in the supplementary material), the empirical adversarial risk tends to decrease to zero, 
and the F-norm of the weights tends to grow to infinity as $t$ approaches infinity.

In continuation of Experiment A.1, we also focus to binary classification tasks involving digits 3 and 7, 
utilizing FSGM adversarial attacks. The t-NN model is structured with three t-product layers and one fully connected layer, 
with weight tensor dimensions set to $D\times 28 \times 28$ for $\underline{\textbf{W}}^{(1)}$, 
$D \times D \times 28$ for $\underline{\textbf{W}}^{(2)}$ and 
$\underline{\textbf{W}}^{(3)}$, and $28D$ for the fully connected layer weight $\textbf{w}$. Our experiments involve setting the value of $D$ to 128 and 256, respectively, 
and we track the effective rank of each weight tensor, the empirical adversarial risk, and the F-norm of the weights as the number of epochs progresses. 
Since implementing gradient flow with infinitely small step size is impractical in real experiments, 
we opt for SGD with a constant learning rate and batch-size of 80, following the setting on fully connected layers in Ref. [27].

To varify the Statements A.2.1 and A.2.2, you can run the demo code for Section A.2 "Implicit Bias of GF-based Adversarial Training to Approximately Transformed Low-rank Weight Tensors" in the appendix as follows
```
# when D = 256 (by setting dim_latent = 256 in the source code)
python demo4ExpA2_ImplicitRegularization.py
```

## Experiment A.3 Additional Regularization for a Better Low-rank Parameterized t-NN
It is natural to ask: *instead of using adversarial training with GF in highly over-parameterized settings to train a approximately transformed low-rank t-NN, 
is it possible to apply some extra regularizations in training to achieve a better low-rank parameterization?*

Yes, it is possible to apply additional regularizations during training to achieve a better low-rank representation in t-NNs. 
Instead of relying solely on adversarial training with gradient flow in highly over-parameterized settings, 
these extra regularizations can potentially promote and enforce low-rankness in the network.

To validate the concern regarding the addition of an extra regularization term, we performed a preliminary experiment. 
In this experiment, we incorporated the tubal nuclear norm [45] as an explicit regularizer to induce low-rankness in the transformed domain. 
Specifically, we add the tubal nuclear norm regularization to the t-NN with three t-product layer $D=128$ in Experiment A.2 with a regularization parameter $0.01$, 
and keep the other settings the same as Experiment A.2. We explore how the stable ranks of tensor weights evolve with the epoch number with/without tubal nuclear norm 
regularization.

You can run the demo code for Section A.3 "Additional Regularization for a Better Low-rank Parameterized t-NN" in the appendix as follows
```
python demo4ExpA3_ExplicitRegularization.py
```

