## 用PaddlePaddle复现Fourier Neural Operator

本项目旨在用PaddlePaddle复现Fourier Neural Operator 一文^[1].

> [1]: Zongyi Li, Nikola B. Kovachki, Kamyar Azizzadenesheli, Burigede Liu, Kaushik Bhattacharya, Andrew M. Stuart, & Animashree Anandkumar (2020). Fourier Neural Operator for Parametric Partial Differential Equations arXiv: Learning.

这篇文章提出了一种新的神经算子模型——傅里叶神经算子，用于求解参数化的偏微分方程。与其他的用深度学习求解PDE的论文最大的不同在于，傅里叶神经算子直接在傅里叶空间中对积分核进行参数化，使得模型更加高效且具有更强的表达能力。此外，该模型还能够在零样本超分辨率方面取得卓越的表现。相对于传统的PDE求解器，傅里叶神经算子的速度提升了三个数量级，并且在固定分辨率下具有更高的准确性。

本项目的代码是基于[neuraloperator/neuraloperator: Learning in infinite dimension with neural operators. (github.com)](https://github.com/NeuralOperator/neuraloperator)的Paddle重写。


## 最简单的用例

在`eval.ipynb`中，我们展示了一个最小的用例，即用一个已经训练好的模型对示例数据集进行推理，并绘图。

## 从头开始训练一个模型

在`main.ipynb`中，我们展示了从0开始训练一个模型，计算Darcy flow。

## 本项目与NeuralOperator官方代码的差异

主要差异在以下方面：

- 本项目暂时不支持多GPU训练
- 本项目中的conv层与官方代码相比，暂时不支持Tucker或者Cp等分解方法，参数量会大于官方代码。