import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss
import paddle
import numpy as np


y = paddle.to_tensor(np.load("y.npy"))
out = paddle.to_tensor(np.load("out.npy"))

h1_loss = H1Loss(d=2)

print(h1_loss(out, y))
