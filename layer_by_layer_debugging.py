import paddle
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss

model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64)

# fill every parameter with 0.1 for debugging

for param in model.parameters():
    param.set_value(paddle.ones_like(param) * 0.1)

n_params = count_params(model)
print(f'\nOur model has {n_params} parameters.')
x = paddle.ones([1, 3, 16, 16])

y = model(x)

print(y)

h1_loss = H1Loss(d=2)
