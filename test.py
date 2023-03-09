# %%
from neuralop.datasets.darcy import load_darcy_flow_small

# %%
train_loader, test_loaders, output_encoder = load_darcy_flow_small(
    n_train=1000, batch_size=32,
    test_resolutions=[16, 32], n_tests=[100, 50], test_batch_sizes=[32, 32],
)

# %%
train_dataset = train_loader.dataset

# %%
for res, test_loader in test_loaders.items():
    print('res: ', res)
    test_data = train_dataset[0]
    x = test_data['x']
    y = test_data['y']

    print(f'Testing samples for res {res} have shape {x.shape[1:]}')

# %%
import matplotlib.pyplot as plt
data = train_dataset[0]
x = data['x'] # [3, 16, 16]
y = data['y'] # [1, 16, 16]

print(f'Training sample have shape {x.shape[1:]}')


# Which sample to view
# index = 0

# data = train_dataset[index]
# x = data['x']
# y = data['y']
# fig = plt.figure(figsize=(7, 7))
# ax = fig.add_subplot(2, 2, 1)
# ax.imshow(x[0], cmap='gray')
# ax.set_title('input x')
# ax = fig.add_subplot(2, 2, 2)
# ax.imshow(y.squeeze())
# ax.set_title('input y')
# ax = fig.add_subplot(2, 2, 3)
# ax.imshow(x[1])
# ax.set_title('x: 1st pos embedding')
# ax = fig.add_subplot(2, 2, 4)
# ax.imshow(x[2])
# ax.set_title('x: 2nd pos embedding')
# fig.suptitle('Visualizing one input sample', y=0.98)
# plt.tight_layout()
# fig.show()

# %%
import paddle
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss

# %%
model = TFNO(n_modes=(16, 16), hidden_channels=32, projection_channels=64)
model.load_dict(paddle.load('model_trained.pdparams'))
n_params = count_params(model)
# print model summary
print(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()

print(model(paddle.randn((1, 3, 16, 16))))
# %%
optimizer = paddle.optimizer.Adam(learning_rate=5e-5 , parameters=model.parameters())

# cosine annealing scheduler
scheduler = paddle.optimizer.lr.CosineAnnealingDecay(learning_rate=5e-5, T_max=1000)

loss = LpLoss(d=2, p=2)
h1_loss = H1Loss(d=2)

train_loss = h1_loss
eval_losses = {"h1": h1_loss, "l2": loss}

print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()



trainer = Trainer(model, n_epochs=100,
                  mg_patching_levels=0,
                  wandb_log=False,
                  log_test_interval=3,
                  verbose=True)


trainer.train(train_loader, test_loaders,
              output_encoder,
              model, 
              optimizer,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

import pdb; pdb.set_trace()

test_samples = test_loaders[32].dataset

fig = plt.figure(figsize=(7, 7))
for index in range(3):
    data = test_samples[index]
    # Input x
    x = data['x']
    # Ground-truth
    y = data['y']
    # Model prediction
    out = model(x.unsqueeze(0).cuda()).cpu()

    ax = fig.add_subplot(3, 3, index*3 + 1)
    ax.imshow(x[0], cmap='gray')
    if index == 0:
        ax.set_title('Input x')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 2)
    ax.imshow(y.squeeze())
    if index == 0:
        ax.set_title('Ground-truth y')
    plt.xticks([], [])
    plt.yticks([], [])

    ax = fig.add_subplot(3, 3, index*3 + 3)
    ax.imshow(out.squeeze().detach().numpy())
    if index == 0:
        ax.set_title('Model prediction')
    plt.xticks([], [])
    plt.yticks([], [])

fig.suptitle('Inputs, ground-truth output and prediction.', y=0.98)
plt.tight_layout()
fig.show()
# wait for the plot to show
input('Press enter to continue...')

# save model
paddle.save(model.state_dict(), 'model_train_in_paddle.pdparams')
