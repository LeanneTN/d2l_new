import torch
import torchDrug as td
from torchdrug import datasets
from torchdrug import core, models, tasks
from torch import utils
from torch.utils.data import Dataset, DataLoader

dataset = datasets.ClinTox("./data/")
lengths = [int(0.8 * len(dataset)), int(1.0 * len(dataset))]
lengths += [len(dataset) - sum(lengths)]
train_set, valid_set, test_set = utils.data.random_split(dataset, lengths)
model = models.GIN(input_dim=dataset.node_feature_dim, hidden_dims=[256, 256, 256, 256],
                   short_cut=True, batch_norm=True, concat_hidden=True)
task = tasks.PropertyPrediction(model, task=dataset.tasks, criterion="bce", metric=("auprc", "auroc"))
optimizer = torch.optim.Adam(task.parameters(), lr=1e-3)
solver = core.Engine(task, train_set, valid_set, test_set, optimizer, batch_size=1024, gpus=[0])
solver.train(num_epoch=100)

