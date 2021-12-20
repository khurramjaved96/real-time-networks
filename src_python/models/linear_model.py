import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, step_size, bias=False):
      super(LinearModel, self).__init__()
      self.all_synapses = []
      self.imprinted_features = []
      self.step_size = step_size
      self.fc = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, input):
        out = self.fc(input)
        return out
