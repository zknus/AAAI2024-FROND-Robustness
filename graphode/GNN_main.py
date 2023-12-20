import torch
from torch import nn
import torch.nn.functional as F
from .base_classes import BaseGNN
from .model_configurations import set_block, set_function


# Define the GNN model.
class GNN_main(BaseGNN):
  def __init__(self, opt, dataset, device=torch.device('cpu')):
    super(GNN_main, self).__init__(opt, dataset, device)
    self.f = set_function(opt)
    block = set_block(opt)
    time_tensor = torch.tensor([0, self.T]).to(device)
    self.odeblock = block(self.f, opt, device, t=time_tensor).to(device)
    # self.prelu = nn.PReLU()
    self.bn = nn.BatchNorm1d(opt['hidden_dim'])
  def forward(self, x,adj):
    x = F.dropout(x, self.opt['input_dropout'], training=self.training)
    x = self.m1(x)
    if self.opt['use_mlp']:
      x = F.dropout(x, self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m11(F.relu(x)), self.opt['dropout'], training=self.training)
      x = F.dropout(x + self.m12(F.relu(x)), self.opt['dropout'], training=self.training)

    if self.opt['batch_norm']:
      x = self.bn_in(x)
    if 'grand' not in self.opt['function']:
      vt = x.clone()
      x = torch.cat([x, vt], dim=-1)
    self.odeblock.set_x0(x)
    z = self.odeblock(x,adj)
    if 'grand' not in self.opt['function']:
      z = z[:, self.opt['hidden_dim']:]

    # Activation.
    z = F.relu(z)
    if self.opt['fc_out']:
      z = self.fc(z)
      z = F.relu(z)

    # Dropout.
    z = F.dropout(z, self.opt['dropout'], training=self.training)
    # Decode each node embedding to get node label.
    z = self.m2(z)
    return z.log_softmax(dim=-1)
