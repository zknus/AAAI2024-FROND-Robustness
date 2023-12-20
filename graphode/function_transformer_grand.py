import torch
from torch import nn
import torch_sparse


from .function_transformer_attention import SpGraphTransAttentionLayer


from torch_geometric.nn.conv import MessagePassing

class ODEFunc(MessagePassing):

  # currently requires in_features = out_features
  def __init__(self, opt, device):
    super(ODEFunc, self).__init__()
    self.opt = opt
    self.device = device
    self.edge_index = None
    self.edge_weight = None
    self.attention_weights = None
    self.alpha_train = nn.Parameter(torch.tensor(0.0))
    self.beta_train = nn.Parameter(torch.tensor(0.0))
    self.x0 = None
    self.nfe = 0
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))


class ODEFuncTransformerAtt_GRAND(ODEFunc):

  def __init__(self, in_features, out_features, opt,  device):
    super(ODEFuncTransformerAtt_GRAND, self).__init__(opt,  device)

    self.multihead_att_layer = SpGraphTransAttentionLayer(in_features, out_features, opt,
                                                          device, edge_weights=self.edge_weight).to(device)

  def multiply_attention(self, x, attention, v=None):
    if self.opt['mix_features']:
      vx = torch.mean(torch.stack(
        [torch_sparse.spmm(self.edge_index, attention[:, idx], v.shape[0], v.shape[0], v[:, :, idx]) for idx in
         range(self.opt['heads'])], dim=0),
        dim=0)
      ax = self.multihead_att_layer.Wout(vx)
    else:
      mean_attention = attention.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # t is needed when called by the integrator
    attention, values = self.multihead_att_layer(x, self.edge_index)
    ax = self.multiply_attention(x, attention, values)

    if not self.opt['no_alpha_sigmoid']:
      # print('alpha sigmoid')
      alpha = torch.sigmoid(self.alpha_train)
    else:
      alpha = self.alpha_train
    if not self.opt['no_alpha']:
      # print('alpha:', alpha )
      f = alpha * (ax - x)
    else:
      f = self.opt['weightax'] * (ax - x)
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0
    # f = torch.relu(f)
    return f

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'




