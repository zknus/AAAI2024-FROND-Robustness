from .base_classes import ODEblock
import torch
from torch_geometric.utils import to_edge_index
from torch_sparse import coalesce
import torch_sparse
from .torchfde import fdeint

class ConstantODEblock_FRAC(ODEblock):
  def __init__(self,odefunc,  opt,  device, t=torch.tensor([0, 1])):
    super(ConstantODEblock_FRAC, self).__init__(odefunc,  opt,  device, t)
    self.odefunc = odefunc(opt['hidden_dim'], opt['hidden_dim'], opt,  device)
    self.device = device
    self.opt = opt
  def forward(self, x,adj):
    func = self.odefunc
    state = x

    if isinstance(adj, tuple):
      # print("adj is tuple")
      self.edge_index = adj[0]
      self.edge_attr = adj[1]
      # edege index to int 64
      self.edge_index = self.edge_index.to(torch.int64)
      index, value = coalesce(self.edge_index, self.edge_attr, m=state.size(0), n=state.size(0))
      adj = torch_sparse.SparseTensor(row=index[0], col=index[1], value=value)
      self.odefunc.adj = adj
    else:
      # print("adj is not tuple")
      self.odefunc.adj = adj
      self.edge_index, self.edge_attr = to_edge_index(adj)
    edge_index = self.edge_index
    edge_weight = self.edge_attr
    self.odefunc.edge_index = edge_index.to(self.device)
    self.odefunc.edge_weight = edge_weight.to(self.device)

    alpha = torch.tensor(self.opt['alpha_ode'])
    z = fdeint(func, state, alpha,t =self.opt['time'] ,step_size=self.opt['step_size'], method=self.opt['method'])
    return z

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"
