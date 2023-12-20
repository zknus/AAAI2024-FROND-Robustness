import torch
from torch import nn
from torch_geometric.nn.conv import MessagePassing
from .utils import Meter
class ODEblock(nn.Module):
  def __init__(self, odefunc, opt, device, t):
    super(ODEblock, self).__init__()
    self.opt = opt
    self.t = t
    self.odefunc = odefunc(opt['hidden_dim'],  opt['hidden_dim'], opt, device)



  def set_x0(self, x0):
    self.odefunc.x0 = x0.clone().detach()
    # self.reg_odefunc.odefunc.x0 = x0.clone().detach()



  def reset_tol(self):
    self.atol = 1e-7
    self.rtol = 1e-9
    self.atol_adjoint = 1e-7
    self.rtol_adjoint = 1e-9

  def set_time(self, time):
    self.t = torch.tensor([0, time]).to(self.device)

  def __repr__(self):
    return self.__class__.__name__ + '( Time Interval ' + str(self.t[0].item()) + ' -> ' + str(self.t[1].item()) \
           + ")"


class ODEFunc(MessagePassing):

  # currently requires in_features = out_features
  def __init__(self, opt,  device):
    super(ODEFunc, self).__init__()
    self.opt = opt
    self.device = device
    self.edge_index = None
    self.edge_weight = None
    self.attention_weights = None
    self.attention_weights_2 = None
    self.alpha_train = nn.Parameter(torch.tensor(0.0))#nn.Parameter(torch.zeros(opt['hidden_dim']))
    self.beta_train = nn.Parameter(torch.tensor(0.0))
    self.beta_train2 = nn.Parameter(torch.tensor(0.0))
    self.x0 = None
    self.nfe = 0
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))

  def __repr__(self):
    return self.__class__.__name__


class BaseGNN(MessagePassing):
  def __init__(self,  opt,num_features, device=torch.device('cpu')):
    super(BaseGNN, self).__init__()
    self.opt = opt
    self.T = opt['time']
    self.num_classes = opt['num_classes']
    self.num_features = num_features
    self.device = device
    self.fm = Meter()
    self.bm = Meter()
    self.m1 = nn.Linear(num_features, opt['hidden_dim'])
    if self.opt['use_mlp']:
      self.m11 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
      self.m12 = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
    self.hidden_dim = opt['hidden_dim']
    if opt['fc_out']:
      self.fc = nn.Linear(opt['hidden_dim'], opt['hidden_dim'])
    self.m2 = nn.Linear(opt['hidden_dim'], self.num_classes)
    if self.opt['batch_norm']:
      self.bn_in = torch.nn.BatchNorm1d(opt['hidden_dim'],affine=False)
      self.bn_out = torch.nn.BatchNorm1d(opt['hidden_dim'])

    # self.regularization_fns, self.regularization_coeffs = create_regularization_fns(self.opt)

  def getNFE(self):
    # return self.odeblock.odefunc.nfe + self.odeblock.reg_odefunc.odefunc.nfe
    return self.odeblock.odefunc.nfe

  def resetNFE(self):
    self.odeblock.odefunc.nfe = 0
    # self.odeblock.reg_odefunc.odefunc.nfe = 0

  def reset(self):
    self.m1.reset_parameters()
    self.m2.reset_parameters()

  def __repr__(self):
    return self.__class__.__name__
