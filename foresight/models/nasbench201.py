import torch.nn as nn

from .nasbench201_ops import NormalCell, ReductionCell


class NASBench201Network(nn.Module):

  def __init__(self, C, N, genotype, num_classes, bn=True):
    super(NASBench201Network, self).__init__()
    self._C               = C
    self._layerN          = N

    if bn:
      self.stem = nn.Sequential(
                      nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False),
                      nn.BatchNorm2d(C))
    else:
      self.stem = nn.Sequential(
                      nn.Conv2d(3, C, kernel_size=3, padding=1, bias=False))
  
    layer_channels   = [C    ] * N + [C*2 ] + [C*2  ] * N + [C*4 ] + [C*4  ] * N    
    layer_reductions = [False] * N + [True] + [False] * N + [True] + [False] * N

    C_prev = C
    self.cells = nn.ModuleList()
    for index, (C_curr, reduction) in enumerate(zip(layer_channels, layer_reductions)):
      if reduction:
        cell = ReductionCell(C_prev, C_curr, 2, True, bn=bn)
      else:
        cell = NormalCell(genotype, C_prev, C_curr, 1, bn=bn)
      self.cells.append( cell )
      C_prev = cell.out_dim
    self._Layer= len(self.cells)

    if bn:
      self.lastact = nn.Sequential(nn.BatchNorm2d(C_prev), nn.ReLU(inplace=True))
    else:
      self.lastact = nn.Sequential(nn.ReLU(inplace=True))
    self.global_pooling = nn.AdaptiveAvgPool2d(1)
    self.classifier = nn.Linear(C_prev, num_classes)

  def get_message(self):
    string = self.extra_repr()
    for i, cell in enumerate(self.cells):
      string += '\n {:02d}/{:02d} :: {:}'.format(i, len(self.cells), cell.extra_repr())
    return string

  def extra_repr(self):
    return ('{name}(C={_C}, N={_layerN}, L={_Layer})'.format(name=self.__class__.__name__, **self.__dict__))

  def forward(self, inputs, pre_GAP=False):
    feature = self.stem(inputs)
    for i, cell in enumerate(self.cells):
      feature = cell(feature)

    feature = self.lastact(feature)
    out = self.global_pooling( feature )
    out = out.view(out.size(0), -1)
    logits = self.classifier(out)

    if pre_GAP:
        return feature
    else:
        return logits
