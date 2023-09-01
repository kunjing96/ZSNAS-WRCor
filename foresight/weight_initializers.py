import torch.nn as nn

def init_net(net, w_type, b_type):
    if w_type == 'none':
        pass
    elif w_type == 'xavier':
        net.apply(init_weights_vs)
    elif w_type == 'kaiming':
        net.apply(init_weights_he)
    elif w_type == 'zero':
        net.apply(init_weights_zero)
    elif w_type == 'N(0,1)':
        net.apply(init_weights_N01)
    else:
        raise NotImplementedError(f'init_type={w_type} is not supported.')

    if b_type == 'none':
        pass
    elif b_type == 'xavier':
        net.apply(init_bias_vs)
    elif b_type == 'kaiming':
        net.apply(init_bias_he)
    elif b_type == 'zero':
        net.apply(init_bias_zero)
    elif b_type == 'N(0,1)':
        net.apply(init_bias_N01)
    else:
        raise NotImplementedError(f'init_type={b_type} is not supported.')

def init_weights_vs(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)

def init_bias_vs(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if m.bias is not None:
            nn.init.xavier_normal_(m.bias)

def init_weights_he(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight)

def init_bias_he(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if m.bias is not None:
            nn.init.kaiming_normal_(m.bias)

def init_weights_zero(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        m.weight.data.fill_(.0)

def init_bias_zero(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if m.bias is not None:
            m.bias.data.fill_(.0)

def init_weights_N01(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.normal_(m.weight)

def init_bias_N01(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if m.bias is not None:
            nn.init.normal_(m.bias)

    
