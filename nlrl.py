import torch

class NLRL_AO(torch.nn.Module):
    """
    This Class implements the NLRL with And- and Or-Calculation based on the paper https://arxiv.org/pdf/1907.00878.pdf.
    Init-Parameters
    ----------
    in_features : int
        Number of input features of the layer.
    out_features : int
        Number of the output features of the layer.
    eps : float, optional
        Number for mathematical stability in the log(.) calculation. This value is used as the minium Input-value and has no effect on the upper bound. The default is 0.0001.
    Forward
    -------
    input : torch.float32
        PyTorch float-tensor with the shape (B x in_features).
    return : torch.float32
        PyTorch float-tensor with the shape (B x out_features).
    """
    def __init__(self, in_features, out_features, eps=0.0001):
        super(NLRL_AO, self).__init__()
        self.IN_FEATURES=in_features
        self.OUT_FEATURES=out_features
        self.register_parameter("I",torch.nn.Parameter((torch.rand(1, in_features, out_features)-0.5)*1))
        self.register_parameter("A",torch.nn.Parameter((torch.rand(1, in_features, out_features)-0.5)*1))
        self.register_parameter("O",torch.nn.Parameter((torch.rand(1, out_features)-0.5)*1))
        self.I_INIT=self.I*1.0
        self.A_INIT=self.A*1.0
        self.O_INIT=self.O*1.0
        self.EPS=0.0001
    def __repr__(self):
        return "NLRL_AO(in_features=%i, out_features=%i)"%(self.IN_FEATURES, self.OUT_FEATURES)
    def forward(self, ins):
        BatchSize=ins.shape[0]
        ins=ins.unsqueeze(2).repeat(1, 1, self.OUT_FEATURES)
        OnesIn=torch.ones_like(ins, device=ins.device)
        # Sigmoids of Parameters
        YI=torch.sigmoid(self.I.repeat(BatchSize, 1, 1))
        ASig=torch.sigmoid(self.A.repeat(BatchSize, 1, 1))
        YO=torch.sigmoid(self.O.repeat(BatchSize, 1))
        # input negation
        XHat=(OnesIn-ins)*YI + ins*(OnesIn-YI)
        # And- and Or-Calculation
        and_out=self.AndFunc(XHat, ASig)
        or_out=self.OrFunc(XHat, ASig)
        # output negation
        outs=and_out*YO + or_out*(1-YO)
        return outs
    def OrFunc(self, ins, asig):
        return 1-self.AndFunc((1-ins), asig)
    def AndFunc(self, ins, asig):
        ins=torch.clamp(ins, self.EPS, 1.0- self.EPS)
        outs=torch.exp((torch.log(ins)*asig).sum(dim=1))
        #outs=torch.clamp(outs, self.EPS, 1.0- self.EPS)
        return outs

# '''===============
class InverseSig(torch.nn.Module):
    def __init__(self, *args):
        super(InverseSig, self).__init__()
    def forward(self, x, eps=0.0001):
        return torch.log((x+eps)/(1-x+eps))
# '''======
