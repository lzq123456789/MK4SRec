import torch
from torch import nn
import numpy as np
import math

def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    device = tau.device  # 获取tau的设备
    w = w.to(device)     # 确保w在同一设备上
    b = b.to(device)
    w0=w0.to(device)
    b0=b0.to(device)

    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        #print(w.shape, t1.shape, b.shape)
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    #print(v1.shape)
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class Time2Vec(nn.Module):
    def __init__(self, activation, hiddem_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, hiddem_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, hiddem_dim)
        self.fc1 = nn.Linear(hiddem_dim, 2)

    def forward(self, x):
        x = self.l1(x)
        x = self.fc1(x)
        return x




if __name__ == "__main__":
    sineact = SineActivation(1, 64)
    cosact = CosineActivation(1, 64)

    print(sineact(torch.Tensor([[7]])).shape)
    print(cosact(torch.Tensor([[7]])).shape)
    
