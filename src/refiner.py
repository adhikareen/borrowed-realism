from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
            nn.Conv3d(cout, cout, 3, padding=1),
            nn.GroupNorm(8, cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class Refiner(nn.Module):

    def __init__(self, code_dim=6, width=64):
        super().__init__()
        self.in_conv = nn.Conv3d(code_dim + 1, width, 3, padding=1)
        self.enc1 = ConvBlock(width, width)
        self.down = nn.Conv3d(width, width * 2, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.bott = ConvBlock(width * 2, width * 2)
        self.up = nn.ConvTranspose3d(width * 2, width, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.dec1 = ConvBlock(width * 2, width)
        self.out_conv = nn.Conv3d(width, code_dim, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, mix_codes, mask):
        x = torch.cat([mix_codes, mask], dim=1)
        h0 = self.in_conv(x)
        e1 = self.enc1(h0)
        d = self.down(e1)
        b = self.bott(d)
        u = self.up(b)
        dec = self.dec1(torch.cat([u, e1], dim=1))
        delta = self.out_conv(dec)
        return torch.tanh(mix_codes + delta)

    @staticmethod
    def count_params(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
