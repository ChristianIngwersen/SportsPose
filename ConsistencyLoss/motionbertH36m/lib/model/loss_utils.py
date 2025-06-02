"""This code was originally stolen from the internet by CIN from:
    https://github.com/wangleiphy/tensorgrad/blob/master/tensornets/adlib/svd.py

Licensed under Apache 2.0. See ../../DSTformer.py for license.
Modifications to the code post 06-12-2023 do not fall under the Apache 2.0 license."""

import torch
from torch import Tensor


def safe_inverse(x, epsilon=1e-12):
    return x / (x**2 + epsilon)


class SVD(torch.autograd.Function):
    @staticmethod
    def forward(self, A: Tensor) -> torch.return_types.svd:
        USV = torch.svd(A)
        self.save_for_backward(*USV)
        return USV

    @staticmethod
    def backward(self, dU: Tensor, dS: Tensor, dV: Tensor) -> Tensor:
        U, S, V = self.saved_tensors
        Vt = V.mT
        Ut = U.mT
        M = U.size(-2)
        N = V.size(-2)
        NS = S.size(-1)

        F = S[..., None, :] - S[..., None]
        F = safe_inverse(F)
        F.diagonal().fill_(0)

        G = S[..., None, :] + S[..., None]
        G.diagonal().fill_(torch.inf)
        G = 1 / G

        UdU = Ut @ dU
        VdV = Vt @ dV

        Su = (F + G) * (UdU - UdU.mT) / 2
        Sv = (F - G) * (VdV - VdV.mT) / 2

        dA = U @ (Su + Sv + torch.diag_embed(dS)) @ Vt
        if M > NS:
            dA = (
                dA
                + (torch.eye(M, dtype=dU.dtype, device=dU.device) - U @ Ut)
                @ (dU / S[..., None, :])
                @ Vt
            )
        if N > NS:
            dA = dA + (U / S[..., None, :]) @ dV.mT @ (
                torch.eye(N, dtype=dU.dtype, device=dU.device) - V @ Vt
            )
        return dA
