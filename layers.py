
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functions import M_transform

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn

class tProdLayerDCTBatch(nn.Module):
    def __init__(self, D, d, c):
        super(tProdLayerDCTBatch, self).__init__()

        # Weight tensor
        self.weight = nn.Parameter(torch.Tensor(D, d, c))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return t_product_dct_batch(self.weight, x)


def t_product_dct_batch(A, B):
    # A: [D, d, c]
    # B: [batch_size, d, 1, c]
    # Output: [batch_size, D, 1, c]
    assert A.shape[1] == B.shape[1] and A.shape[2] == B.shape[3], f"Incompatible shapes for t-product: A.shape = {A.shape}, B.shape= {B.shape}"
    # Repeat weight tensor A to match batch size of B
    M = M_transform()
    A = M.DCT(A)
    B = M.DCT(B)
    C = torch.zeros(B.shape[0],B.shape[3],A.shape[0],B.shape[2]).to(device)
    A = A.permute(2,0,1)
    B = B.permute(0,3,1,2)
    for i in range(B.shape[0]):
             C[i,:,:,:] = torch.bmm(A,B[i,:,:,:])
    C = C.permute(0,2,3,1)
    C = C.to(device)
    return M.inv_DCT(C)

# def f_slice_product_batch_matmul(A,B):
#     A = A.permute(2,0,1)
#     B = B.permute(0,3,1,2)
#     C = torch.matmul(A,B)
#     return C.permute(0,2,3,1)

def t_product_dct(A, B):
    # A: [D, d, c]
    # B: [batch_size, d, 1, c]
    # Output: [batch_size, D, 1, c]
    assert A.shape[1] == B.shape[0] and A.shape[2] == B.shape[2], f"Incompatible shapes for t-product: A.shape = {A.shape}, B.shape= {B.shape}"
    # Repeat weight tensor A to match batch size of B
    M = M_transform()
    A = M.DCT(A)
    B = M.DCT(B)
    A_t = A.permute(2,0,1)
    B_t = B.permute(2,0,1)
    C = torch.zeros(B.shape[2],A.shape[0],B.shape[1])
    C = torch.bmm(A_t,B_t)
    C = C.permute(1,2,0).to(device)
    return M.inv_DCT(C)

def t_spectral_norm(A):
    # A: [D, d, c]
    M = M_transform()
    A = M.DCT(A)
    v_spec_norms = torch.zeros(A.shape[2])
    for i in range(A.shape[2]):
          v_spec_norms[i] = torch.linalg.norm(A[:,:,i].squeeze(),2,dim=(-2,-1))
    return v_spec_norms

def ratio_spec2frob(A):
    norm_f = torch.norm(A)
    norm_f = norm_f.to(device)
    v_spec_norms = t_spectral_norm(A)
    norm_sp = max(v_spec_norms)
    norm_sp = norm_sp.to(device)
    return norm_sp/(norm_f+1e-10)
