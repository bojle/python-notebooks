#!/usr/bin/env python
# coding: utf-8

# In[40]:

import numpy as np
from math import ceil

# In[154]:

class ctx:
    # TODO: needs reworking with a compact syntax
    def __init__(self, ifmap, kernels, stride=1, padding=0):
        self.N, self.C, self.IH, self.IW = ifmap.shape
        self.KH, self.KW = kernels[0].shape
        self.S, self.P = (stride, padding)
        self.KN = kernels.shape[0]
        self.Hout = ceil((self.IW - self.KW)/stride) + 1
        self.Wout = ceil((self.IH - self.KH)/stride) + 1


# In[159]:


def _conv2d(ctx, ifmap, kernel):
    """ conv2d helper - conv ifmap[0,0,i,j] with kernel """
    out = np.zeros(ctx.Hout * ctx.Wout)
    out_index = 0
    for i in range(ctx.Hout):
        for j in range(ctx.Wout):
            for ii in range(ctx.KH):
                for jj in range(ctx.KW):
                     out[out_index] = out[out_index] + (ifmap[0,0,ii+i,jj+j] * kernel[ii, jj])                       
            out_index = out_index + 1
    return out.reshape(ctx.Hout, ctx.Wout)
    
def conv2d(ctx, ifmap, kernels):
    out = np.empty((ctx.KN, ctx.Hout, ctx.Wout))
    for i in range(ctx.KN):
        out[i] = _conv2d(ctx, ifmap, kernels[i])
    return out

def get_im2col_indices(ctx):
    # First figure out what the size of the output should be
    N, C, H, W = (ctx.N, ctx.C, ctx.IH, ctx.IW)
    field_height, field_width = (ctx.KH, ctx.KW)
    stride, padding = (ctx.S, ctx.P)
    out_height, out_width = (ctx.Hout, ctx.Wout)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k, i, j)


def im2col_indices(ctx, x):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    field_height, field_width = (ctx.KH, ctx.KW)
    p = ctx.P
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(ctx)
    cols = x_padded[:, k, i, j]
    C = ctx.C
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols

def conv2d_im2col(ctx, ifmap, kernels):
    """ Only conv kernel[0] for now """
    out = np.empty((ctx.KN, ctx.Hout, ctx.Wout))
    for i in range(ctx.KN):
        out1 = kernels[i].flatten().reshape(1, ctx.KH * ctx.KW)
        out2 = im2col_indices(ctx, ifmap)
        out[i] = (out1 @ out2).reshape(ctx.Hout, ctx.Wout)
    return out


# In[160]:


if __name__ == "__main__":
    vgg_image = np.random.random((1,3,224,224))

    ifm = np.array([[1,2,3,2],[2,3,2,1],[1,2,1,2],[2,1,3,2]]).reshape(1,1,4,4)
    kernels = np.array([[1,2,2],[1,2,3],[4,2,3]]).reshape(1,3,3)
    ctxo = ctx(ifm, kernels, stride=1, padding=0)
    print(f"Conv2d without im2col: {conv2d(ctxo, ifm, kernels)}")
    print(f"Conv2d with im2col: {conv2d_im2col(ctxo, ifm, kernels)}")

# In[ ]:
