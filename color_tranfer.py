from skimage.io import imread,imsave
import os
import numpy as np
from functools import reduce
class MKL():
    def __call__(self,img1,img2):
        assert img1.shape[2] == 3 and img2.shape[2] == 3
        im1 = np.reshape(img1 / 255,(-1,img1.shape[2])).astype(np.float32)
        im2 = np.reshape(img2 / 255,(-1,img2.shape[2])).astype(np.float32)

        cov1 = np.cov(im1, rowvar=False)
        cov2 = np.cov(im2, rowvar=False)

        T = self.get_T(cov1,cov2)
        mean = np.mean(im1,0,keepdims=True)
        mim1 = np.tile(np.mean(im1,0,keepdims=True),(im1.shape[0],1))
        mim2 = np.tile(np.mean(im2,0,keepdims=True),(im2.shape[0],1))

        #mim1 = np.repeat(np.mean(im1,0,keepdims=True),im1.shape[0],0)
        #mim2 = np.repeat(np.mean(im2,0,keepdims=True),im2.shape[0],0)

        res = np.dot((im1 - mim1), T) + mim1
        res = np.reshape(res,img1.shape)
        return res

    def get_T(self, cov1, cov2):
        n = cov1.shape[0]
        w,v = np.linalg.eig(cov1)
        w[w < 0] = 0
        da = np.diag(np.sqrt(w + 1e-15))
        C = reduce(np.dot,[da,v.T,cov2,v,da])
        w2,v2 = np.linalg.eig(C)
        w2[w2<0] = 0
        dc = np.diag(np.sqrt(w2 + 1e-15))
        da_inv = np.diag(1 / np.diag(da))
        T = reduce(np.dot,[v,da_inv,v2,dc,v2.T,da_inv,v.T])
        return T
if __name__ == "__main__":
    im1 = imread('scotland_house.png')
    im2 = imread('scotland_plain.png')
    #mkl
    mkl = MKL()
    res_mkl = mkl(im1,im2)
    imsave('res_mkl.png',res_mkl)