from skimage.io import imread,imsave
import os
import numpy as np
from functools import reduce
from scipy.linalg import orth
import cv2
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

class IDT():
    def __init__(self, iterations):
        self.iterations = iterations
    def __call__(self, img1, img2):
        assert img1.shape[2] == 3 and img2.shape[2] == 3
        d1 = np.reshape(img1 / 255, (-1,3)).astype(np.float32).T
        d2 = np.reshape(img2 / 255, (-1,3)).astype(np.float32).T
        R = []
        R.append(np.array([[1,0,0],[0,1,0],[0,0,1],[2/3,2/3,-1/3],[2/3,-1/3,2/3],[-1/3,2/3,2/3]]))
        for i in range(self.iterations-1):
            R.append(R[0].dot(orth(np.random.randn(3,3))))
        dr = self.pdf_transfer(d1,d2,R)
        
        res = np.reshape(dr.T,img1.shape)
        return res

    def pdf_transfer(self,d1,d2,R):
        relaxation = 1
        for i in range(self.iterations):
            Rot = R[i]
            nb_pro = Rot.shape[0]
            d1r = np.dot(Rot,d1)
            d2r = np.dot(Rot,d2)
            d1r_ = np.zeros(d1r.shape)
            for j in range(nb_pro):
                datamin = min(d1r[j,:].min(),d2r[j,:].min()) - 1e-6
                datamax = max(d1r[j,:].max(),d2r[j,:].max()) + 1e-6
                u = np.asarray(range(300)) / 299 * (datamax - datamin) + datamin
                u_edge = np.zeros(u.shape[0]+1)
                u_edge[0] = u[0] - (u[2]-u[1])/2
                u_edge[-1] = u[-1] + (u[-1] - u[-2])/2
                u_edge[1:-1] = u[:-1] + np.diff(u)/2

                p1r,_ = np.histogram(d1r[j,:], u_edge)
                p2r,_ = np.histogram(d2r[j,:], u_edge)

                f = self.pdf_transfer1D(p1r,p2r)
                d1r_[j,:] = (np.interp(d1r[j,:],u,f.T)-1) / 299*(datamax-datamin) + datamin
            d1 = np.linalg.pinv(Rot).dot(d1r_ - d1r) + d1
        return d1
    
    def pdf_transfer1D(self, px, py):
        nbins = max(px.shape)
        eps = 1e-6
        PX = np.cumsum(px+eps)
        PX = PX/PX[-1]

        PY = np.cumsum(py+eps)
        PY = PY/PY[-1]

        f = np.interp(PX, PY, list(range(nbins)))
        f[PX<=PY[0]] = 0
        f[PX>=PY[-1]] = nbins - 1
        return f

class CT():
    def __call__(self,img0,img1):
        im0 = cv2.cvtColor(img0,cv2.COLOR_RGB2LAB)
        im1 = cv2.cvtColor(img1,cv2.COLOR_RGB2LAB)
        mean0, std0 = cv2.meanStdDev(im0)
        mean1, std1 = cv2.meanStdDev(im1)

        mean0_all = np.tile(mean0.T,(im0.shape[0], im0.shape[1], 1))
        std0_all = np.tile(std0.T,(im0.shape[0], im0.shape[1], 1))

        mean1_all = np.tile(mean1.T,(im1.shape[0], im1.shape[1], 1))
        std1_all = np.tile(std1.T,(im1.shape[0], im1.shape[1], 1))

        res = np.round(((im0 - mean0_all) * (std1_all/std0_all)) + mean1_all)
        res = np.clip(res,0,255).astype(np.uint8)
        res = cv2.cvtColor(res,cv2.COLOR_LAB2RGB)
        return res
    


if __name__ == "__main__":
    # im1 = imread('scotland_house.png')
    # im2 = imread('scotland_plain.png')
    im1 = imread('1.jpg')
    im2 = imread('2.jpg')
    #lab
    ct = CT()
    res_fast = ct(im1,im2)
    imsave('res_fast.png',res_fast)
    #mkl
    mkl = MKL()
    res_mkl = mkl(im1,im2)
    imsave('res_mkl.png',res_mkl)
    #IDT
    idt = IDT(10)
    res_idt = idt(im1,im2)
    imsave('res_idt.png',res_idt)