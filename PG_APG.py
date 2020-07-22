import numpy as np
from matplotlib import pyplot as plt
import numpy.random as random
def generate_data(m,n,k):
    A = np.random.randn(m,n)
    x = np.random.rand(n,1)
    index_zero = np.random.choice(n,n-k,False)
    x[index_zero] = 0
    y = A.dot(x)
    return A,x,y

def soft(w,lam):
    w_ = np.concatenate((np.abs(w)-lam,np.zeros(w.shape)),1)
    return np.sign(w)*np.max(w_,axis=1,keepdims=True)

def PG(y,A,iter_times,prox=soft):
    '''
    proximal gradient algorithm
    proximal operator can be changed according to the problem
    '''
    eigen_w,_ = np.linalg.eig(np.dot(A.T,A))
    L = np.max(eigen_w)
    lam = 1
    x = np.random.rand(A.shape[1],1)
    x_hist = []
    x_hist.append(x)
    for i in range(iter_times):
        w = x - 1/L * A.T.dot(A.dot(x) - y)
        x = prox(w,lam/L)
        x_hist.append(x)
    return x_hist

def APG(y,A,iter_times,prox=soft):
    '''
    accelerated proximal gradient algorithm
    '''
    eigen_w,_ = np.linalg.eig(np.dot(A.T,A))
    L = np.max(eigen_w)
    lam = 1
    x = np.random.rand(A.shape[1],1)
    x_hist = []
    x_hist.append(x)
    t = 1
    p = x
    for i in range(iter_times):
        x_last = x_hist[-2] if i>0 else x
        beta = t - 1
        t = (1 + np.sqrt(1+4*t**2)) / 2
        beta = beta / t
        p = x + beta * (x - x_last)
        w = p - 1/L * A.T.dot(A.dot(p) - y)
        x = prox(w,lam/L)
        x_hist.append(x)
    return x_hist


def error_list(x,x_hist):
    '''
    x is the ground truth,x_hist is the list of the result
    error: l2 norm
    '''
    error_list = []
    for res in x_hist:
        error_list.append(np.linalg.norm(np.array(res)-x))
    return error_list

if __name__=="__main__":
    n = 800
    m = int(n * 0.5)
    k = int(m * 0.5)
    A, x, y = generate_data(m, n, k)
    iter_times = 3000
    x_hist_PG = PG(y, A,iter_times)
    x_hist_APG = APG(y, A,iter_times)
    errors_PG = error_list(x, x_hist_PG)
    errors_APG = error_list(x, x_hist_APG)
    plt.title("n=800,m=400,k=200")
    plt.xlabel("iter_times")
    plt.ylabel("error")
    plt.plot(errors_PG,label='PG')
    plt.plot(errors_APG,label='APG')
    plt.legend()
    plt.show()
