import numpy as np
from matplotlib import pyplot as plt

def generate_data(m,n,k):
    A = np.random.rand(m,n)
    x = np.random.rand(n,1)
    index_random_zero = np.random.choice(n,n-k,False)
    x[index_random_zero]=0#generate sparse x
    y = A.dot(x)
    return y,A,x

def subgradient(x):
    gradient = np.sign(x)
    return gradient

def projection(A,y):
    I = np.identity(A.shape[1])
    A_H = A.conj().T
    AA_H_inv = np.linalg.inv(A.dot(A_H))
    T = I - A_H.dot(AA_H_inv).dot(A)
    x_ = A_H.dot(AA_H_inv).dot(y)
    return T,x_

def PSD_solver(A,y,times):
    T,x_ = projection(A,y)
    x = np.zeros((A.shape[1],1))
    for i in range(1,times+1):
        x = x_ + T.dot(x - 1/i * subgradient(x))
    return x

if __name__ == "__main__":
    m = 200
    n = 500
    iter_times = 2000
    thred = 0.01
    exp_times = 100
    success_rate_res = []
    for k in range(500):
        print("now k is %d"%k)
        success_rate = 0
        #test 50 times and get the average rate
        for exp_time in range(exp_times):
            y, A, x_gt=generate_data(m,n,k)
            T,x_ = projection(A,y)
            x_pred = PSD_solver(A,y,iter_times)
            error = np.abs(x_gt - x_pred)
            if np.mean(error<thred) == 1:
                success_rate += 1
                #success_rate += np.mean(error<thred)
        success_rate = success_rate / exp_times
        success_rate_res.append(success_rate)
        print("k=%d,success rate=%f"%(k,success_rate))
    #draw the fig
    plt.plot(list(range(n)),success_rate_res)
    plt.show()
    
    
