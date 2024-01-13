import numpy as np
import simufunc
import scipy.stats as st

def gen_cor_binomials(p1, p2, cor):
    x1 = np.random.binomial(1, p1)
    virt = np.random.rand(len(p1))
    x2 = np.random.binomial(1, p2) * (virt < cor) + np.random.binomial(1, 1-p2) * (virt >= cor)
    return np.column_stack((x1, x2))

def data_generative(N=100, s=1, type="normal", hypo="h0", xfun=None, yfun=None, cor=0.4, vx=5, vy=5):
    '''Generate H0 samples with continuous Z'''
    Z = np.random.uniform(0, 1, N)

    if xfun == None:
        Zx = Z
    else:
        Zx = xfun(Z)

    if yfun == None:
        Zy = Z
    else:
        Zy = yfun(Z)   

    # if hypo == "h0":
    #     if type == "normal":
    #         X = np.random.normal(loc=0, scale=vx, size=N) + Zx
    #         Y = np.random.normal(loc=0, scale=vy, size=N) + Zy
    #     elif type == "skewed_normal":
    #         X = st.skewnorm.rvs(a=-5, loc=0, scale=vx, size=N) + Zx
    #         Y = st.skewnorm.rvs(a=-5, loc=0, scale=vy, size=N) + Zy
    #     else:
    #         raise ValueError("Non-existing distribution type!")
    
    # elif hypo == "h1":
    #     Zxy = np.column_stack((Zx, Zy))
    #     if type == "normal":
    #         data = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[vx, np.sqrt(vx*vy)*cor],[np.sqrt(vx*vy)*cor, vy]], size=1) for i in range(Zxy.shape[0])])
    #         X = data[:, 0]
    #         Y = data[:, 1]
    #     elif type == "skewed_normal":
    #         skewness = [5, -5]  # Skewness vector
    #         normal_samples = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[vx*0.8, np.sqrt(vx*vy)*cor*0.8],[np.sqrt(vx*vy)*cor*0.8, vy*0.8]], size=1) for i in range(Zxy.shape[0])])
    #         skew_samples = st.skewnorm.rvs(skewness, loc=0, scale=[vx*0.2, vy*0.2], size=(N, 2))
    #         skewed_normal_samples = normal_samples + skew_samples
    #         X = skewed_normal_samples[:, 0]
    #         Y = skewed_normal_samples[:, 1]
    #     else:
    #         raise ValueError("Non-existing distribution type!")
    # else:
    #     raise ValueError("Non-existing Hypothesis type!")
    Zxy = np.column_stack((Zx, Zy))
    if type == "normal":
        data = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[vx, np.sqrt(vx*vy)*cor],[np.sqrt(vx*vy)*cor, vy]], size=1) for i in range(Zxy.shape[0])])
        X = data[:, 0]
        Y = data[:, 1]
    elif type == "skewed_normal":
        skewness = [5, -5]  # Skewness vector
        normal_samples = np.array([st.multivariate_normal.rvs(mean=Zxy[i,], cov=[[vx*0.8, np.sqrt(vx*vy)*cor*0.8],[np.sqrt(vx*vy)*cor*0.8, vy*0.8]], size=1) for i in range(Zxy.shape[0])])
        skew_samples = st.skewnorm.rvs(skewness, loc=0, scale=[vx*0.2, vy*0.2], size=(N, 2))
        skewed_normal_samples = normal_samples + skew_samples
        X = skewed_normal_samples[:, 0]
        Y = skewed_normal_samples[:, 1]
    elif type == "binomial":
        data = gen_cor_binomials(Zx, Zy, cor)
        X = data[:, 0]
        Y = data[:, 1]
    else:
        raise ValueError("Non-existing distribution type!")

    return X, Y, Z

def experiment(i, N=100, M=10, type="normal", \
    xfun=None, yfun=None, perm="y", cor=0.8, vx=5, vy=5):
    if i%5 == 0:
        print(i)
    if cor == 0:
        hypoo = "h0"
    else:
        hypoo = "h1"
    X, Y, Z = data_generative(N=N, s=i, type=type, hypo=hypoo, yfun=yfun, xfun=xfun, cor=cor, vx=vx, vy=vy)
    p1, p2, p3, p4, p5, p6, p7 = simufunc.LPT(X=X, Y=Y, Z=Z, B = 100, M = M, perm=perm)
    alpha = 0.05
    return int(p1 <= alpha), int(p2 <= alpha), int(p3 <= alpha), int(p4 <= alpha), int(p5 <= alpha), int(p6 <= alpha), int(p7 <= alpha)
    

def experiment2(i, N=100, M=10, type="normal", hypo="h1", \
    xfun=None, yfun=None, perm="y", cor=0.8, vx=5, vy=5):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative(N=N, s=i, type=type, hypo=hypo, yfun=yfun, xfun=xfun, cor=cor, vx=vx, vy=vy)
    p1, p2 = simufunc.LPT_resid(X, Y, Z, B = 100, M = M, perm=perm)
    alpha = 0.05
    return int(p1 <= alpha), int(p2 <= alpha)

def Z_to_Y(V):
    return 4 * (V**2) - 4*V - 5

def Z_to_Y2(Z):
    return (np.log(Z+1) /np.log(2))-3

def Z_to_Y3(Z):
    return 2/(Z+1)

def Z_to_Y4(Z):
    return Z**2

def Z_to_Y5(Z):
    return 1 - Z**3

def experiment3(i, N=100, M=10, type="normal", hypo="h1", \
    xfun=None, yfun=None, perm="y", cor=0.8, vx=5, vy=5):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative(N=N, s=i, type=type, hypo=hypo, yfun=yfun, xfun=xfun, cor=cor, vx=vx, vy=vy)
    p1, p2, p3, p4, p5, p6 = simufunc.LPT_partial(X, Y, Z, B = 100, M = M, perm=perm)
    alpha = 0.05
    return int(p1 <= alpha), int(p2 <= alpha), int(p3 <= alpha), int(p4 <= alpha), int(p5 <= alpha), int(p6 <= alpha)

def experiment4(i, N=100, M=10, type="binomial", hypo="h1", \
    xfun=None, yfun=None, perm="y", cor=0, vx=5, vy=5):
    if i%5 == 0:
        print(i)
    X, Y, Z = data_generative(N=N, s=i, type=type, hypo=hypo, yfun=yfun, xfun=xfun, cor=cor, vx=vx, vy=vy)
    p1, p2, p3, p4, p5 = simufunc.LPT(X, Y, Z, B = 100, M = M, perm=perm)
    alpha = 0.05
    return int(p1 <= alpha), int(p2 <= alpha), int(p3 <= alpha), int(p4 <= alpha), int(p5 <= alpha)