import multiprocessing as mp
import numpy as np
import pandas as pd
import statsmodels.api as sm
import scipy.stats as st
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, sys.path[0]+"/../")
import expfunc

def _perm(V, G, M):
    new_V = V.copy()
    for g in range(M):
        inds = np.where(G==g)[0]
        new_inds = inds.copy()
        np.random.shuffle(new_inds)
        new_V[new_inds] = new_V[inds]
    return new_V

def compute_resid(Z, Y):
    Z1 = sm.add_constant(Z)
    mod = sm.OLS(Y, Z1).fit()
    return mod.resid

def corr(X, Y):
    return np.abs(np.corrcoef(X, Y)[0, 1])

def compute_G(Z, M):
    G = pd.qcut(Z, M, labels=[i for i in range(M)])
    Z_means = [np.mean(Z[G == g]) for g in range(M)]
    return G, np.array([Z_means[G[i]] for i in range(Z.shape[0])])

def perm_decomp(xfunc, vx, yfunc, vy):
    xy_percentile = np.zeros(200)
    cor_percentile = np.zeros(200)
    sdx_percentile = np.zeros(200)
    div_percentile = np.zeros(200)

    def _perm_var(xfunc, vx, yfunc, vy):
        X, Y, Z = expfunc.data_generative(N=100, s=1, type="normal", hypo="h0", xfun=xfunc, yfun=yfunc, cor=0.4, vx=vx, vy=vy)
        X_resid_o = compute_resid(Z, X)
        Y_resid_o = compute_resid(Z, Y)
        # print(xfunc, vx, "=======")
        # print("X", np.mean(X_resid_o), np.std(X_resid_o))
        # print("Y", np.mean(Y_resid_o), np.std(Y_resid_o))
        #T_sam = np.dot(X_resid_o, X_resid_o)
        T_sam = np.std(X_resid_o)
        xy_sam = np.abs(np.dot(X_resid_o, Y_resid_o))
        cor_sam = np.abs(np.corrcoef(X_resid_o, Y_resid_o)[0, 1])
        div_sam = xy_sam / (T_sam)

        B = 1000
        m = 50
        T_per_z = np.zeros(B)
        xy_per = np.zeros(B)
        cor_per = np.zeros(B)
        div_per = np.zeros(B)

        G, _ = compute_G(Z, m)
        for i in range(B):
            new_X = _perm(X, G, m)
            X_resid_temp = compute_resid(Z, new_X)
            T_per_z[i] = np.std(X_resid_temp)
            xy_per[i] = np.abs(np.dot(X_resid_temp, Y_resid_o))
            cor_per[i] = np.abs(np.corrcoef(X_resid_temp, Y_resid_o)[0, 1])
            div_per[i] = xy_per[i] / (T_per_z[i])

        p1 = (np.sum(T_sam > T_per_z) + 0.5 * np.sum(T_sam == T_per_z)) / B
        p2 = (np.sum(xy_sam > xy_per) + 0.5 * np.sum(xy_sam == xy_per)) / B
        p3 = (np.sum(cor_sam > cor_per) + 0.5 * np.sum(cor_sam == cor_per)) / B
        p4 = (np.sum(div_sam > div_per) + 0.5 * np.sum(div_sam == div_per)) / B
        return p1, p2, p3, p4
    
    for t in range(200):
        print(t)
        p1, p2, p3, p4 = _perm_var(xfunc=xfunc, vx=vx, yfunc=yfunc, vy=vy)
        sdx_percentile[t] = p1
        xy_percentile[t] = p2
        cor_percentile[t] = p3
        div_percentile[t] = p4

        
    return sdx_percentile, xy_percentile, cor_percentile, div_percentile

def plot_once(yf, vy):
    vxs = [0.01]
    yfuns = [None, expfunc.Z_to_Y]
    xfuns = [None, expfunc.Z_to_Y, expfunc.Z_to_Y2, expfunc.Z_to_Y3]
    xfuns_label = ['$X = Z + \epsilon_X$', '$X = 4Z^2 - 4Z-5 + \epsilon_X$', '$X = \log_2(Z+1)-3 + \epsilon_X$', '$X = 2/(Z+1) + \epsilon_X$']
    plt.figure(figsize=(20, 16))
    plt.subplots_adjust(wspace=.4, hspace=.4)
    for vx1 in range(len(vxs)):
        for xf in range(len(xfuns)):
            sdx, xy, cor_, div = perm_decomp(xfuns[xf], vxs[vx1], yfuns[yf], vy)
            plt.subplot(3, 4, 0*4+xf+1)
            plt.hist(sdx, bins=20, weights=np.zeros_like(sdx)+1./sdx.size)
            plt.xlabel("p-value")
            plt.ylabel("frequency")
            plt.title(f"xfunc={xfuns_label[xf]}, SD(X)={vxs[vx1]}, SD(X)")
            plt.subplot(3, 4, 1*4+xf+1)
            plt.hist(xy, bins=20, weights=np.zeros_like(sdx)+1./sdx.size)
            plt.xlabel("p-value")
            plt.ylabel("frequency")
            plt.title(f"xfunc={xfuns_label[xf]}, SD(X)={vxs[vx1]}, XY")
            plt.subplot(3, 4, 2*4+xf+1)
            plt.hist(div, bins=20, weights=np.zeros_like(sdx)+1./sdx.size)
            plt.xlabel("p-value")
            plt.ylabel("frequency")
            plt.title(f"xfunc={xfuns_label[xf]}, SD(X)={vxs[vx1]}, XY/SD(X)")
    plt.savefig("./results/yfunc_" + str(yf) + "_vx_" + str(vy) +".png")
    plt.show()

def wrapper(args):
    return plot_once(*args)

def perm_decomp2(xfunc, vx, yfunc, vy):
    xy_percentile = np.zeros(200)
    resid_percentile = np.zeros(200)

    def _perm_var(xfunc, vx, yfunc, vy):
        X, Y, Z = expfunc.data_generative(N=100, s=1, type="normal", hypo="h0", xfun=xfunc, yfun=yfunc, cor=0.4, vx=vx, vy=vy)
        X_resid_o = compute_resid(Z, X)
        Y_resid_o = compute_resid(Z, Y)
        # print(xfunc, vx, "=======")
        # print("X", np.mean(X_resid_o), np.std(X_resid_o))
        # print("Y", np.mean(Y_resid_o), np.std(Y_resid_o))
        #T_sam = np.dot(X_resid_o, X_resid_o)
        xy_sam = np.abs(np.dot(X_resid_o, Y_resid_o))
        resid_sam = np.abs(np.dot(X_resid_o, Y_resid_o))

        B = 1000
        m = 50
        xy_per = np.zeros(B)
        resid_per = np.zeros(B)

        G, _ = compute_G(Z, m)
        for i in range(B):
            new_X = _perm(X, G, m)
            X_resid_temp = compute_resid(Z, new_X)
            xy_per[i] = np.abs(np.dot(X_resid_temp, Y_resid_o))
            X_resid_per = _perm(X_resid_o, G, m)
            resid_per[i] = np.abs(np.dot(X_resid_per, Y_resid_o))

        p1 = (np.sum(xy_sam > xy_per) + 0.5 * np.sum(xy_sam == xy_per)) / B
        p2 = (np.sum(resid_sam > resid_per) + 0.5 * np.sum(resid_sam == resid_per)) / B
        return p1, p2
    
    for t in range(200):
        print(t)
        p1, p2 = _perm_var(xfunc=xfunc, vx=vx, yfunc=yfunc, vy=vy)
        xy_percentile[t] = p1
        resid_percentile[t] = p2

        
    return xy_percentile, resid_percentile

def plot_twice(vx, vy):
    yfuns = [None, expfunc.Z_to_Y, expfunc.Z_to_Y2, expfunc.Z_to_Y3]
    xfuns = [None, expfunc.Z_to_Y, expfunc.Z_to_Y2, expfunc.Z_to_Y3]
    xfuns_label = ['$X = Z + \epsilon_X$', '$X = 4Z^2 - 4Z-5 + \epsilon_X$', '$X = \log_2(Z+1)-3 + \epsilon_X$', '$X = 2/(Z+1) + \epsilon_X$']
    yfuns_label = ['$Y = Y + \epsilon_Y$', '$Y = 4Z^2 - 4Z-5 + \epsilon_Y$', '$Y = \log_2(Z+1)-3 + \epsilon_Y$', '$Y = 2/(Z+1)  + \epsilon_Y$']

    fig1, axes1 = plt.subplots(4, 4, figsize=(20, 20))
    fig2, axes2 = plt.subplots(4, 4, figsize=(20, 20))
    fig1.subplots_adjust(wspace=.4, hspace=.4)
    fig2.subplots_adjust(wspace=.4, hspace=.4)
    for xf in range(len(xfuns)):
        for yf in range(len(yfuns)):
            ax1 = axes1[xf, yf]
            ax2 = axes2[xf, yf]
            xy, resid_ = perm_decomp2(xfuns[xf], vx, yfuns[yf], vy)
            ax1.hist(xy, bins=20, weights=np.zeros_like(xy)+1./xy.size)
            ax1.xlabel("percentile")
            ax1.ylabel("frequency")
            ax1.title(f"{xfuns_label[xf]}, {yfuns_label[yf]}")
            ax2.hist(resid_, bins=20, weights=np.zeros_like(xy)+1./xy.size)
            ax2.xlabel("percentile")
            ax2.ylabel("frequency")
            ax2.title(f"{xfuns_label[xf]}, {yfuns_label[yf]}")
    fig1.savefig("./results/meanZ_vx_" + str(vx) + "_vy_" + str(vy) +".png")
    fig2.savefig("./results/residZ_vx_" + str(vx) + "_vy_" + str(vy) +".png")

def wrapper2(args):
    return plot_twice(*args)


# if __name__ == '__main__':
#     pool = mp.Pool(processes=4)
    
#     args_list = [(yf, vy) for yf in [0, 1] for vy in [0.01, 0.5]]
    
#     with pool:
#         pool.map(wrapper, args_list)
#         pool.close()

if __name__ == '__main__':
    pool = mp.Pool(processes=4)
    
    args_list = [(vx, vy) for vx in [0.01, 0.5] for vy in [0.01, 0.5]]
    
    with pool:
        pool.map(wrapper2, args_list)
        pool.close()