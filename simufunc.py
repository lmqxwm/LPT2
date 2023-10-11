import numpy as np
import pandas as pd
import statsmodels.api as sm
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri

FOCI = importr("FOCI")
codec = FOCI.codec


def compute_G(Z, M):
    G = pd.qcut(Z, M, labels=[i for i in range(M)])
    Z_means = [np.mean(Z[G == g]) for g in range(M)]
    return G, np.array([Z_means[G[i]] for i in range(Z.shape[0])])

def compute_T_partial(X, Y):
    XX = sm.add_constant(X)
    mod = sm.OLS(Y, XX).fit()
    return mod.params[1]

def compute_T_double(X, Y, Z):
    Z1 = sm.add_constant(Z)
    modx = sm.OLS(X, Z1).fit()
    mody = sm.OLS(Y, Z1).fit()
    return np.corrcoef(modx.resid, mody.resid)[0, 1]

def compute_resid(Z, Y):
    Z1 = sm.add_constant(Z)
    mod = sm.OLS(Y, Z1).fit()
    return mod.resid

def corr(X, Y):
    return np.abs(np.corrcoef(X, Y)[0, 1])


def LPT(X, Y, Z, M, B=100, gfunc=compute_G, perm="x"):
    G, ZZ = gfunc(Z, M)
    X_resid_o = compute_resid(Z, X)
    Y_resid_o = compute_resid(Z, Y)
    X_resid_m = compute_resid(ZZ, X)
    Y_resid_m = compute_resid(ZZ, Y)

    T_xy_o_sam = np.abs(X_resid_o @ Y_resid_o)
    T_xy_o_per = np.zeros(B)
    T_xy_m_sam = np.abs(X_resid_m @ Y_resid_m)
    T_xy_m_per = np.zeros(B)
    T_xy_sam = np.abs(X @ Y)
    T_xy_per = np.zeros(B)

    numpy2ri.activate()
    T_char_o_sam = robjects.conversion.rpy2py(codec(numpy2ri.py2rpy(X), numpy2ri.py2rpy(Y), numpy2ri.py2rpy(Z)))[0]
    T_char_m_sam = robjects.conversion.rpy2py(codec(numpy2ri.py2rpy(X), numpy2ri.py2rpy(Y), numpy2ri.py2rpy(ZZ)))[0]
    T_char_o_per = np.zeros(B)
    T_char_m_per = np.zeros(B)


    T_double_o_sam = np.abs(compute_T_double(X, Y, Z))
    T_double_o_per = np.zeros(B)
    # T_double_m_sam = np.abs(compute_T_double(X, Y, ZZ))
    # T_double_m_per = np.zeros(B)

    T_sam = corr(X, Y)
    T_per = np.zeros(B)

    # X_resid_o = compute_resid(Z, X)
    # Y_resid_o = compute_resid(Z, Y)
    # T_resid_sam = corr(X_resid_o, Y_resid_o)
    # assert T_double_o_sam == T_resid_sam
    # T_resid_per = np.zeros(B)

    def _perm(V, M):
        new_V = V.copy()
        for g in range(M):
            inds = np.where(G==g)[0]
            new_inds = inds.copy()
            np.random.shuffle(new_inds)
            new_V[new_inds] = new_V[inds]
        return new_V
    
    for b in range(B):
        if perm == "x":
            new_X = _perm(X, M)
            T_double_o_per[b] = np.abs(compute_T_double(new_X, Y, Z))
            #T_double_m_per[b] = np.abs(compute_T_double(new_X, Y, ZZ))
            T_per[b] = corr(new_X, Y)
            T_xy_o_per[b] = np.abs(compute_resid(Z, new_X) @ Y_resid_o)
            T_xy_m_per[b] = np.abs(compute_resid(ZZ, new_X) @ Y_resid_m)
            T_xy_per[b] = np.abs(new_X @ Y)
            # new_X_resid_o = _perm(X_resid_o, M)
            # T_resid_per[b] = corr(new_X_resid_o, Y_resid_o)
            T_char_o_per[b] = robjects.conversion.rpy2py(codec(numpy2ri.py2rpy(new_X), numpy2ri.py2rpy(Y), numpy2ri.py2rpy(Z)))[0]
            T_char_m_per[b] = robjects.conversion.rpy2py(codec(numpy2ri.py2rpy(new_X), numpy2ri.py2rpy(Y), numpy2ri.py2rpy(ZZ)))[0]
    
        elif perm == "y":
            new_Y = _perm(Y, M)
            T_double_o_per[b] = np.abs(compute_T_double(X, new_Y, Z))
            #T_double_m_per[b] = np.abs(compute_T_double(X, new_Y, ZZ))
            T_per[b] = corr(X, new_Y)
            T_xy_o_per[b] = np.abs(compute_resid(Z, new_Y) @ X_resid_o)
            T_xy_m_per[b] = np.abs(compute_resid(ZZ, new_Y) @ X_resid_m)
            T_xy_per[b] = np.abs(new_Y @ X)
            # new_Y_resid_o = _perm(Y_resid_o, M)
            # T_resid_per[b] = corr(X_resid_o, new_Y_resid_o)
            T_char_o_per[b] = robjects.conversion.rpy2py(codec(numpy2ri.py2rpy(X), numpy2ri.py2rpy(new_Y), numpy2ri.py2rpy(Z)))[0]
            T_char_m_per[b] = robjects.conversion.rpy2py(codec(numpy2ri.py2rpy(X), numpy2ri.py2rpy(new_Y), numpy2ri.py2rpy(ZZ)))[0]
    
        else:
            raise ValueError("Non-existing permutating variable!")
    
    p_cor_o = ((T_double_o_per >= T_double_o_sam).sum()+1) / (B+1)
    #p_cor_m = ((T_double_m_per >= T_double_m_sam).sum()+1) / (B+1)
    p_cor = ((T_per >= T_sam).sum()+1) / (B+1)
    p_xy_o = ((T_xy_o_per >= T_xy_o_sam).sum()+1) / (B+1)
    p_xy_m = ((T_xy_m_per >= T_xy_m_sam).sum()+1) / (B+1)
    p_xy = ((T_xy_per >= T_xy_sam).sum()+1) / (B+1)
    p_char_o = ((T_char_o_per >= T_char_o_sam).sum()+1) / (B+1)
    p_char_m = ((T_char_m_per >= T_char_m_sam).sum()+1) / (B+1)
    
    # p_o_resid = ((T_resid_per >= T_resid_sam).sum()+1) / (B+1)

    numpy2ri.deactivate()

    return p_cor_o, p_cor, p_xy_o, p_xy_m, p_xy, p_char_o, p_char_m

def LPT_resid(X, Y, Z, M, B=100, gfunc=compute_G, perm="x"):
    G, ZZ = gfunc(Z, M)

    X_resid_o = compute_resid(Z, X)
    X_resid_m = compute_resid(ZZ, X)
    Y_resid_o = compute_resid(Z, Y)
    Y_resid_m = compute_resid(ZZ, Y)

    T_double_o_sam = corr(X_resid_o, Y_resid_o)
    T_double_o_per = np.zeros(B)
    T_double_m_sam = corr(X_resid_m, Y_resid_m)
    T_double_m_per = np.zeros(B)

    def _perm(V, M):
        new_V = V.copy()
        for g in range(M):
            inds = np.where(G==g)[0]
            new_inds = inds.copy()
            np.random.shuffle(new_inds)
            new_V[new_inds] = new_V[inds]
        return new_V
    
    for b in range(B):
        if perm == "x":
            new_X_resid_o = _perm(X_resid_o, M)
            new_X_resid_m = _perm(X_resid_m, M)
            T_double_o_per[b] = corr(new_X_resid_o, Y_resid_o)
            T_double_m_per[b] = corr(new_X_resid_m, Y_resid_m)
        elif perm == "y":
            new_Y_resid_o = _perm(Y_resid_o, M)
            new_Y_resid_m = _perm(Y_resid_m, M)
            T_double_o_per[b] = corr(X_resid_o, new_Y_resid_o)
            T_double_m_per[b] = corr(X_resid_m, new_Y_resid_m)
        else:
            raise ValueError("Non-existing permutating variable!")
    
    p_o = ((T_double_o_per >= T_double_o_sam).sum()+1) / (B+1)
    p_m = ((T_double_m_per >= T_double_m_sam).sum()+1) / (B+1)

    return p_o, p_m

def LPT_partial(X, Y, Z, M, B=100, gfunc=compute_G, perm="x"):
    G, ZZ = gfunc(Z, M)
    T_x_o_sam = compute_T_partial(np.column_stack([Y, Z]), X)
    T_x_m_sam = compute_T_partial(np.column_stack([Y, ZZ]), X)
    T_x_sam = compute_T_partial(Y, X)
    T_y_o_sam = compute_T_partial(np.column_stack([X, Z]), Y)
    T_y_m_sam = compute_T_partial(np.column_stack([X, ZZ]), Y)
    T_y_sam = compute_T_partial(X, Y)
    T_x_o_per = np.zeros(B)
    T_x_m_per = np.zeros(B)
    T_x_per = np.zeros(B)
    T_y_o_per = np.zeros(B)
    T_y_m_per = np.zeros(B)
    T_y_per = np.zeros(B)

    def _perm(V, M):
        new_V = V.copy()
        for g in range(M):
            inds = np.where(G==g)[0]
            new_inds = inds.copy()
            np.random.shuffle(new_inds)
            new_V[new_inds] = new_V[inds]
        return new_V
    
    for b in range(B):
        if perm == "x":
            new_X = _perm(X, M)
            T_x_o_per[b] = compute_T_partial(np.column_stack([Y, Z]), new_X)
            T_x_m_per[b] = compute_T_partial(np.column_stack([Y, ZZ]), new_X)
            T_y_o_per[b] = compute_T_partial(np.column_stack([new_X, Z]), Y)
            T_y_m_per[b] = compute_T_partial(np.column_stack([new_X, ZZ]), Y)
            T_x_per[b] = compute_T_partial(Y, new_X)
            T_y_per[b] = compute_T_partial(new_X, Y)
        elif perm == "y":
            new_Y = _perm(Y, M)
            T_x_o_per[b] = compute_T_partial(np.column_stack([new_Y, Z]), X)
            T_x_m_per[b] = compute_T_partial(np.column_stack([new_Y, ZZ]), X)
            T_y_o_per[b] = compute_T_partial(np.column_stack([X, Z]), new_Y)
            T_y_m_per[b] = compute_T_partial(np.column_stack([X, ZZ]), new_Y)
            T_x_per[b] = compute_T_partial(new_Y, X)
            T_y_per[b] = compute_T_partial(X, new_Y)
        else:
            raise ValueError("Non-existing permutating variable!")
    
    p_o_x = ((T_x_o_per >= T_x_o_sam).sum()+1) / (B+1)
    p_m_x = ((T_x_m_per >= T_x_m_sam).sum()+1) / (B+1)
    p_o_y = ((T_y_o_per >= T_y_o_sam).sum()+1) / (B+1)
    p_m_y = ((T_y_m_per >= T_y_m_sam).sum()+1) / (B+1)
    p_x = ((T_x_per >= T_x_sam).sum()+1) / (B+1)
    p_y = ((T_y_per >= T_y_sam).sum()+1) / (B+1)

    return p_o_x, p_m_x, p_x, p_o_y, p_m_y, p_y



            





