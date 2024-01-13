import multiprocessing as mp
import pandas as pd
from functools import partial
import numpy as np
import math
import sys 
sys.path.insert(0, sys.path[0]+"/../")
import expfunc
import simufunc

if __name__ == '__main__':
    pool = mp.Pool(processes=6)
    N = 1000
    Ms = [math.ceil(N**(1/10)), 5, math.ceil(N**(1/2)), 16, 25, 50]

    #types = ["normal", "skewed_normal", "normal", "skewed_normal", "uni", "poi"]
    
    types = ["binomial"]
    hs =  ["h0"] * 1
    assert len(types) == len(hs)
    print("All M:", Ms)
    with pool:
        print("===============================process two")
        xfuns = [None, expfunc.Z_to_Y4, expfunc.Z_to_Y5]
        yfuns = [None, expfunc.Z_to_Y4, expfunc.Z_to_Y5]


        
        for xf in range(len(xfuns)):
            for yf in range(len(yfuns)):
                print("Processing xfunc=", xf)
                print("Processing yfunc=", yf)
                results = np.zeros([5, len(Ms)])
                for c in range(len(Ms)):
        
                    print("Processing M=", Ms[c])
                    
                    result = pool.map(partial(expfunc.experiment4, 
                                N=N, M=Ms[c], type="binomial", cor=0,
                                xfun=xfuns[xf], yfun=yfuns[yf], perm="x",
                                vx=1, vy=1), 
                                [i for i in range(100)])


                    results[0, c] = np.mean([r[0] for r in result])
                    results[1, c] = np.mean([r[1] for r in result])
                    results[2, c] = np.mean([r[2] for r in result])
                    results[3, c] = np.mean([r[3] for r in result])
                    results[4, c] = np.mean([r[4] for r in result])
                    results[5, c] = np.mean([r[5] for r in result])
                    

            pd.DataFrame(results, 
                columns=Ms, 
                index=["cor_Z", "noZ", "XY_Z", "XY_meanZ", "XY_noZ"]).to_csv(
                    sys.path[0]+"/results/result_n1000_ber_func_"+str(xf)+"_"+str(yf)+".csv")
