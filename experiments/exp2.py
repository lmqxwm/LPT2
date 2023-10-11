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
    N = 100
    Ms = [math.ceil(N**(1/10)), 5, math.ceil(N**(1/2)), 16, 25, 50]

    # types = ["normal", "skewed_normal", "normal", "skewed_normal", "uni", "poi"]
    
    types = ["normal"] * 2
    hs =  ["h0"] * 1 +["h1"]
    assert len(types) == len(hs)
    print("All M:", Ms)
    with pool:
    #     print("===============================process one")
    #     xfuns = [expfunc.Z_to_Y2, expfunc.Z_to_Y3]
    #     yfuns = [expfunc.Z_to_Y2, expfunc.Z_to_Y3]
    #     vxs = [1, 0.01]
    #     vys = [1, 0.01]

    #     for t in range(len(types)):
    #         for xf in range(len(xfuns)):
    #             for yf in range(len(yfuns)):
    #                 for vx1 in range(len(vxs)):
    #                     for vy1 in range(len(vys)):
    #                         results = np.zeros([2, len(Ms)])
    #                         for m in range(len(Ms)):
    #                             print("Processing type=", types[t])
    #                             print("Processing M=", Ms[m])
    #                             result = pool.map(partial(expfunc.experiment2, 
    #                                         N=N, M=Ms[m], type=types[t], hypo=hs[t], cor=0.4,
    #                                         xfun=xfuns[xf], yfun=yfuns[yf], perm="x",
    #                                         vx=vxs[vx1], vy=vys[vy1]), 
    #                                         [i for i in range(200)])

               
    #                             results[0, m] = np.mean([r[0] for r in result])
    #                             results[1, m] = np.mean([r[1] for r in result])

    #                             pd.DataFrame(results, 
    #                                 columns=Ms, 
    #                                 index=["double_Z", "double_meanZ"]).to_csv(
    #                                     sys.path[0]+"/results/result_m22_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_"+hs[t]+"_"+types[t]+".csv")

        print("===============================process two")
        xfuns = [expfunc.Z_to_Y, expfunc.Z_to_Y2, expfunc.Z_to_Y3, None]
        yfuns = [expfunc.Z_to_Y, expfunc.Z_to_Y2, expfunc.Z_to_Y3, None]
        vxs = [0.5, 0.01]
        vys = [0.5, 0.01]

        for t in range(len(types)):
            for xf in range(len(xfuns)):
                for yf in range(len(yfuns)):
                    for vx1 in range(len(vxs)):
                        for vy1 in range(len(vys)):
                            results = np.zeros([2, len(Ms)])
                            for m in range(len(Ms)):
                                print("Processing type=", types[t])
                                print("Processing M=", Ms[m])
                                result = pool.map(partial(expfunc.experiment2, 
                                            N=N, M=Ms[m], type=types[t], hypo=hs[t], cor=0.4,
                                            xfun=xfuns[xf], yfun=yfuns[yf], perm="x",
                                            vx=vxs[vx1], vy=vys[vy1]), 
                                            [i for i in range(100)])

            
                                results[0, m] = np.mean([r[0] for r in result])
                                results[1, m] = np.mean([r[1] for r in result])

                                pd.DataFrame(results, 
                                    columns=Ms, 
                                    index=["resid_Z", "resid_meanZ"]).to_csv(
                                        sys.path[0]+"/results/result_m29_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_"+hs[t]+"_"+types[t]+".csv")
        pool.close()