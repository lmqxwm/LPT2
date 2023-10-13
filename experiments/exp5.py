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
    
    types = ["normal"]
    hs =  ["h0"] * 1
    assert len(types) == len(hs)
    print("All M:", Ms)
    with pool:
        # print("===============================process one")
        # xfuns = [expfunc.Z_to_Y2, expfunc.Z_to_Y3]
        # yfuns = [expfunc.Z_to_Y2, expfunc.Z_to_Y3]
        # vxs = [0.5, 0.01]
        # vys = [0.5, 0.01]

        # for t in range(len(types)):
        #     for xf in range(len(xfuns)):
        #         for yf in range(len(yfuns)):
        #             for vx1 in range(len(vxs)):
        #                 for vy1 in range(len(vys)):
        #                     results = np.zeros([2, len(Ms)])
        #                     for m in range(len(Ms)):
        #                         print("Processing type=", types[t])
        #                         print("Processing M=", Ms[m])
        #                         result = pool.map(partial(expfunc.experiment, 
        #                                     N=N, M=Ms[m], type=types[t], hypo=hs[t], cor=0.4,
        #                                     xfun=xfuns[xf], yfun=yfuns[yf], perm="x",
        #                                     vx=vxs[vx1], vy=vys[vy1]), 
        #                                     [i for i in range(100)])

               
        #                         results[0, m] = np.mean([r[0] for r in result])
        #                         results[1, m] = np.mean([r[1] for r in result])

        #                         pd.DataFrame(results, 
        #                             columns=Ms, 
        #                             index=["double_Z", "double_meanZ"]).to_csv(
        #                                 sys.path[0]+"/results/result_7_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_"+hs[t]+"_"+types[t]+".csv")
        # print("===============================process one")
        # xfuns = [None, expfunc.Z_to_Y]
        # yfuns = [None, expfunc.Z_to_Y]
        # vxs = [1, 0.001]
        # vys = [1, 0.001]

        # for t in range(len(types)):
        #     for xf in range(len(xfuns)):
        #         for yf in range(len(yfuns)):
        #             for vx1 in range(len(vxs)):
        #                 for vy1 in range(len(vys)):
        #                     results = np.zeros([3, len(Ms)])
        #                     for m in range(len(Ms)):
        #                         print("Processing type=", types[t])
        #                         print("Processing M=", Ms[m])
        #                         result = pool.map(partial(expfunc.experiment, 
        #                                     N=N, M=Ms[m], type=types[t], hypo=hs[t], cor=0.4,
        #                                     xfun=xfuns[xf], yfun=yfuns[yf], perm="x",
        #                                     vx=vxs[vx1], vy=vys[vy1]), 
        #                                     [i for i in range(100)])

               
        #                         results[0, m] = np.mean([r[0] for r in result])
        #                         results[1, m] = np.mean([r[1] for r in result])
        #                         results[2, m] = np.mean([r[2] for r in result])

        #                         pd.DataFrame(results, 
        #                             columns=Ms, 
        #                             index=["double_Z", "double_meanZ", "noZ"]).to_csv(
        #                                 sys.path[0]+"/results/result_8_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_"+hs[t]+"_"+types[t]+".csv")
        # print("===============================process one2")
        # xfuns = [None, expfunc.Z_to_Y]
        # yfuns = [None, expfunc.Z_to_Y]
        # vxs = [5, 0.01]
        # vys = [5, 0.01]

        # for t in range(len(types)):
        #     for xf in range(len(xfuns)):
        #         for yf in range(len(yfuns)):
        #             for vx1 in range(len(vxs)):
        #                 for vy1 in range(len(vys)):
        #                     results = np.zeros([3, len(Ms)])
        #                     for m in range(len(Ms)):
        #                         print("Processing type=", types[t])
        #                         print("Processing M=", Ms[m])
        #                         result = pool.map(partial(expfunc.experiment, 
        #                                     N=N, M=Ms[m], type=types[t], hypo=hs[t], cor=0.4,
        #                                     xfun=xfuns[xf], yfun=yfuns[yf], perm="x",
        #                                     vx=vxs[vx1], vy=vys[vy1]), 
        #                                     [i for i in range(100)])

               
        #                         results[0, m] = np.mean([r[0] for r in result])
        #                         results[1, m] = np.mean([r[1] for r in result])
        #                         results[2, m] = np.mean([r[2] for r in result])

        #                         pd.DataFrame(results, 
        #                             columns=Ms, 
        #                             index=["double_Z", "double_meanZ", "noZ"]).to_csv(
        #                                 sys.path[0]+"/results/result_8_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_"+hs[t]+"_"+types[t]+".csv")

        print("===============================process two")
        xfuns = [None, expfunc.Z_to_Y, expfunc.Z_to_Y2, expfunc.Z_to_Y3]
        yfuns = [None, expfunc.Z_to_Y, expfunc.Z_to_Y2, expfunc.Z_to_Y3]
        
        vxs = [0.5, 0.01]
        vys = [0.5, 0.01]
        cors = 0.1 * np.arange(11)/10

        for t in range(len(types)):
            for xf in range(4):
                yfs = [0, 1] if xf <= 1 else [2, 3]
                for yf in yfs:
                    for vx1 in range(len(vxs)):
                        for vy1 in range(len(vys)):
                            print("Processing type=", types[t])
                            print("Processing xf, yf, vx, vy=", xf, yf, vx1, vy1)
                            results = np.zeros([7, len(cors)])
                            for c in range(len(cors)):
                                print("Processing cor=", cors[c])
                                result = pool.map(partial(expfunc.experiment, 
                                            N=N, M=50, type=types[t], cor=cors[c],
                                            xfun=xfuns[xf], yfun=yfuns[yf], perm="x",
                                            vx=vxs[vx1], vy=vys[vy1]), 
                                            [i for i in range(100)])

            
                                results[0, c] = np.mean([r[0] for r in result])
                                results[1, c] = np.mean([r[1] for r in result])
                                results[2, c] = np.mean([r[2] for r in result])
                                results[3, c] = np.mean([r[3] for r in result])
                                results[4, c] = np.mean([r[4] for r in result])
                                results[5, c] = np.mean([r[5] for r in result])
                                results[6, c] = np.mean([r[6] for r in result])

                                pd.DataFrame(results, 
                                    columns=cors, 
                                    index=["cor_Z", "noZ", "XY_Z", "XY_meanZ", "XY_noZ", "Char_Z", "Char_meanZ"]).to_csv(
                                        sys.path[0]+"/results/result_n1000_m50_cor01_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_"+hs[t]+"_"+types[t]+".csv")

        # for t in range(len(types)):
        #     for xf in range(4):
        #         yfs = [0, 1] if xf <= 1 else [2, 3]
        #         for yf in yfs:
        #             for vx1 in range(len(vxs)):
        #                 for vy1 in range(len(vys)):
        #                     print("Processing type=", types[t])
        #                     print("Processing xf, yf, vx, vy=", xf, yf, vx1, vy1)
        #                     results = np.zeros([7, len(cors)])
        #                     for c in range(len(cors)):
        #                         print("Processing cor=", cors[c])
        #                         result = pool.map(partial(expfunc.experiment, 
        #                                     N=N, M=25, type=types[t], cor=cors[c],
        #                                     xfun=xfuns[xf], yfun=yfuns[yf], perm="x",
        #                                     vx=vxs[vx1], vy=vys[vy1]), 
        #                                     [i for i in range(100)])

            
        #                         results[0, c] = np.mean([r[0] for r in result])
        #                         results[1, c] = np.mean([r[1] for r in result])
        #                         results[2, c] = np.mean([r[2] for r in result])
        #                         results[3, c] = np.mean([r[3] for r in result])
        #                         results[4, c] = np.mean([r[4] for r in result])
        #                         results[5, c] = np.mean([r[5] for r in result])
        #                         results[6, c] = np.mean([r[6] for r in result])

        #                         pd.DataFrame(results, 
        #                             columns=cors, 
        #                             index=["cor_Z", "noZ", "XY_Z", "XY_meanZ", "XY_noZ", "Char_Z", "Char_meanZ"]).to_csv(
        #                                 sys.path[0]+"/results/result_n1000_m25_small_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_"+hs[t]+"_"+types[t]+".csv")

        # for t in range(len(types)):
        #     for xf in range(4):
        #         yfs = [0, 1] if xf <= 1 else [2, 3]
        #         for yf in yfs:
        #             for vx1 in range(len(vxs)):
        #                 for vy1 in range(len(vys)):
        #                     print("Processing type=", types[t])
        #                     print("Processing xf, yf, vx, vy=", xf, yf, vx1, vy1)
        #                     results = np.zeros([7, len(cors)])
        #                     for c in range(len(cors)):
        #                         print("Processing cor=", cors[c])
        #                         result = pool.map(partial(expfunc.experiment, 
        #                                     N=N, M=10, type=types[t], cor=cors[c],
        #                                     xfun=xfuns[xf], yfun=yfuns[yf], perm="x",
        #                                     vx=vxs[vx1], vy=vys[vy1]), 
        #                                     [i for i in range(100)])

            
        #                         results[0, c] = np.mean([r[0] for r in result])
        #                         results[1, c] = np.mean([r[1] for r in result])
        #                         results[2, c] = np.mean([r[2] for r in result])
        #                         results[3, c] = np.mean([r[3] for r in result])
        #                         results[4, c] = np.mean([r[4] for r in result])
        #                         results[5, c] = np.mean([r[5] for r in result])
        #                         results[6, c] = np.mean([r[6] for r in result])

        #                         pd.DataFrame(results, 
        #                             columns=cors, 
        #                             index=["cor_Z", "noZ", "XY_Z", "XY_meanZ", "XY_noZ", "Char_Z", "Char_meanZ"]).to_csv(
        #                                 sys.path[0]+"/results/result_n1000_m10_small_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_"+hs[t]+"_"+types[t]+".csv")

        # xfuns = [None, expfunc.Z_to_Y, expfunc.Z_to_Y2, expfunc.Z_to_Y3]
        # yfuns = [None, expfunc.Z_to_Y, expfunc.Z_to_Y2, expfunc.Z_to_Y3]
        
        # vxs = [0.5, 0.01]
        # vys = [0.5, 0.01]
        # cors = np.arange(11)/10

        # for t in range(len(types)):
        #     for xf in range(4):
        #         yfs = [0, 1] if xf <= 1 else [2, 3]
        #         for yf in yfs:
        #             for vx1 in range(len(vxs)):
        #                 for vy1 in range(len(vys)):
        #                     print("Processing type=", types[t])
        #                     print("Processing xf, yf, vx, vy=", xf, yf, vx1, vy1)
        #                     results = np.zeros([5, len(cors)])
        #                     for c in range(len(cors)):
        #                         print("Processing cor=", cors[c])
        #                         result = pool.map(partial(expfunc.experiment, 
        #                                     N=N, M=10, type=types[t], cor=cors[c],
        #                                     xfun=xfuns[xf], yfun=yfuns[yf], perm="x",
        #                                     vx=vxs[vx1], vy=vys[vy1]), 
        #                                     [i for i in range(100)])

            
        #                         results[0, c] = np.mean([r[0] for r in result])
        #                         results[1, c] = np.mean([r[1] for r in result])
        #                         results[2, c] = np.mean([r[2] for r in result])
        #                         results[3, c] = np.mean([r[3] for r in result])
        #                         results[4, c] = np.mean([r[4] for r in result])
                                

        #                         pd.DataFrame(results, 
        #                             columns=cors, 
        #                             index=["cor_Z", "noZ", "XY_Z", "XY_meanZ", "XY_noZ"]).to_csv(
        #                                 sys.path[0]+"/results/result_n1000_m10_all_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_"+hs[t]+"_"+types[t]+".csv")

        
        # print("===============================process three")
        # xfuns = [None, expfunc.Z_to_Y, expfunc.Z_to_Y2, expfunc.Z_to_Y3]
        # yfuns = [None, expfunc.Z_to_Y, expfunc.Z_to_Y2, expfunc.Z_to_Y3]
        # vxs = [0.5, 0.01]
        # vys = [0.5, 0.01]

        # for t in range(len(types)):
        #     for xf in range(len(xfuns)):
        #         for yf in range(len(yfuns)):
        #             for vx1 in range(len(vxs)):
        #                 for vy1 in range(len(vys)):
        #                     results = np.zeros([3, len(Ms)])
        #                     for m in range(len(Ms)):
        #                         print("Processing type=", types[t])
        #                         print("Processing M=", Ms[m])
        #                         result = pool.map(partial(expfunc.experiment, 
        #                                     N=N, M=Ms[m], type=types[t], cor=0,
        #                                     xfun=xfuns[xf], yfun=yfuns[yf], perm="x",
        #                                     vx=vxs[vx1], vy=vys[vy1]), 
        #                                     [i for i in range(100)])

            
        #                         results[0, m] = np.mean([r[0] for r in result])
        #                         results[1, m] = np.mean([r[1] for r in result])
        #                         results[2, m] = np.mean([r[2] for r in result])

        #                         pd.DataFrame(results, 
        #                             columns=Ms, 
        #                             index=["double_Z", "double_meanZ", "noZ"]).to_csv(
        #                                 sys.path[0]+"/results/result_m50_x_func_"+str(xf)+"_"+str(yf)+"_var_"+str(vxs[vx1])+"_"+str(vys[vy1])+"_h0_"+types[t]+".csv")


       
        pool.close()