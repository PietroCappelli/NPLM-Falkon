import os
import numpy as np
import time
from sklearn import datasets, model_selection
import torch
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import falkon
from scipy.stats import chi2
# from plot_utils import plot_one_t



def learn_t(sigma, M, l, seed):
    
    N_REF      = 200_000
    N_BKG      = 2_000
    N_SIG      = 0
    SIG_LOC    = 6.4
    SIG_STD    = 0.16
    # poisson fluctuate the number of events in each sample
    # torch.manual_seed(seed)
    N_bkg_p = int(torch.distributions.Poisson(rate=N_BKG).sample())
    N_sig_p = int(torch.distributions.Poisson(rate=N_SIG).sample())

    # the reference rate will not have nuisance parameters
    feature_ref_dist = torch.distributions.Exponential(rate=8)

    # the data rate will have nuisance parameters   
    feature_bkg_dist = torch.distributions.Exponential(rate=8)
    feature_sig_dist = torch.distributions.Normal(loc=SIG_LOC, scale=SIG_STD)

    feature_ref  = feature_ref_dist.sample((N_REF,1))
    feature_data = torch.cat(
        (
            feature_bkg_dist.sample((N_bkg_p, 1)),
            feature_sig_dist.sample((N_sig_p, 1))
        )
    )
    
    # feature_ref  = feature_ref / torch.max(feature_ref)
    # feature_data = feature_data / torch.max(feature_data)
    feature = torch.cat((feature_ref, feature_data), dim=0)
    
    target_ref  = torch.zeros((N_REF, 1))
    target_data = torch.ones((N_bkg_p + N_sig_p, 1))

    target = torch.cat((target_ref, target_data), dim=0)
    print("target shape",target.shape)
    
    logflk_opt = falkon.FalkonOptions(cg_tolerance=np.sqrt(1e-7), keops_active='no', use_cpu=False, debug = False)
    logflk_kernel = falkon.kernels.GaussianKernel(sigma=sigma,  opt=logflk_opt)
    logloss = falkon.gsc_losses.WeightedCrossEntropyLoss(logflk_kernel, neg_weight = N_BKG/N_REF)

    config = {
        "kernel"       : logflk_kernel,
        "M"            : M,
        "penalty_list" : [l],
        "iter_list"    : [100_000],
        "options"      : logflk_opt,
        "seed"         : None,
        "loss"         : logloss,
    }

    logflk = falkon.LogisticFalkon(**config)
    logflk.fit(feature, target)
    ref_pred, data_pred = logflk.predict(feature_ref), logflk.predict(feature_data)
    diff = N_BKG/N_REF *torch.sum(1 - torch.exp(ref_pred))
    t = 2 * (diff + torch.sum(data_pred).item()).item()
    
    return t
    

def toy(toy : int, model_params : dict):
    
    sigma = model_params['sigma']
    M     = model_params['M']
    l     = model_params['l']
    output_path = model_params['output_path']
    
    t_list=[]
    for _ in range(toy):
        t = learn_t(sigma, M, l)
        t_list.append(t)

        
    current_date  = str(datetime.now().year)        + "_"
    current_date += str(datetime.now().month)       + "_"
    current_date += str(datetime.now().day)         + "_"
    current_date += str(datetime.now().hour)        + "_"
    current_date += str(datetime.now().minute)      + "_"
    current_date += str(datetime.now().second)      
    
    outfile = output_path + current_date + ".txt"
    
    with open(outfile, 'w') as f:
        for t in t_list:
            f.write("%f\n"%(t))


def hyperparameter_train(toys, M_list, l_list, sigma, FILE_PATH):
    
    """
    return list of t values and save the results in csv files
    """

    for l in l_list:
        t_M_list = []
        for M in M_list:
            t_list_tmp = []
            for _ in range(toys):
                train_start = time.time()
                t = learn_t(sigma=sigma, M=M, l=l, seed=time.time())
                train_stop = time.time()
                t_list_tmp.append(t)
                
                FILE_NAME = "/ttest_time_"+str(l) +"_"+ str(M) + ".csv"
                
                with open(FILE_PATH + FILE_NAME, "a", newline="\n") as f:
                    writer = csv.writer(f, dialect="excel" ,delimiter="\t")
                    writer.writerow([t, (train_stop-train_start)])
                                

            t_M_list.append(t_list_tmp)
            del t_list_tmp
        t_list.append(t_M_list)
        del t_M_list
    
    return t_list
    
def read_data(l, out, path):
    t = []
    for M in [500, 1000, 2000, 3000, 5000, 6000]:
        if out != 'time':
            with open(path + "/ttest_time_"+str(l)+"_"+str(M)+".csv") as f:
            # for row in f:
                t_list_tmp = np.array([float(row.split()[0]) for row in f])
                if out == 'mean':
                    t.append(np.mean(t_list_tmp))
                if out == 'chi':
                    df_fit, _, _ = chi2.fit(t_list_tmp, floc=0, fscale=1)
                    t.append(df_fit)
                if out == 'median':
                    t.append(np.median(t_list_tmp))
        if out == 'time':
            with open(path + "/ttest_time_"+str(l)+"_"+str(M)+".csv") as f:
                timing = np.array([float(row.split()[1]) for row in f])
                t.append(np.mean(timing))
                        
    return t

if __name__ == "__main__":

    current_date  = str(datetime.now().year)        + "_"
    current_date += str(datetime.now().month)       + "_"
    current_date += str(datetime.now().day)         + "_"
    current_date += str(datetime.now().hour)        + "-"
    current_date += str(datetime.now().minute)      + "_"

    FILE_PATH = "output/1D/"+current_date
    os.makedirs(FILE_PATH, exist_ok=True)
    
    toys   = 10
    M_list = [500, 1000, 2000, 3000, 5000, 6000]
    l_list = [1e-8, 1e-7, 1e-6]#, 1e-9]
    sigma  = 0.3
    t_list = []

    t_list = hyperparameter_train(toys, M_list, l_list, sigma, FILE_PATH)

    path = "/home/ubuntu/NPLM-Falkon/output/1D/"+current_date
    
    fig, ax = plt.subplots(figsize=(8,8))
    ax.plot(M_list,read_data(1e-6, 'mean', path=path), marker='o', markersize=6 , label="$\lambda$=1e-6")
    ax.plot(M_list,read_data(1e-7, 'mean', path=path), marker='o', markersize=6 , label="$\lambda$=1e-7")
    ax.plot(M_list,read_data(1e-8, 'mean', path=path), marker='o', markersize=6 , label="$\lambda$=1e-8")

    ax.set_xlabel('M', fontsize=12)
    ax.set_ylabel('mean t', fontsize=12)
    ax.legend(fontsize =12)

    plt.savefig("plot/meant_vs_M.png")

