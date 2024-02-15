import numpy as np
import time
from datetime import datetime
from sklearn import datasets, model_selection
import torch
import matplotlib.pyplot as plt
import falkon

from scipy.stats import chi2
from plot_utils import plot_one_t



def learn_t(sigma, M, l):
    
    N_REF      = 200000
    N_BKG      = 2000
    N_SIG      = 0  
    SIG_LOC    = 6.4
    SIG_STD    = 0.16
    # poisson fluctuate the number of events in each sample
    N_bkg_p = int(torch.distributions.Poisson(rate=N_BKG).sample())
    N_sig_p = int(torch.distributions.Poisson(rate=N_SIG).sample())

    # the reference rate will not have nuisance parameters
    feature_ref_dist = torch.distributions.Exponential(rate=1)

    # the data rate will have nuisance parameters   
    feature_bkg_dist = torch.distributions.Exponential(rate=1)
    feature_sig_dist = torch.distributions.Normal(loc=SIG_LOC, scale=SIG_STD)

    feature_ref  = feature_ref_dist.sample((N_REF,1))
    feature_data = torch.cat(
        (
            feature_bkg_dist.sample((N_bkg_p, 1)),
            feature_sig_dist.sample((N_sig_p, 1))
        )
    )
    
    feature_ref  = feature_ref / torch.max(feature_ref)
    feature_data = feature_data / torch.max(feature_data)
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


def Hyperparameter_tuning(toys : int, output_path = "/home/ubuntu/NPLM-Falkon/output/"):

    M_list = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000]
    l_list = [1e-7, 1e-8, 1e-9, 1e-10]
    sigma  = 2.3
    t_M_list = []

    for M in M_list:
        t_list_tmp = []
        for _ in range(toys):
            t = learn_t(sigma=sigma, M=M, l=l_list[0])
            t_list_tmp.append(t)

        t_M_list.append(t_list_tmp)
        del t_list_tmp
        
    current_date  = str(datetime.now().year)        + "_"
    current_date += str(datetime.now().month)       + "_"
    current_date += str(datetime.now().day)         + "_"
    current_date += str(datetime.now().hour)        + "_"
    current_date += str(datetime.now().minute)      + "_"
    current_date += str(datetime.now().second)      
    
    outfile = output_path + current_date + ".txt"
    
    t_median = [np.median(t_M_list[i]) for i in range(len(t_M_list)) if i!=0]
    with open(outfile, 'w') as f:
        f.write('\n'.join(t_median))
    
    t_M_list_8 = []
    for M in M_list:
        t_list_tmp = []
        for _ in range(toys):
            t = learn_t(sigma=sigma, M=M, l=l_list[1])
            t_list_tmp.append(t)

        t_M_list_8.append(t_list_tmp)
        del t_list_tmp
        
    current_date  = str(datetime.now().year)        + "_"
    current_date += str(datetime.now().month)       + "_"
    current_date += str(datetime.now().day)         + "_"
    current_date += str(datetime.now().hour)        + "_"
    current_date += str(datetime.now().minute)      + "_"
    current_date += str(datetime.now().second)      
    
    outfile = output_path + current_date + ".txt"
    t_median = [np.median(t_M_list_8[i]) for i in range(len(t_M_list_8)) if i!=0]
    with open(outfile, 'w') as f:
        f.write('\n'.join(t_median))
        
    
    t_M_list_7 = []
    for M in M_list:
        t_list_tmp = []
        for _ in range(toys):
            t = learn_t(sigma=sigma, M=M, l=l_list[2])
            t_list_tmp.append(t)

        t_M_list_7.append(t_list_tmp)
        del t_list_tmp
        
    current_date  = str(datetime.now().year)        + "_"
    current_date += str(datetime.now().month)       + "_"
    current_date += str(datetime.now().day)         + "_"
    current_date += str(datetime.now().hour)        + "_"
    current_date += str(datetime.now().minute)      + "_"
    current_date += str(datetime.now().second)      
    
    outfile = output_path + current_date + ".txt"
    t_median = [np.median(t_M_list_7[i]) for i in range(len(t_M_list_7)) if i!=0]
    with open(outfile, 'w') as f:
        f.write('\n'.join(t_median)) 
        
    t_M_list_7 = []
    for M in M_list:
        t_list_tmp = []
        for _ in range(toys):
            t = learn_t(sigma=sigma, M=M, l=l_list[3])
            t_list_tmp.append(t)

        t_M_list_7.append(t_list_tmp)
        del t_list_tmp
        
    current_date  = str(datetime.now().year)        + "_"
    current_date += str(datetime.now().month)       + "_"
    current_date += str(datetime.now().day)         + "_"
    current_date += str(datetime.now().hour)        + "_"
    current_date += str(datetime.now().minute)      + "_"
    current_date += str(datetime.now().second)      
    
    outfile = output_path + current_date + ".txt"
    t_median = [np.median(t_M_list_7[i]) for i in range(len(t_M_list_7)) if i!=0]
    with open(outfile, 'w') as f:
        f.write('\n'.join(t_median))
        
        
        

if __name__ == "__main__":

    model_parameters = {
        'sigma' : 2.3,
        'M' : 3000,
        'l' : 1e-8,
        'output_path' : "/home/ubuntu/NPLM-Falkon/output/" 
    }
    
    toy(5, model_parameters)

    Hyperparameter_tuning(50)
