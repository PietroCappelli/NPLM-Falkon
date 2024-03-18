import os
import sys
import csv
import time
import json
import itertools
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from scipy import stats
from scipy.stats import chi2, norm
from scipy.integrate import quad
from scipy.special import erfinv

from sklearn.metrics.pairwise import pairwise_distances

import torch
import falkon

# sys.path.insert(0, "../notebooks")
# from plot_utils import plot_one_t


def load_data(config : dict):
    # DATA_PATH = '../data/creditcard.csv'
    DATA_PATH = config["DATA_PATH"]
    df = pd.read_csv(DATA_PATH)

    #normalize Time and Amount features
    df['Time']   = (df['Time'] - df['Time'].mean()) / df['Time'].std()
    df['Amount'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()

    df_ref  = df[df['Class']==0]
    df_data = df[df['Class']==1]

    N_REF_TOT  = len(df_ref)
    N_DATA_TOT = len(df_data)

    Ref  = df_ref[df_ref.columns[:-1]].to_numpy()
    Data = df_data[df_data.columns[:-1]].to_numpy()

    Target_Ref  = df_ref['Class'].to_numpy()
    Target_Data = df_data['Class'].to_numpy()
    
    return Ref, Data


def learn_t(Ref, Data, config_json: dict):
    
    '''GENERATE DATASET'''
    # Statistics
    N_REF      = config_json['N_REF']
    N_BKG      = config_json['N_BKG']
    N_SIG      = config_json['N_SIG']
    # HYPER-PARAMS
    M = config_json['M']
    l = config_json['l']
    sigma = config_json['sigma']
    
    shape_only = config_json['shape_only']
    
    if shape_only:
        N_sig_obs = np.random.poisson(lam=N_SIG)
        N_bkg_obs = N_BKG - N_sig_obs
    else: 
        N_bkg_obs = np.random.poisson(lam=N_BKG)
        N_sig_obs = np.random.poisson(lam=N_SIG)
    # print("bkg obs:",N_bkg_obs)
    # print("sig obs:",N_sig_obs)
    
    # select a random chosen subset of the dataset
    idx_ref = np.random.choice(Ref.shape[0], N_REF+N_bkg_obs, replace=False)
    idx_sig = np.random.choice(Data.shape[0], N_sig_obs, replace=False)

    feature_ref  = torch.from_numpy(Ref[idx_ref[:N_REF], :])
    feature_bkg  = torch.from_numpy(Ref[idx_ref[N_REF:], :])
    feature_sig  = torch.from_numpy(Data[idx_sig, :])
    feature_data = torch.cat((feature_bkg, feature_sig), dim=0)

    target_ref  = torch.zeros((N_REF, 1), dtype=torch.float64)
    target_data = torch.ones((N_bkg_obs + N_sig_obs, 1), dtype=torch.float64)

    feature = torch.cat((feature_ref, feature_data), axis=0)
    target  = torch.cat((target_ref, target_data), axis=0)

    print('feature shape:', feature.shape, '\ttarget shape:', target.shape)

    '''DEFINE THE MODEL'''
    
    logflk_opt = falkon.FalkonOptions(cg_tolerance=np.sqrt(1e-7), keops_active='no', use_cpu=False, debug = False)
    logflk_kernel = falkon.kernels.GaussianKernel(sigma=sigma,  opt=logflk_opt)
    # logloss = falkon.gsc_losses.WeightedCrossEntropyLoss(logflk_kernel, neg_weight = N_BKG/N_REF)
    logloss = falkon.gsc_losses.WeightedCrossEntropyLoss(logflk_kernel, neg_weight = N_BKG/N_REF)

    config = {
        "kernel"       : logflk_kernel,
        "M"            : M,
        "penalty_list" : [l],
        "iter_list"    : [1_000_000],
        "options"      : logflk_opt,
        "seed"         : None,
        "loss"         : logloss,
    }

    logflk = falkon.LogisticFalkon(**config)
    
    '''TRAIN'''
    
    logflk.fit(feature, target)
    ref_pred, data_pred = logflk.predict(feature_ref), logflk.predict(feature_data)
    diff = N_BKG/N_REF *torch.sum(1 - torch.exp(ref_pred))
    t = 2 * (diff + torch.sum(data_pred).item()).item()
    # t = 2 * (torch.sum(data_pred)).item()
    # print("diff term",diff.item())
    # print("t ",t)
    del data_pred, ref_pred
    return t


def run_toys(Ref, Data, config):
    """
    return list of t values and save the results in csv files
    """
    toys = config['toys']
    M_list = config['M_list']
    l_list = config['l_list']
    shape_only = config['shape_only']
    
    current_date  = str(datetime.now().year)        + "_"
    current_date += str(datetime.now().month)       + "_"
    current_date += str(datetime.now().day)         + "_"
    current_date += str(datetime.now().hour)        + "_"
    current_date += str(datetime.now().minute)      
    
    directory = current_date + '_ref_' + str(config['N_REF']) + '_bkg_' + str(config['N_BKG']) + '_sig_' + str(config['N_SIG'])
    # if config['N_SIG']!=0:
    #     directory = current_date + '_' + "sig" + "_" + str(config['N_SIG'])
    if shape_only:
        # directory = current_date + '_' + "sig" + "_" + str(config['N_SIG']) + "_SOn"
        directory = directory + "_SOn"
    OUTPUT_PATH = config['OUTPUT_PATH'] + directory
        
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    t_list = []
    for l in l_list:
        t_M_list = []
        for M in M_list:
            t_list_tmp = []
            config_train = {
                "N_REF" : config["N_REF"],
                "N_BKG" : config["N_BKG"],
                "N_SIG" : config["N_SIG"],
                "M"     : M,
                "l"     : l,
                "sigma" : config["sigma"],
                "shape_only" :  config['shape_only']
                }
            for _ in range(toys):
                train_start = time.time()
                t = learn_t(Ref, Data, config_train)
                train_stop = time.time()
                t_list_tmp.append(t)
                
                FILE_NAME = "/ttest_time_"+str(l) +"_"+ str(M) + ".csv"
                
                with open(OUTPUT_PATH + FILE_NAME, "a", newline="\n") as f:
                    writer = csv.writer(f, dialect="excel" ,delimiter="\t")
                    writer.writerow([t, (train_stop-train_start)])
                
            t_M_list.append(t_list_tmp)
            del t_list_tmp
        t_list.append(t_M_list)
        del t_M_list
        
    print('data saved in:',OUTPUT_PATH)
    
    return t_list, directory
    
def extract_l_M(file):
    file = file.replace("ttest_time_", "")
    file = file.replace(".csv", "")
    file = file.replace("_", " ")
    l = file.split()[0]
    m = file.split()[1]
    # l_listtt.append(float(l)); M_listtt.append(int(m))

    # return list(set(l_listtt)), list(set(M_listtt))
    return float(l), int(m)
    

def read_and_plot(config, PATH):

    PATH = config['OUTPUT_PATH']
    for dir in os.listdir(PATH):
        for file in os.listdir(PATH+str(dir)):
            with open(PATH + str(dir) +'/'+ str(file)) as f:
                t_list = np.array([float(row.split()[0]) for row in f])
            with open(PATH + str(dir) +'/'+ str(file)) as f:
                timing = np.array([float(row.split()[1]) for row in f])

            dof_fit, _, _ = chi2.fit(t_list, floc=0, fscale=1)
            ks_statistic, ks_p_value = stats.kstest(t_list, "chi2", args=(dof_fit,))
            l,m = extract_l_M(file)
            # with open(PATH +"results"+ "/results.txt", "a", newline="\n"):
            #     f.write(f'KS p-value for ({m}, {l}): {ks_p_value:.2f} \t DOF of chi2: {dof_fit:.1f} \t Time(mean): {np.mean(timing)}\n')   
        
            t_ref_bins  = np.arange(int(np.min(t_list))-100, int(np.max(t_list))+100, 30)
            xgrid_ref   = np.arange(int(np.min(t_list))-100, int(np.max(t_list))+100, 2)
        #     plot_one_t(
        #         t_distribution  = t_list,
        #         t_bins          = t_ref_bins,
        #         chi2            = chi2(df=dof_fit),
        #         chi2_grid       = xgrid_ref,
        #         show_hist       = True,
        #         show_error      = False,
        #         compute_rate    = False,
        #         err_marker      = "o",
        #         err_markersize  = 10,
        #         err_capsize     = 5,
        #         err_elinewidth  = 4,
        #         err_capthick    = 4,
        #         err_color       = "black",
        #         figsize         = (10, 8),
        #         fontsize        = 18,
        #         cms             = False,
        #         cms_label       = "",
        #         cms_rlabel      = "",
        #         hist_ecolor     = ("#494B69", 1.0),
        #         hist_fcolor     = ("#494B69", 0.1),
        #         chi2_color      = ("#D8707C", 0.8),
        #         hist_lw         = 4,
        #         chi2_lw         = 8,
        #         hist_type       = "stepfilled",
        #         hist_label      = "$\it{t}$ distribution",
        #         chi2_label      = "Target $\chi^2$(dof=%.2f)"%(dof_fit),
        #         xlabel          = r"$t$",
        #         ylabel          = "Density",
        #         show_plot       = False,
        #         save_plot       = False,
        #         plot_name       = "t_distribution_"+str(l)+"_"+str(m),
        #         plot_path       = config['PLOT_PATH'],
        #         plot_format     = "png",
        #         return_fig      = False,
        #         plot_params     = False,
        #         hyperparams     = str(l)+", "+str(m),
        # )

def save_csv(config, filename):
    PATH = config['OUTPUT_PATH']
    OUTPUT_FILE = PATH + filename#"/results.csv"

    df_m = pd.DataFrame(columns=['lambda', 'M', 'p_value', 'dof','mean', 'median', 'timing'])
    for dir in os.listdir(PATH):
        if(str(dir)!="results"):
            for file in os.listdir(PATH+str(dir)):
                with open(PATH + str(dir) +'/'+ str(file)) as f:
                    t_list = np.array([float(row.split()[0]) for row in f])
                with open(PATH + str(dir) +'/'+ str(file)) as f:
                    timing = np.array([float(row.split()[1]) for row in f])
                l,m = extract_l_M(file)
                dof_fit, _, _ = chi2.fit(t_list, floc=0, fscale=1)
                _, ks_p_value = stats.kstest(t_list, "chi2", args=(dof_fit,))
                df_m = pd.concat([pd.DataFrame([[l,m, ks_p_value, dof_fit, np.mean(t_list), np.median(t_list), np.mean(timing)]], columns=df_m.columns), df_m], ignore_index=True)

    df_m = df_m.sort_values('lambda', ascending =False)
    df_m['p_value'] = df_m['p_value'].round(4)
    df_m['dof'] = df_m['dof'].round(2)
    df_m['mean'] = df_m['mean'].round(3)
    df_m['median'] = df_m['median'].round(3)
    df_m['timing'] = df_m['timing'].round(3)
        
    pd.options.display.float_format = '{}'.format
    df_m.to_csv(OUTPUT_FILE, sep='\t', index=False)
    
def save_csv_path(config, directory):
    
    '''SAVE IN A CSV FILE {p_value, dof, mean, median, timing} 
        OF THE DISTRIBUTION '''

    OUTPUT_PATH = config['OUTPUT_PATH'] + directory
    OUTPUT_FILE = config['OUTPUT_PATH'] + 'results/' + directory + '.csv'
    if config['N_SIG'] != 0 :
        OUTPUT_FILE = config['OUTPUT_PATH'] +'results/'+ directory + '_' + str(config['N_SIG']) +'.csv'
    if config['shape_only']:
        OUTPUT_FILE = config['OUTPUT_PATH'] +'results/'+ directory + '_' + str(config['N_SIG'])+ "_SOn" +'.csv'


    df_m = pd.DataFrame(columns=['lambda', 'M', 'p_value', 'dof','mean', 'median', 'timing'])
    for file in os.listdir(OUTPUT_PATH):
        if '.json' not in file:
            with open(OUTPUT_PATH +'/'+ str(file)) as f:
                t_list = np.array([float(row.split()[0]) for row in f])
            with open(OUTPUT_PATH +'/'+ str(file)) as f:
                timing = np.array([float(row.split()[1]) for row in f])
            l,m = extract_l_M(file)
            dof_fit, _, _ = chi2.fit(t_list, floc=0, fscale=1)
            _, ks_p_value = stats.kstest(t_list, "chi2", args=(dof_fit,))
            df_m = pd.concat([pd.DataFrame([[l,m, ks_p_value, dof_fit, np.mean(t_list), np.median(t_list), np.mean(timing)]], columns=df_m.columns), df_m], ignore_index=True)

    df_m = df_m.sort_values('lambda', ascending =False)
    df_m['p_value'] = df_m['p_value'].round(4)
    df_m['dof']     = df_m['dof'].round(2)
    df_m['mean']    = df_m['mean'].round(3)
    df_m['median']  = df_m['median'].round(3)
    df_m['timing']  = df_m['timing'].round(3)

    pd.options.display.float_format = '{}'.format
    df_m.to_csv(OUTPUT_FILE, sep='\t', index=False)
    print('results data saved in:',OUTPUT_FILE)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("-Nr", "--N_REF", type=int, help="size of reference dataset",  required=False, default=5000)
    parser.add_argument("-Nb", "--N_BKG", type=int, help="size of background dataset", required=False, default=1000)
    parser.add_argument("-Ns", "--N_SIG", type=int, help="number of signal events ",   required=False, default=0)
    parser.add_argument("-SO", "--shape_only", type=bool,help="Shape-only signal effect",  required=False, default=True)
    parser.add_argument("-l", "--Lambda", type=float, help="regularization param",  required=False, default=1e-8)
    parser.add_argument("-M", "--M",      type=int,   help="number of N. centers",  required=False, default=1000)
    
    
    args = parser.parse_args()
    print(f"number of ref: {args.N_REF}")
    print(f"number of bkg: {args.N_BKG}")
    print(f"number of sig: {args.N_SIG}")
    # print(f"regularization param: {args.Lambda}")
    # print(f"number of centers: {args.M}")
    # print(f"shapee only effect: {args.shape_only}")
    
    config = {
        'OUTPUT_PATH' : '/home/ubuntu/NPLM-Falkon/output/bank_data/Shape_only/',
        'DATA_PATH'   : '/home/ubuntu/NPLM-Falkon/data/creditcard.csv',
        'PLOT_PATH'   : '/home/ubuntu/NPLM-Falkon/plot/bank/N_study/',
        'toys'   : 80,
        'N_REF'  : args.N_REF,
        'N_BKG'  : args.N_BKG,
        'N_SIG'  : args.N_SIG,
        # 'M_list' : [args.M],
        # 'l_list' : [args.Lambda],
        'M_list' : [2000],
        'l_list' : [1e-7],
        'sigma'  : None,
        'shape_only' : args.shape_only,
        'distribution_path' : None
    }
    
    Ref, Data = load_data(config)
    # idx = np.random.choice(Ref.shape[0], 10_000, replace=False)
    # subset = Ref[idx]
    # distances = pairwise_distances(subset, metric='l2')
    # config['sigma'] = np.quantile(distances, 0.90)
    config['sigma'] = 9.5
    
    t_list, dir = run_toys(Ref, Data, config)
    # read_and_plot(config)
    
    json_file = config['OUTPUT_PATH'] + dir +'/'+ dir +'.json'
    if config['N_SIG']!=0:
        config['distribution_path'] = config['OUTPUT_PATH'] + dir + '/ttest_time_'+str(config['l_list'][0])+'_'+str(config['M_list'][0])+'.csv'
        with open(json_file, "w") as outfile: 
            json.dump(config, outfile, indent=4)    

    if config['N_SIG'] == 0: 
        save_csv_path(config, dir)
