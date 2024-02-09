import os

import time
import torch
import numpy as np
import datetime

from sklearn.metrics import pairwise_distances

# from falkonhep.utils import read_data, generate_seeds, normalize_features


class HEPModel:
    """
    Generic model
    """

    def __init__(self, data_params):
        """Create a model for HEP anomaly detection

        Args:
            reference_path (str): path of directory containing data used as reference data (set of .h5 files)
            data_path (str): path of directory containing data used as datasample (set of .h5 files)
            output_path (str): directory in which results will be stored (If the directory doesn't exist it will be created)
            norm_fun (Optional, Callable): function used to normalize data. It takes the reference and data samples (two numpy.ndarray) and returns two numpy.ndarray representing the normalized reference and data samples (default to Higgs normalization)
        """

        self.model   = None
        self.N_REF   = data_params["N_REF"]
        self.N_BKG   = data_params["N_BKG"]
        self.N_SIG   = data_params["N_SIG"]
        self.SIG_LOC = data_params["SIG_LOC"]
        self.SIG_STD = data_params["SIG_STD"]
        
    @property
    def model_seed(self):
        if self.model is not None:
            return self.model.seed
        raise Exception("Model is not built yet!")
    
    # def generate_seeds(toy_label):
    #     """Given a toy id, this function generate a seed for reference and toy data

    #     Args:
    #         toy_label (int): Toy id

    #     Returns:
    #         (int, int): reference and toy seed
    #     """    
    #     seed_factor = datetime.datetime.now().microsecond + datetime.datetime.now().second+datetime.datetime.now().minute
        
    #     ref_seed = int(2**32-1) - ((int(toy_label) + 1) * seed_factor)
    #     toy_seed = ((int(toy_label) + 1) * seed_factor) % int(2**32-1)
            
    #     return ref_seed, toy_seed


    def generate_dataset(self, seed):
        """Generate dataset

        Args:
            R (int): Reference size
            B (int): Expected background size
            S (int): Expected signal size
            features (list): List of features
            cut (tuple[str, float]): Cut applied to mll
            normalize (bool): If true normalization is applied
            sig_type (int): Type of signal applied (admitted 0: no-signal, 1: resonant, 2: non-resonant)
            ref_state (np.random.RandomSeed): pseudorandom generator used for background size
            sig_state (np.random.RandomSeed): pseudorandom generator used for signal size

        Returns:
            (np.ndarray, np.ndarray, int, int (or None)): reference data, data-sample, background size, signal size (if sig_type=1)
        """   
        
        torch.manual_seed(seed)
        
        # poisson fluctuate the number of events in each sample
        N_bkg_p = int(torch.distributions.Poisson(rate=self.N_BKG).sample())
        N_sig_p = int(torch.distributions.Poisson(rate=self.N_SIG).sample())
        
        feature_ref_dist = torch.distributions.Exponential(rate=1)

        feature_bkg_dist = torch.distributions.Exponential(rate=1)
        feature_sig_dist = torch.distributions.Normal(loc=self.SIG_LOC, scale=self.SIG_STD)
        
        # generate the features
        feature_ref  = feature_ref_dist.sample((self.N_REF, 1)) 
        feature_data = torch.cat(
            (
                feature_bkg_dist.sample((N_bkg_p, 1)),
                feature_sig_dist.sample((N_sig_p, 1))
            )
        )
        # normalize
        # feature_ref = feature_ref / feature_ref.max()
        # feature_data = feature_data / feature_data.max()

        # concatenate the features
        feature = torch.cat((feature_ref, feature_data), dim=0)

        # generate the target
        target_ref  = torch.zeros((self.N_REF, 1))
        target_data = torch.ones((N_bkg_p + N_sig_p, 1))

        target = torch.cat((target_ref, target_data), dim=0)
        
        return feature, target, feature_ref, feature_data, target_ref, target_data
        


    def create_labels(self, ref_size, data_size):
        """Given reference and data size, it returns

        Args:
            ref_size (int): reference sample size
            data_size (int): data sample size
        
        Returns:
            (np.ndarray): returns the label vector
        """        
        raise NotImplementedError("This function is not implemented in general class HEPModel")

    def build_model(self, model_parameters, weight):
        """Function used to build the model

        Args:
            model_parameters (Map): model parameters
            weight (float): weight
        """        
        raise NotImplementedError("This function is not implemented in general class HEPModel")

    def predict(self, data):
        raise NotImplementedError("This function is not implemented in general class HEPModel")


    def fit(self, X, y):
        self.model.fit(X, y)

    def learn_t(self, model_parameters:dict, seeds = None):       
        """Method used to compute the t values 

        Args:
            R (int): Size of the reference \(N_0\)
            B (int): Mean of the Poisson distribution from which the size of the background is sampled
            S (int): Mean of the Poisson distribution from which the size of the signal is sampled
            features (list[str]): List containing the name of the features used
            model_parameters (dict): Dictionary containing the parameters for the model used
            sig_type (int): Type of signal (0: no-signal, 1: resonant, 2: non-resonant).
            cut (int, optional): Cut MLL. Defaults to None.
            normalize (bool, optional): If True data will be normalized before fitting the model. Defaults to False.
            seeds (tuple[int, int], optional): A tuple (reference_seed, data_seed) used to generate reference and data sample, if None two random seeds are generated. Defaults to None.
            pred_features (list[str], optional): List of features to perform predictions. Defaults to None.

        """        
        seed_factor = datetime.datetime.now().microsecond + datetime.datetime.now().second+datetime.datetime.now().minute
        ref_seed = int(2**32-1) - ((int(np.random.randint(10)) + 1) * seed_factor)
        
        ref_seed = seeds if seeds is not None else ref_seed


        feature, target, feature_ref, feature_data, target_ref, target_data = self.generate_dataset(ref_seed)
        
        # data = np.vstack((reference, data_sample))
        # data_size = bck_size + sig_size if sig_size is not None else bck_size
        # # Labels
        # labels = self.create_labels(reference.shape[0], data_size)
            
        # Create and fit model
        weight = self.N_BKG / self.N_REF 
        self.build_model(model_parameters, weight)

        Xtorch = feature
        Ytorch = target   

        train_time = time.time()
        self.fit(Xtorch, Ytorch)
        train_time = time.time() - train_time

        ref_pred, data_pred = self.predict(feature_ref), self.predict(feature_data)

        # Compute Nw and t

        Nw = weight*torch.sum(torch.exp(ref_pred))
        diff = weight*torch.sum(1 - torch.exp(ref_pred))
        t = 2 * (diff + torch.sum(data_pred).item()).item()

        del data_pred
        return t, Xtorch, Ytorch#, Nw, train_time, ref_seed #, data_seed#, ref_pred.numpy().reshape(-1)#ref_pred



    def save_result(self, fname, i, t, Nw, train_time, ref_seed, sig_seed):
        """Function which save the result of learn_t in a file

        Args:
            fname (str): File name in which the result will be stored (the file will be stored in the output path specified). 
            If the file already exists, the result will be appended. 
            i (int): Toy identifier
            t (float): value of t obtained
            Nw (float): Nw
            train_time (float): Time spent in fitting the model
            ref_seed (int): seed to reproduce the reference sample
            sig_seed (int): seed to reproduce the data sample
        """        
        with open(self.output_path + "/{}".format(fname), "a") as f:
            f.write("{},{},{},{},{},{},{}\n".format(i, t, Nw, train_time, ref_seed, sig_seed, self.model_seed))



