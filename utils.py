# -*- coding: utf-8 -*-
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from distfit import distfit
from math import exp,log,sqrt
from statistics import stdev,mean
from scipy.stats import norm, beta, lognorm, uniform, ks_2samp

# colour palette. Change to whatever you like.
palette = {
    'beta': '#470024',
    'uniform': '#3489bb',
    'lognorm': '#f3754e',
    'norm': '#5b58a4',
    }

def show_info(
        N, 
        T, 
        sigma, 
        mean_path,
        ):
    '''
    A func to show the info abt a simulation.

    Parameters
    ----------
    N : int
        # sims
    T : int
        # periods for each sim
    sigma : float
        Valitility
    mean_path : arrary-like var
        The mean path calculated from N times of sims.

    Returns
    -------
    None.
    '''
    
    print('-- SUMMARY OF MONTE CARLO SIMULATION --')
    print(f'Num. of simulations: {N}')
    print(f'Num. periods: {T}')
    print(f'Estimated volatility: {sigma}')
    print(f'Mean path:\n {mean_path}')

def gen_pdf(
        name, 
        x, 
        mu, 
        sigma, 
        args,
        ):
    '''
    Generate PDF from given dist.

    Parameters
    ----------
    name : str
        Name of the dist to be generated
    x : arrary-like
        Range of the x axis
    mu : float
        Mean of the dist
    sigma : float
        Volatility of the dist
    args : arrary-like
        Other attrs of the dist, if any

    Returns
    -------
    p : arrary-like
        Data for the Y axis
    '''
    
    if name == 'beta':
        a, b = args
        p = beta.pdf(x, a, b, loc=mu, scale=sigma)
    elif name == 'uniform':
        a,b = mu, sigma
        p = uniform.pdf(x, a, b)
    elif name == 'lognorm':
        support, = args
        p = lognorm.pdf(x, s=sigma, scale=mu)
    elif name == 'norm':
        p = norm.pdf(x, mu, sigma)
        
    return p

def gen_sample(name, n, mu, sigma, args):
    '''
    Generate a sample from a given dist

    Parameters
    ----------
    name : str
        Name of the dist
    n : int
        # obs
    mu : float
        Mean of the dist
    sigma : float
        Volatility of the dist
    args : arrary-like
        Other attrs of the dist, if any

    Returns
    -------
    s : arrary-like
        Sample generated
    '''
    
    if name == 'beta':
        a, b = args
        s = mu + sigma*np.random.beta(a, b, n)
    elif name == 'uniform':
        a,b = mu, sigma
        s = np.random.uniform(a, b, n)
    elif name == 'lognorm':
        shape, = args
        s = np.random.lognormal(log(sigma), shape, n)
    elif name == 'norm':
        s = np.random.normal(mu, sigma,n)
        
    return s
    

class DataProperties:
    def __init__(self, data):
        '''
        This class analyses the data and return the dist with 
        the best finess.
        To get an instance of it, fit the data to be analysed
        to initialise.
        '''
        
        self.name = data.name
        self.data = data
        dist = distfit()
        dist.fit_transform(self.data,verbose=1)
        self.fit_summary = dist.summary.set_index('distr')
        
    @property
    def _summary_stats(self):
        '''
        Return simple summary stats of the data
        '''
        
        return {
            'max':max(self.data),
            'min':min(self.data),
            'mean':mean(self.data),
            'std':stdev(self.data),
            'sample size':len(self.data),
            }
    
    @property
    def _fitness(self):
        '''
        Return the results of fitting. The dists are ranked by
        fitness score.
        '''
        
        dist = self.fit_summary
        return dist.loc[['norm','uniform'], #'beta','lognorm',
                        :].sort_values(by=['score'])
                        
    def get_best_dist(self):
        '''
        Get the best dist and its attrs, and conduct a ks test.
        '''
        
        best = self._fitness.iloc[0,:]
        name = best.name
        mu,sigma = best['loc'], best['scale']
        args = best['arg']
        n = len(self.data)
        
        samples = gen_sample(name, (int(1e4),n), mu, sigma, args)
        sample = samples.mean(0)
        kstest_stat, kstest_p = ks_2samp(self.data, sample)
        print('----------------------------------------')
        print(f'The dist with the best fitness is: {name}')
        print(f'Var: {self.name}')
        print('Params')
        print(f'Mean: {mu}')
        print(f'Std: {sigma}')
        if args:
            print(f'Other Args: {args}')
        print(f'KS test p-value: {kstest_p}')
        
        return {
            'name': name,
            'mean': mu,
            'sigma': sigma,
            'args': args,
            'ks test': (kstest_stat, kstest_p),
            }
        
    def plot_dists(self):
        '''
        Plot the PDF of the fitted dists and the original data.
        Currently, this method only supports fitting of uniform and 
        normal dists.
        '''
        
        plt.hist(
            self.data,
            density=True,
            color='#81d8d0',
            )
        xmin, xmax = plt.xlim()
        x = np.linspace(xmin, xmax, 100)
        
        fit_summary = self._fitness
        for name in fit_summary.index:
            params = fit_summary.loc[name,:]
            mu,sigma = params['loc'], params['scale']
            args = params['arg']
        
            p = gen_pdf(
                name, 
                x, 
                mu, 
                sigma, 
                args,
                )
            plt.plot(
                x, 
                p, 
                linewidth=2, 
                color=palette[name], 
                label=name,
                )
            
        plt.title('Fit Result')
        plt.legend(loc='upper right')
        plt.show()

class MonteCarloSim:
    def __init__(
            self, 
            data_path: str, 
            by_default=True, 
            custom_params=None,
            ):
        '''
        Initialise the MonteCarloSim instance.

        Parameters
        ----------
        data_path : str
            The path leading to the original data.
            The format should be exactly the same as the data.xlsx.
        by_default : bool, optional
            Fit the dists for net revenues, depreciation of fixed assets,
            and amortisation of intangible assets automatically. If False,
            set the attrs of the dists for all three vars through custom_params.
        custom_params : dist, optional
            Customised dists of three vars. Default None.
        '''
        
        df = pd.read_excel(data_path)
        net_rev = DataProperties(df.loc[:,'net_rev'])
        depr_FA = DataProperties(df.loc[:,'depr_FA'])
        amor_IA = DataProperties(df.loc[:,'amor_IA'])
        if by_default:
            sim_params = {
                'net_rev': net_rev.get_best_dist(),
                'depr_FA': depr_FA.get_best_dist(),
                'amor_IA': amor_IA.get_best_dist(),
                }
        else:
            sim_params = custom_params
        self.sim_params = sim_params
            
    def run(
            self, 
            WACC, 
            N=1e6, 
            T=10, 
            show=True,
            ):
        '''
        Run the sim using given params.

        Parameters
        ----------
        WACC : float
            Weighted average cost of capital
        N : int, optional
            # sims. Default 1e6.
        T : int, optional
            # periods per sim. Default 10.
        show : bool, optional
            Show the info of sim and the mean path. Default True.

        Returns
        -------
        sigma : float
            Estimated volatility of free cash flow

        '''
        print('----------------------------------------')
        start = time.time()
        N = int(N)
        FCF = np.zeros((N,T))
        samples = {}
        discount_r = np.cumprod(
            np.ones((N,T))/(1+WACC),
            axis=1
            )
        
        for key, params in self.sim_params.items():
            sample = gen_sample(
                params['name'], 
                (N,T), 
                params['mean'], 
                params['sigma'],
                params['args']
                )
            FCF += sample
            samples[key] = sample
        samples['FCF'] = FCF
        
        NPV = np.multiply(FCF, discount_r)
        samples['NPV'] = NPV
        mean_path = NPV.mean(0)
        sigma = stdev(mean_path)/(mean(mean_path) * sqrt(T))
        print(f'SIMULATION ENDED. COST {time.time()-start}s')
        
        if show:
            print('----------------------------------------')
            show_info(N, T, sigma, mean_path)
        self.sim_results = samples
        
        return sigma
    
    def plot_paths(
            self, 
            var, 
            num_paths, 
            type_ = 'individual',
            ):
        '''
        Plot the paths of sims.

        Parameters
        ----------
        var : str
            The paths of which var to be shown. Show be one of the following:
                - net_rev
                - depr_FA
                - amor_IA
        num_paths : int
            # paths to be shown
        type_ : str, optional
            Show mean path or individual paths. Default 'individual'.

        Returns
        -------
        None.
        '''
        
        paths = self.sim_results[var][np.random.randint(0,1e6,int(num_paths))]
        N,T = paths.shape 
        x = np.linspace(0, T, T, dtype=int)
        
        if type_ == 'individual':
            for idx in range(len(paths)):
                plt.plot(x, paths[idx,:])
            plt.title(f'paths of {var}, n = {num_paths}')
        elif type_ == 'average':
            path = paths.mean(0)
            plt.plot(x, path)
            plt.title(f'mean path of {var}')
    
    def sim_efficiency(
            self, 
            WACC, 
            bottom, 
            top, 
            num = 10,
            ):
        '''
        Show how volatility estimated changes as the # sims increases.

        Parameters
        ----------
        WACC : float
            Weighted average cost of capital
        bottom : int
            Min # sims
        top : int
            Max # sims
        num : int, optional
            # points

        Returns
        -------
        None.
        '''
        
        seps = np.linspace(int(bottom), int(top), int(num))
        sigmas = [self.run(WACC, N=int(sep), show = False)
                  for sep in seps]
        print(sigmas)
        plt.plot(seps, sigmas)
        plt.title('#Simulations and Votalities')
    
    @staticmethod
    def module_info():
        with open('./module_info.txt','r', encoding = 'utf8') as f:
            text = f.read()
            print(text)