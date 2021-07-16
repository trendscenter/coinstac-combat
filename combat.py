
import numpy as np
import pandas as pd
import numpy.linalg as la
import math
import copy
import scipy.io

def load_data(urls, index = 0,simulated=False):
    mat = scipy.io.loadmat(urls)
    data = mat['sim_data']
    sites = mat['site_index'].tolist()
    site_ = list(filter(lambda x: x == index, sites[0]))
    filtered = np.zeros([1, 741]) 
    for idx, val in enumerate(sites[0]):
        if val == index:
            filtered= np.vstack((filtered,data[idx,:]))
    filtered = np.delete(filtered, 0, axis=0)         
    return  filtered.T , len(filtered) 

def send_sample_infomation(urls, index = 0):
    mat = scipy.io.loadmat(urls)
    data = mat['sim_data']
    sites = mat['site_index'].tolist()
    site_ = list(filter(lambda x: x == index, sites[0]))
    filtered = np.zeros([1, 741]) 
    for idx, val in enumerate(sites[0]):
        if val == index:
            filtered= np.vstack((filtered,data[idx,:]))
    filtered = np.delete(filtered, 0, axis=0)
    return np.shape(filtered)

def create_covariance_matrix(url, batch_col_name, site_number):
    data_shape = send_sample_infomation(url,site_number)
    site_number = int(site_number)
    covars = pd.DataFrame({ batch_col_name : [site_number]* data_shape[0]})
    return covars, batch_col_name
    
def create_design_matrix(url, batch_col_name, site_number):
    
    def to_categorical(y, nb_classes=None):
        if not nb_classes:
            nb_classes = np.max(y)+1
        Y = np.zeros((len(y), nb_classes))
        for i in range(len(y)):
            Y[i, y[i]] = 1.
        return Y

    covars, batch_col_name = create_covariance_matrix(url,batch_col_name, site_number)
    covar_labels = np.array(covars.columns)
    covars = np.array(covars, dtype='object') 
    batch_col = np.where(covar_labels==batch_col_name)[0][0]
    hstack_list = []
    batch = np.unique(covars[:,batch_col],return_inverse=True)[-1]
    batch_onehot = to_categorical(batch, len(np.unique(batch)))
    hstack_list.append(batch_onehot)
    design = np.hstack(hstack_list)
    (batch_levels, sample_per_batch) = np.unique(covars[:,batch_col],return_counts=True)
    return design, batch_levels, sample_per_batch

def local_grand_mean_with_betas(url, info_dict, batch_col_name, site_number):
    n_batch = 1
    n_sample = info_dict['n_sample']
    X, local_n_sample = load_data(url, index=site_number)
    design, batch_levels, sample_per_batch = create_design_matrix(url,batch_col_name, site_number)
    
    def get_beta_with_nan(yy, mod):
        wh = np.isfinite(yy)
        mod = mod[wh,:]
        yy = yy[wh]
        B = np.dot(np.dot(la.inv(np.dot(mod.T, mod)), mod.T), yy.T)
        return B    
    betas = []
    for i in range(X.shape[0]):
        betas.append(get_beta_with_nan(X[i,:], design))
    B_hat = np.vstack(betas).T
    local_grand_mean = np.dot((sample_per_batch/ float(n_sample)).T, B_hat[:n_batch,:])
    var_pooled = np.dot(((X - np.dot(design, B_hat).T)**2), np.ones((local_n_sample, 1)) / float(n_sample))
    var_pooled[var_pooled==0] = np.median(var_pooled!=0)
    mod_mean = 0
    if design is not None:
        tmp = copy.deepcopy(design)
        tmp[:,range(0,n_batch)] = 0
        mod_mean = np.transpose(np.dot(tmp, B_hat))   
    return local_grand_mean, var_pooled
    
def standarize_data(url, grand_mean, var_pooled, info_dict, batch_col_name, site_number):
    n_batch = 1
    X, local_n_sample = load_data(url,site_number)
    design, batch_levels, sample_per_batch = create_design_matrix(url,batch_col_name, site_number)
    def get_beta_with_nan(yy, mod):
        wh = np.isfinite(yy)
        mod = mod[wh,:]
        yy = yy[wh]
        B = np.dot(np.dot(la.inv(np.dot(mod.T, mod)), mod.T), yy.T)
        return B    
    betas = []
    for i in range(X.shape[0]):
        betas.append(get_beta_with_nan(X[i,:], design))
    B_hat = np.vstack(betas).T    
    if design is not None:
        tmp = copy.deepcopy(design)
        tmp[:,range(0,n_batch)] = 0
        mod_mean = np.transpose(np.dot(tmp, B_hat))    
    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, local_n_sample)))
    s_data = ((X- stand_mean - mod_mean) / np.dot(np.sqrt(var_pooled), np.ones((1, local_n_sample))))
    return s_data, mod_mean

def convert_zeroes(x):
    x[x==0] = 1
    return x

def aprior(delta_hat):
    m = np.mean(delta_hat)
    s2 = np.var(delta_hat,ddof=1)
    return (2 * s2 +m**2) / float(s2)

def bprior(delta_hat):
    m = delta_hat.mean()
    s2 = np.var(delta_hat,ddof=1)
    return (m*s2+m**3)/s2

def fit_LS_model_and_find_priors(url, grand_mean, var_pooled, info_dict, batch_col_name, site_number, index = 0):
    n_batch = 1
    
    covars, batch_col_name = create_covariance_matrix(url,batch_col_name, site_number)
    covar_labels = np.array(covars.columns)
    covars = np.array(covars, dtype='object') 
    batch_col = np.where(covar_labels==batch_col_name)[0][0]
    (batch_levels, sample_per_batch) = np.unique(covars[:,batch_col],return_counts=True)
    batch_info = [list(np.where(covars[:,batch_col]==idx)[0]) for idx in batch_levels]
    
    design, batch_levels, sample_per_batch = create_design_matrix(url,batch_col_name, site_number)
    batch_design = design[:,:n_batch]
    
    
    s_data, mod_mean = standarize_data(url, grand_mean, var_pooled, info_dict, batch_col_name, site_number)
    gamma_hat = np.dot(np.dot(la.inv(np.dot(batch_design.T, batch_design)), batch_design.T), s_data.T)    
    delta_hat = []
    for i, batch_idxs in enumerate(batch_info):
        delta_hat.append(np.var(s_data[:,batch_idxs],axis=1,ddof=1))
    delta_hat = list(map(convert_zeroes,delta_hat))
    gamma_bar = np.mean(gamma_hat, axis=1) 
    t2 = np.var(gamma_hat,axis=1, ddof=1)
    a_prior = list(map(aprior, delta_hat))
    b_prior = list(map(bprior, delta_hat))    
    
    LS_dict = {}
    LS_dict['gamma_hat'] = gamma_hat
    LS_dict['delta_hat'] = delta_hat
    LS_dict['gamma_bar'] = gamma_bar
    LS_dict['t2'] = t2
    LS_dict['a_prior'] = a_prior
    LS_dict['b_prior'] = b_prior
    return LS_dict

def postmean(g_hat, g_bar, n, d_star, t2):
    return (t2*n*g_hat+d_star * g_bar) / (t2*n+d_star)

def postvar(sum2, n, a, b):
    return (0.5 * sum2 + b) / (n / 2.0 + a - 1.0)

def it_sol(sdat, g_hat, d_hat, g_bar, t2, a, b, conv=0.0001):
    n = (1 - np.isnan(sdat)).sum(axis=1)
    g_old = g_hat.copy()
    d_old = d_hat.copy()

    change = 1
    count = 0
    while change > conv:
        g_new = postmean(g_hat, g_bar, n, d_old, t2)
        sum2 = ((sdat - np.dot(g_new.reshape((g_new.shape[0], 1)), np.ones((1, sdat.shape[1])))) ** 2).sum(axis=1)
        d_new = postvar(sum2, n, a, b)

        change = max((abs(g_new - g_old) / g_old).max(), (abs(d_new - d_old) / d_old).max())
        g_old = g_new #.copy()
        d_old = d_new #.copy()
        count = count + 1
    adjust = (g_new, d_new)
    return adjust 

def int_eprior(sdat, g_hat, d_hat):
    r = sdat.shape[0]
    gamma_star, delta_star = [], []
    for i in range(0,r,1):
        g = np.delete(g_hat,i)
        d = np.delete(d_hat,i)
        x = sdat[i,:]
        n = x.shape[0]
        j = np.repeat(1,n)
        A = np.repeat(x, g.shape[0])
        A = A.reshape(n,g.shape[0])
        A = np.transpose(A)
        B = np.repeat(g, n)
        B = B.reshape(g.shape[0],n)
        resid2 = np.square(A-B)
        sum2 = resid2.dot(j)
        LH = 1/(2*math.pi*d)**(n/2)*np.exp(-sum2/(2*d))
        LH = np.nan_to_num(LH)
        gamma_star.append(sum(g*LH)/sum(LH))
        delta_star.append(sum(d*LH)/sum(LH))
    adjust = (gamma_star, delta_star)
    return adjust

def parametric_adjustment(url, grand_mean, var_pooled, info_dict, batch_col_name, site_number, index = 0):
    covars, batch_col_name = create_covariance_matrix(url,batch_col_name, site_number)
    covar_labels = np.array(covars.columns)
    covars = np.array(covars, dtype='object') 
    batch_col = np.where(covar_labels==batch_col_name)[0][0]
    (batch_levels, sample_per_batch) = np.unique(covars[:,batch_col],return_counts=True)
    batch_info = [list(np.where(covars[:,batch_col]==idx)[0]) for idx in batch_levels]
    
    gamma_star, delta_star = [], []
    LS = fit_LS_model_and_find_priors(url, grand_mean, var_pooled, info_dict, batch_col_name, site_number)
    s_data, mod_mean = standarize_data(url, grand_mean, var_pooled, info_dict, batch_col_name, site_number)
    
    for i, batch_idxs in enumerate(batch_info):
        temp = int_eprior(s_data[:,batch_idxs], LS['gamma_hat'][i],
                    LS['delta_hat'][i])
        gamma_star.append(temp[0])
        delta_star.append(temp[1])

    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)
    return gamma_star, delta_star, LS

def adjust_data(url, grand_mean, var_pooled, info_dict, batch_col_name, site_number):
    
    s_data, mod_mean = standarize_data(url, grand_mean, var_pooled, info_dict, batch_col_name, site_number)
    
    dat, local_n_sample = load_data(url,site_number)
    stand_mean = np.dot(grand_mean.T.reshape((len(grand_mean), 1)), np.ones((1, local_n_sample)))

    gamma_star, delta_star, LS = parametric_adjustment(url, grand_mean, var_pooled, info_dict, batch_col_name, site_number)
    
    design, batch_levels, sample_per_batch = create_design_matrix(url,batch_col_name, site_number)
    n_batch = 1
    n_sample = info_dict['n_sample']
    
    covars, batch_col_name = create_covariance_matrix(url,batch_col_name, site_number)
    covar_labels = np.array(covars.columns)
    covars = np.array(covars, dtype='object') 
    batch_col = np.where(covar_labels==batch_col_name)[0][0]
    (batch_levels, sample_per_batch) = np.unique(covars[:,batch_col],return_counts=True)
    batch_info = [list(np.where(covars[:,batch_col]==idx)[0]) for idx in batch_levels]
    
    batch_design = design[:,:n_batch]
    bayesdata = s_data
    gamma_star = np.array(gamma_star)
    delta_star = np.array(delta_star)
    
    for j, batch_idxs in enumerate(batch_info):
        dsq = np.sqrt(delta_star[j,:])
        dsq = dsq.reshape((len(dsq), 1))
        denom = np.dot(dsq, np.ones((1, sample_per_batch[j])))
        numer = np.array(bayesdata[:,batch_idxs] - np.dot(batch_design[batch_idxs,:], gamma_star).T)
        
        bayesdata[:,batch_idxs] = numer / denom

    vpsq = np.sqrt(var_pooled).reshape((len(var_pooled), 1))
    bayesdata = bayesdata * np.dot(vpsq, np.ones((1, local_n_sample))) + stand_mean + mod_mean    

    bayes_data = np.array(bayesdata)
    estimates = {
        "batches": batch_levels,
        "var_pooled": var_pooled, 
        "stand_mean": stand_mean, 
        "mod_mean": mod_mean, 
        "gamma_star": gamma_star, 
        "delta_star": delta_star
    }
    estimates = {**LS, **estimates, }
    return {
        'data': bayes_data,
        'estimates': estimates,
        'info': info_dict
    }
