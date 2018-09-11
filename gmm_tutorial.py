# GMM tutorial
import numpy as np
from numpy.random import multivariate_normal as mvn
from numpy.random import normal
import matplotlib.pyplot as plt
import sklearn

import multiprocessing as mp
from matplotlib import animation
#  __ _____  
# /_ |  __ \ 
#  | | |  | |
#  | | |  | |
#  | | |__| |
#  |_|_____/ 
#

# For every cluster, calculate the proabability that all points belong to that cluster
# Find the new mean and variance of the cluster by using the responsibilities calculated in the previous step
class gmm_1d:
    def __init__(self,data,clusters, max_iters, threshold):
        self.data = data
        self.clusters = clusters
        self.mean = np.random.random(clusters)*np.mean(data)
        self.var = np.random.random(clusters)*np.std(data)
        self.probs = np.zeros((clusters,data.size)) # Probabilty of point belonging to cluster
        self.resps = np.zeros((clusters,data.size)) # Responsibility of each cluster (normalized probs)
        self.max_iters = max_iters
        self.iters = 0
        self.threshold = threshold
        self.likelihood = [0,1]
    
    def normal_func(x,mean,var):
        return (1/np.sqrt(2*np.pi*var))*np.exp((-(x-mean)**2)/(2*var))
    
    def plot_data(self):
        plt.figure()
        for cluster in range(self.clusters):
            x_range = np.linspace(np.min(self.data), np.max(self.data),100)
            y_range = normal_func(x_range, self.mean[cluster], self.var[cluster])
            plt.plot(x_range,y_range)
        plt.scatter(self.data,(np.random.random(self.data.size))*0.1)
    
    def e_step(self):
        for cluster in range(self.clusters):
            for point in range(self.data.size):
                self.probs[cluster,point] = normal_func(self.data[point],self.mean[cluster],self.var[cluster])
        self.likelihood.append(np.sum(np.log(np.sum(np.multiply(self.probs,np.divide(self.probs,np.sum(self.probs,axis=0))),axis=0))))
        
    
    def m_step(self):
        self.resps = np.divide(self.probs,np.sum(self.probs,axis=0))
        for cluster in range(self.clusters):
            self.mean[cluster] = np.sum(np.multiply(self.data,self.resps[cluster,:]))/np.sum(self.resps[cluster,:])
            self.var[cluster] = np.sum(np.multiply((self.data-self.mean[cluster])**2,self.resps[cluster,:]))/np.sum(self.resps[cluster,:])
            
    def fit_model(self):
        while (np.abs(self.likelihood[-1] - self.likelihood[-2]) > self.threshold) and (self.iters < self.max_iters):
            self.e_step()
            self.m_step()
            #self.plot_data()
            model.iters += 1

def run_gmm_1d(data,clusters, max_iters, threshold,seed):
    np.random.seed(seed)
    model = gmm_1d(data,clusters, max_iters, threshold)
    model.fit_model()
    return model

def run_gmm_1d_multi(data,clusters, max_iters, threshold, num_seeds):
    pool = mp.Pool(processes = mp.cpu_count())
    results = [pool.apply_async(run_gmm_1d, args = (data,clusters, max_iters, threshold,i)) for i in range(num_seeds)]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    
    all_liks = []
    for seed in output:
        all_liks.append(seed.likelihood[-1])
    fin_model = output[np.nanargmax(all_liks)]  
    
    return fin_model

  
# Generate data
normal_func = lambda x,mean,var : (1/np.sqrt(2*np.pi*var))*np.exp((-(x-mean)**2)/(2*var))
means = np.random.random(4)*20
var = np.random.random(4)
x_lab = [normal(means[i],var[i],300) for i in range(len(means))]
for i in range(len(x_lab)):
    plt.scatter(x_lab[i],(np.random.random(x_lab[i].size))*0.1)
    x_range = np.linspace(np.min(np.asarray(x_lab)), np.max(np.asarray(x_lab)),100)
    y_range = normal_func(x_range, means[i], var[i])
    plt.plot(x_range,y_range)
    
x = np.asarray(x_lab).flatten()

model = run_gmm_1d_multi(x,4,1000,1e-6,20)
print(model.likelihood[-1])
model.plot_data()
     
    

#  __  __       _ _   _        _____  _           
# |  \/  |     | | | (_)      |  __ \(_)          
# | \  / |_   _| | |_ _ ______| |  | |_ _ __ ___  
# | |\/| | | | | | __| |______| |  | | | '_ ` _ \ 
# | |  | | |_| | | |_| |      | |__| | | | | | | |
# |_|  |_|\__,_|_|\__|_|      |_____/|_|_| |_| |_|
#

means = np.asarray([[1,2],[7,8],[1,20]])
# Array gets reshaped as [1,2,3,4]
# | 1 2 |
# | 3 4 |
covs = [[2,1,1,2],[5,2,2,5],[1,1,1,30]]
covs = np.asarray([np.reshape(np.array(i),(2,2)) for i in covs])

# Labelled data for plotting
x_lab=[]
x_lab = [mvn(means[i,:],covs[i,:,:],500) for i in range(len(means))]
for i in range(len(means)):
    plt.scatter(x_lab[i][:,0],x_lab[i][:,1])

# Unlabelled data for EM
x = np.asarray(x_lab)
x = np.reshape(x,(x.shape[2],x.shape[0]*x.shape[1]))
plt.scatter(x[:,0],x[:,1])

class gmm_mv:
    
    def __init__(self, data, dims, clusters, max_iters, threshold):
        self.data = data
        self.dims = dims
        self.clusters = clusters
        self.mean = np.random.rand((dims,clusters))*np.mean(data)
        self.var = np.random.random((dims,clusters))*np.std(data)
        self.probs = np.zeros((dims,clusters,data.size))
        self.max_iters = max_iters
        self.iters = 0
        self.threshold = threshold
        self.likelihood = [0,1]
        
    def mv_normal_func(x,means,cov_mat): 
        # means is column vectors
        # every column in x is a data point
        # Housekeeping to keep dimensions straight
        means.shape = (len(means),1)
        #x_sub = x - means
        out = np.empty(x.shape[1])
        for i in range(x.shape[1]):    
            out[i] = np.exp(-0.5*np.matmul((x[:,i][:,None]-means).T,np.matmul(np.linalg.inv(cov_mat),(x[:,i][:,None]-means))))/(np.sqrt(((2*np.pi)**means.size)*np.linalg.det(cov_mat)))                                                 
        return out

    def plot_data2d(means,cov_mat):
        means.shape = (len(means),1)
        x_range = np.linspace(np.min(x[0,:]),np.max(x[0,:]),100)
        y_range = np.linspace(np.min(x[1,:]),np.max(x[1,:]),100)
        z = np.empty(len(x_range)*len(y_range))
        i = 0
        for this_x in x_range:
            for this_y in y_range:
                z[i] = md_normal_func(np.asarray([this_x,this_y])[:,None],means,cov_mat)
                i += 1
        xv, yv = np.meshgrid(x_range,y_range)
        z.shape = xv.shape
        plt.contour(xv,yv,z)

test = md_normal_func(np.stack([x_range,y_range]).T,means[0],covs[0])

#   _____                           _         _____        _        
#  / ____|                         | |       |  __ \      | |       
# | |  __  ___ _ __   ___ _ __ __ _| |_ ___  | |  | | __ _| |_ __ _ 
# | | |_ |/ _ \ '_ \ / _ \ '__/ _` | __/ _ \ | |  | |/ _` | __/ _` |
# | |__| |  __/ | | |  __/ | | (_| | ||  __/ | |__| | (_| | || (_| |
#  \_____|\___|_| |_|\___|_|  \__,_|\__\___| |_____/ \__,_|\__\__,_|
#

means = np.asarray([[1,2],[7,8],[1,20]])
# Array gets reshaped as [1,2,3,4]
# | 1 2 |
# | 3 4 |
covs = [[2,1,1,2],[5,2,2,5],[1,1,1,30]]
covs = np.asarray([np.reshape(np.array(i),(2,2)) for i in covs])

# Labelled data for plotting
x_lab=[]
x_lab = [mvn(means[i,:],covs[i,:,:],500) for i in range(len(means))]
for i in range(len(means)):
    plt.scatter(x_lab[i][:,0],x_lab[i][:,1])

# Unlabelled data for EM
x = np.asarray(x_lab)
x = np.reshape(x,(x.shape[0]*x.shape[1],x.shape[2]))
plt.scatter(x[:,0],x[:,1])
    
#  ______ __  __ 
# |  ____|  \/  |
# | |__  | \  / |
# |  __| | |\/| |
# | |____| |  | |
# |______|_|  |_|
#
# Equations taken from
# http://www.cse.iitm.ac.in/~vplab/courses/DVP/PDF/gmm.pdf
# Pg 9
# Numer = numerator, denom = denominator

########## Define functions ##########
def unit_lik(x_vec,mu_vec,sig_mat):
    val = np.exp(-0.5*np.matmul(np.matmul(np.subtract(x_vec,mu_vec), np.linalg.inv(sig_mat)),np.subtract(x_vec,mu_vec))) / (np.sqrt(np.linalg.det(sig_mat))*((2*np.pi)**2))
    return val

# dat = N X dims
# mu = clusts X dims
# sig = (dims X clusts) X clusts, so 6x2 for 2D with 3 clusters
def dat_log_lik(dat,mu,sig,pi):
    all_log_liks = np.zeros((dat.shape[0],len(pi)))
    for dat_point in range(dat.shape[0]):
        for clust in range(len(pi)):
            all_log_liks[dat_point,clust] = unit_lik(x[dat_point,:],mu[clust,:],sig[(clust*2):(clust*2)+2,:])
    fin_log_lik = np.sum(np.log(np.matmul(all_log_liks,pi)))
    return fin_log_lik

def resp(dat,mu,sig,pi):
    all_resps = np.zeros((dat.shape[0],len(pi)))
    for dat_point in range(dat.shape[0]):
        for clust in range(len(pi)):
            all_resps[dat_point,clust] = unit_lik(x[dat_point,:],mu[clust,:],sig[(clust*2):(clust*2)+2,:])
    numer = np.multiply(all_resps,pi)
    denom = np.sum(numer,axis=1)
    fin_resps = np.zeros((len(denom),len(pi)))
    for i in range(len(denom)):
        fin_resps[i] = numer[i,:]/denom[i]
        
    return fin_resps

def mu_prime(dat, resps):
    mu_p = np.zeros((resps.shape[1],2))
    for i in range(dat.shape[1]):
        mu_p[:,i] = np.divide(np.matmul(dat[:,i].T,resps),np.sum(resps,axis=0))
    return mu_p

def sig_prime(dat, mu_prime, resps):
    sig_p = np.zeros((resps.shape[1]*2,2))
    for clust in range(resps.shape[1]):
        all_numer = np.ndarray((dat.shape[1],dat.shape[1],dat.shape[0]))
        for dat_point in range(dat.shape[0]):
            temp_var = np.reshape(dat[dat_point,:] - mu_prime[clust,:],(2,1))
            all_numer[:,:,dat_point] = np.matmul(temp_var,temp_var.T)*resps[dat_point,clust]
        sig_p[(clust*2):(clust*2)+2,:] = np.sum(all_numer,axis=2)/np.sum(resps[:,clust]) 
    return sig_p

def pi_prime(resps):
    pi_p = np.mean(resps,axis=0)
    return pi_p

def plot_dat(dat,mu,sig,bins):
    fig = plt.figure()
    plt.scatter(dat[:,0],dat[:,1])
    x_min, x_max = np.min(dat[:,0]), np.max(dat[:,0])
    y_min, y_max = np.min(dat[:,1]), np.max(dat[:,1])
    x = np.arange(x_min,x_max,(x_max-x_min)/bins)
    y = np.arange(y_min,y_max,(y_max-y_min)/bins)
    X, Y = np.meshgrid(x, y)
    
    x_p = np.reshape(X,X.shape[0]**2)
    y_p = np.reshape(Y,Y.shape[0]**2)
    xy = np.unique(np.stack((x_p,y_p)),axis=0).T
    
    Z = np.zeros((xy.shape[0],mu.shape[0]))
    for dat_point in range(xy.shape[0]):
        for clust in range(mu.shape[0]):
             Z[dat_point,clust] = unit_lik(xy[dat_point,:],mu[clust,:],sig[(clust*2):(clust*2)+2,:])
    
    for clust in range(Z.shape[1]):
       plt.contour(X,Y,np.reshape(Z[:,clust],(bins,bins)))
       
    fig.show()
    
       
    
    
        

############## Initialize parameters #################
clusters = 3

mu0_min, mu0_max = np.min(x[:,0]), np.max(x[:,0])
mu1_min, mu1_max = np.min(x[:,1]), np.max(x[:,1])
mu0_in = (np.random.random((clusters,1))*(mu0_max-mu0_min))+ mu0_min
mu1_in = (np.random.random((clusters,1))*(mu1_max-mu1_min))+ mu1_min
mu_in = np.concatenate((mu0_in,mu1_in),axis=1)

dat_var0, dat_var1 = np.var(x[:,0])/clusters, np.var(x[:,1])/clusters
sig_in = np.tile(np.reshape(np.asarray([dat_var0,0,0,dat_var1]),(2,2)),(clusters,1))

pi_in = np.random.rand(3)
pi_in = pi_in/np.sum(pi_in)
      
########### Run EM #############    
  
tolerance = 1e-9
max_iters = 300  

all_log_lik = []
all_log_lik.append(0)
all_log_lik.append(dat_log_lik(x,mu_in,sig_in,pi_in))
resps = resp(x,mu_in,sig_in,pi_in)

while np.abs(all_log_lik[-1] - all_log_lik[-2]) > tolerance:
#while len(all_log_lik) < max_iters:
    mu_p = mu_prime(x,resps)
    sig_p = sig_prime(x,mu_p,resps)
    pi_p = pi_prime(resps)
    resps_p = resp(x,mu_p,sig_p,pi_p)
    all_log_lik.append(dat_log_lik(x,mu_p,sig_p,pi_p))
    print(np.abs(all_log_lik[-1] - all_log_lik[-2]))
    plot_dat(x,mu_p,sig_p,100)


# Test case
# =============================================================================
# x1_vec = np.asarray([1.1,2.1])
# x2_vec = np.asarray([0.5,0.5])
# mu_vec = np.asarray([1,2])
# sig_mat = np.reshape(np.asarray([2,1,1,2]),(2,2))
# 
# print(unit_lik(x1_vec,mu_vec,sig_mat))
# print(unit_lik(x2_vec,mu_vec,sig_mat))
# =============================================================================
    

