# GMM tutorial
import numpy as np
from numpy.random import multivariate_normal as mvn
from numpy.random import normal
import matplotlib.pyplot as plt
import sklearn

import multiprocessing as mp
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
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
     
###############################################################################
###############################################################################    

#  __  __       _ _   _        _____  _           
# |  \/  |     | | | (_)      |  __ \(_)          
# | \  / |_   _| | |_ _ ______| |  | |_ _ __ ___  
# | |\/| | | | | | __| |______| |  | | | '_ ` _ \ 
# | |  | | |_| | | |_| |      | |__| | | | | | | |
# |_|  |_|\__,_|_|\__|_|      |_____/|_|_| |_| |_|
#

class gmm_mv:
    
    def __init__(self, data, dims, clusters, max_iters, threshold):
        self.data = data
        self.dims = dims
        self.clusters = clusters
        self.mean = [np.random.rand(dims)*np.max(data,axis=-1) for i in range(clusters)]
        self.var = [np.eye(dims)*np.std(data) for i in range(clusters)]
        self.probs = np.zeros((clusters,data.shape[-1]))
        self.resps = np.zeros((clusters,data.shape[-1]))
        self.max_iters = max_iters
        self.iters = 0
        self.threshold = threshold
        self.log_likelihood = [0,1]
        mixing = np.random.random(clusters)
        self.mixing = (mixing/np.sum(mixing))[:,None]
        
    def mv_normal_func(self,x,means,cov_mat): 
        # means and x are column vectors
        # Housekeeping to keep dimensions straight
        means.shape = (means.size,1)
        x.shape = (len(x),1)
        x_sub = x - means 
        return np.exp(-0.5*np.matmul(x_sub.T,np.matmul(np.linalg.inv(cov_mat),x_sub)))/(np.sqrt(((2*np.pi)**means.size)*np.linalg.det(cov_mat)))                                                 

    def plot_data2d(self):
        for cluster in range(self.clusters):
            means = self.mean[cluster]
            means.shape = (means.size,1)
            cov_mat = self.var[cluster]
            x_range = np.linspace(np.min(self.data[0,:]),np.max(self.data[0,:]),100)
            y_range = np.linspace(np.min(self.data[1,:]),np.max(self.data[1,:]),100)
            z = np.empty(len(x_range)*len(y_range))
            i = 0
            for this_x in x_range:
                for this_y in y_range:
                    z[i] = self.mv_normal_func(np.asarray([this_x,this_y])[:,None],means,cov_mat)
                    i += 1
            xv, yv = np.meshgrid(x_range,y_range)
            z.shape = xv.shape
            plt.contour(xv,yv,z)
        plt.scatter(self.data[0,:],self.data[1,:])
        
    def e_step(self):
        for cluster in range(self.clusters):
            for point in range(self.data.shape[-1]):
                self.probs[cluster,point] = self.mv_normal_func(self.data[:,point],self.mean[cluster],self.var[cluster])
        temp_resps = np.multiply(self.probs,self.mixing)
        self.resps = np.divide(temp_resps,np.sum(temp_resps,axis=0))
    
    def m_step(self):
        for cluster in range(self.clusters):
            self.mean[cluster] = np.divide(np.sum(np.multiply(self.data,self.resps[cluster,:]),axis=-1),np.sum(self.resps[cluster,:]))
            x_sub_sq = (self.data-self.mean[cluster][:,None])
            cov_array = np.zeros((x_sub_sq.shape[0],x_sub_sq.shape[0],self.data.shape[-1]))
            for i in range(cov_array.shape[-1]):
                cov_array[:,:,i] = np.matmul(x_sub_sq[:,i][:,None],x_sub_sq[:,i][:,None].T)
            self.var[cluster] = np.sum(np.multiply(cov_array,self.resps[cluster,:]),axis=-1)/np.sum(self.resps[cluster,:])
        self.mixing = (np.sum(self.resps,axis = 1)/ np.sum(self.resps))[:,None]
        self.log_likelihood.append(np.sum(np.log(np.sum(np.multiply(self.probs,self.mixing),axis=0))))
            
    def fit_model(self):
        while (np.abs(self.log_likelihood[-1] - self.log_likelihood[-2]) > self.threshold) and (self.iters < self.max_iters):
            self.e_step()
            self.m_step()
            #plt.figure()
            #self.plot_data2d()
            model.iters += 1


def mv_normal_func(x,means,cov_mat): 
    # means and x are column vectors
    # Housekeeping to keep dimensions straight
    means.shape = (len(means),1)
    x.shape = (len(x),1)
    x_sub = x - means 
    return np.exp(-0.5*np.matmul(x_sub.T,np.matmul(np.linalg.inv(cov_mat),x_sub)))/(np.sqrt(((2*np.pi)**means.size)*np.linalg.det(cov_mat)))                                                 

def plot_data2d(x,means,cov_mat):
    means.shape = (means.size,1)
    x_range = np.linspace(np.min(x[0,:]),np.max(x[0,:]),100)
    y_range = np.linspace(np.min(x[1,:]),np.max(x[1,:]),100)
    z = np.empty(len(x_range)*len(y_range))
    i = 0
    for this_x in x_range:
        for this_y in y_range:
            z[i] = mv_normal_func(np.asarray([this_x,this_y])[:,None],means,cov_mat)
            i += 1
    xv, yv = np.meshgrid(x_range,y_range)
    z.shape = xv.shape
    plt.contour(xv,yv,z)

def run_gmm_mv(data, dims, clusters, max_iters, threshold, seed):
    np.random.seed(seed)
    model = gmm_mv(data, dims, clusters, max_iters, threshold)
    model.fit_model()
    return model

def run_gmm_mv_multi(data, dims, clusters, max_iters, threshold, num_seeds):
    pool = mp.Pool(processes = mp.cpu_count())
    results = [pool.apply_async(run_gmm_mv, args = (data, dims, clusters, max_iters, threshold,i)) for i in range(num_seeds)]
    output = [p.get() for p in results]
    pool.close()
    pool.join()
    
    all_liks = []
    for seed in output:
        all_liks.append(seed.log_likelihood[-1])
    fin_model = output[np.nanargmax(all_liks)]  
    
    return fin_model


dims = 3
clusters = 3
means = [np.random.rand(1,dims)*10 for i in range(clusters)]
covs = [np.eye(dims)*np.random.random()*2 for i in range(clusters)]

# Labelled data for plotting
x_lab=[]
x_lab = [mvn(means[i].flatten(),covs[i],500) for i in range(len(means))]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
for i in range(len(means)):
    ax.scatter(x_lab[i][:,0],x_lab[i][:,1],x_lab[i][:,2])
    #plt.scatter(x_lab[i][:,0],x_lab[i][:,1])
    #plot_data2d(x_lab[i].T, means[i],covs[i])


# Unlabelled data for EM
x = np.vstack(x_lab).T

# Plotting is funky for some reason
# Estimated paramters are correct!!
model = run_gmm_mv_multi(x,dims,clusters,1e3,1e-9,20)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
col_map = plt.get_cmap('viridis')
colors = np.sum(np.multiply(model.resps,np.asarray(range(1,clusters+1))[:,None]),axis=0)/clusters
#colors = np.multiply(model.resps,np.asarray(range(1,clusters+1))[:,None])/clusters
ax.scatter(xs=x[:,0],ys=x[:,1],zs=x[:,2],c=col_map(colors))
    

