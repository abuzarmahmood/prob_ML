#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 15 20:42:16 2018

@author: abuzarmahmood
"""
"""
Machine Learning 10-601
Tom Mitchell Spring 2015
Problem Set 2
"""


import numpy as np
import pylab as plt
"""
   ____  ___  
  / __ \|__ \ 
 | |  | |  ) |
 | |  | | / / 
 | |__| |/ /_ 
  \___\_\____|
"""
# Part B
n = 10
ons = 6
offs = 4

bern_likelihood = lambda theta,ons,offs: (theta**ons)*((1-theta)**offs)

theta_vec = np.arange(0,1,0.01)
lik_vec = np.empty(theta_vec.size)
for i in range(len(theta_vec)):
    lik_vec[i] = np.log(bern_likelihood(theta_vec[i],ons,offs))
plt.plot(theta_vec,lik_vec)

# Part C
theta_mle = theta_vec[np.nanargmax(lik_vec)]
plt.plot(theta_vec,lik_vec)
theta_mle = theta_vec[np.nanargmax(lik_vec)]
plt.vlines(theta_mle,ymin = np.nanmin(lik_vec[np.isfinite(lik_vec)]), ymax = np.nanmax(lik_vec)*1.1,colors='r')
plt.title('MLE = %.2f' % theta_mle)

# Part D
theta_vec = np.arange(0,1,0.01)
sets = dict([('ons',[3,60,5]),('offs',[2,40,5])])
for i in range(len(sets['ons'])):
    ons = sets['ons'][i]
    offs = sets['offs'][i]
    lik_vec = np.empty(theta_vec.size)
    for i in range(len(theta_vec)):
        lik_vec[i] = bern_likelihood(theta_vec[i],ons,offs)
    plt.figure()
    plt.plot(theta_vec,lik_vec)
    theta_mle = theta_vec[np.argmax(lik_vec)]
    plt.vlines(theta_mle,ymin = 0, ymax = np.max(lik_vec)*1.1,colors='r')
    plt.title('MLE = %.2f' % theta_vec[np.argmax(lik_vec)])

# Part f,g,h
n = 10
ons = 6
offs = 4
prior = lambda theta: np.log((theta**2)*((1-theta)**2)/0.0333)
posterior = lambda theta, ons, offs: np.log(bern_likelihood(theta,ons,offs))+np.log(prior(theta))
theta_vec = np.arange(0,1,0.01)
map_vec = np.empty(theta_vec.size)
for i in range(len(theta_vec)):
    map_vec[i] = posterior(theta_vec[i],ons,offs)
plt.plot(theta_vec,map_vec)
plt.title('MAP = %.3f' % theta_vec[np.nanargmax(map_vec)])
plt.vlines(theta_vec[np.nanargmax(map_vec)],ymin = np.nanmin(map_vec), ymax = np.nanmax(map_vec)*1.1,colors='r')

# Part i
ons = 600
offs = 400 
prior = lambda theta: np.log((theta**2)*((1-theta)**2)/0.0333)
posterior = lambda theta, ons, offs: np.log(bern_likelihood(theta,ons,offs))+np.log(prior(theta))
theta_vec = np.arange(0,1,0.01)
map_vec = np.empty(theta_vec.size)
for i in range(len(theta_vec)):
    map_vec[i] = posterior(theta_vec[i],ons,offs)
plt.plot(theta_vec,map_vec)
plt.title('MAP = %.3f' % theta_vec[np.nanargmax(map_vec)])

#plt.figure()
lik_vec = np.empty(theta_vec.size)
for i in range(len(theta_vec)):
    lik_vec[i] = bern_likelihood(theta_vec[i],ons,offs)
plt.plot(theta_vec,np.log(lik_vec))
theta_mle = theta_vec[np.argmax(lik_vec)]
plt.title('MLE = %.3f' % theta_vec[np.argmax(lik_vec)])