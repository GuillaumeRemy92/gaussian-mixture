
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math

# We start by defining some basic functions

# Given two points of R^d represented as lists of length d, compute the euclidean distance square

def dist_sq(list1, list2):    
    
    ans = 0
        
    for i in range(0, len(list1)):
        ans = ans + (list1[i] - list2[i])**2
    return ans

# Computes value of d-dimensional Gaussian density of given mean and variance at a point, the variance is a scalar 

def gaussian_density(mean, var, point):
    l = len(mean)
    dist_sq = 0
    for i in range(0,l):
        dist_sq = dist_sq + (mean[i] - point[i])**2
    ans = np.exp(- (0.5/var)* dist_sq)
    return ans

# Define a first class to create the basic distributions of points we will be working wwith

class BaseDistribution:
    
    def __init__(self, dim_space, num_points):
        self.dim_space = dim_space
        self.num_points = num_points
        self.list_points = [[0]*dim_space]*num_points
        
        self.center_clu = []
        self.proba_vec = []
    
    def print_points(self):
        print(self.list_points)
        
    # Sample the points uniformly on the unit circle

    def circle_dis(self):
        
        pi = math.pi
        
        new_points = []
        
        for i in range(0, self.num_points):
            angle = rd.random()
            x = math.cos(2*pi*angle)
            y = math.sin(2*pi*angle)
            new_points = new_points + [[x,y]]

        self.dim_space = 2
        self.list_points = new_points
    
    # Sample the points uniformly on a T-shaped figure

    def t_dis(self):
        
        new_points = []
        
        bi_vec = np.random.binomial(1, 0.5, self.num_points)
        uni_vec = np.random.uniform(0, 1, self.num_points)

        
        for i in range(0, self.num_points):
            
            if bi_vec[i]==0:
                x = -0.5 + uni_vec[i]
                y = 0.5
            else:
                x = 0
                y = -0.5 + uni_vec[i]
            
            new_points = new_points + [[x,y]]
            
        self.dim_space = 2
        self.list_points = new_points
        
    # Sample the points according to the distributions corresponding to num_base_pts point masses in the space
    # The location of the point masses are chosen at random

    def set_of_points(self, num_base_pts):
        
        list_base_pts = []
    
        for i in range(0, num_base_pts):
            new_point = []
            for j in range(0, self.dim_space):
                new_point = new_point + [rd.randint(-10, 10)]
            list_base_pts = list_base_pts + [new_point]
                
        proba_vec = []
    
        for i in range(0, num_base_pts):
            proba_vec = proba_vec + [rd.random()]
        
        sum1 = np.sum(proba_vec)
    
        for i in range(0, num_base_pts):
            proba_vec[i] = proba_vec[i]/sum1
            
        self.list_points = rd.choices(list_base_pts, weights = proba_vec, k = self.num_points)
        
        self.proba_vec = proba_vec
        self.center_clu = list_base_pts
    
    # Add gaussian to each coordinate of the every point in the distribution
    # The variance taken as input is a scalar
    
    def add_gauss_noise(self, var):
        
        std = np.sqrt(var)
        
        for i in range(0, self.num_points):
            vec_gauss = np.random.normal(0, std, self.dim_space)
            self.list_points[i] = [sum(x) for x in zip(self.list_points[i], vec_gauss)]
            
    
    # Plots the points in 2D, takes as arguments the index of the two dimension one wishes to plot (in case there are more than two)

    def twod_plot(self, dim1, dim2):
        
        list_points_t = np.transpose(self.list_points)
        plt.scatter(list_points_t[dim1], list_points_t[dim2])
        plt.show()
        

# Define the class that will perform the updates based on the Wasserstein gradient flow

class UpdateAlgo:
    
    def __init__(self, dim_space, num_points, init_points):
        self.dim_space = dim_space
        self.num_points = num_points
        self.init_points = init_points
        self.cur_points = init_points
        
        
    def print_cur_pts(self):
        print(self.cur_points)
    
    
    def twod_plot(self, dim1, dim2):
        
        list_points_t = np.transpose(self.cur_points)
        plt.scatter(list_points_t[dim1], list_points_t[dim2])
        plt.show()
        
    # Performs a single update of the points, takes as input the variance of the Gaussian mixture (needs to be the same as the used to generate 
    # the points) and also the learning rate eps used by the gradient flow

    def wgf_update(self, var, eps):
        
        deno1 = 0
        
        var_inv = 1/var
        
        vec_ans = []
        
        for i in range(0, self.num_points):
            deno1 = deno1 + np.exp(- 0.5 * var_inv * dist_sq(self.init_points[i], self.cur_points[i]))
        
        for i in range(0, self.num_points):
            
            update = [0]*self.dim_space
            
            for j in range(0, self.num_points):
                
                exp1 = np.exp(-0.5 * var_inv * dist_sq(self.cur_points[i], self.init_points[j]))
                
                for k in range(0, self.dim_space):
                    update[k] = update[k] + (self.init_points[j][k] - self.cur_points[i][k])*exp1
                    
            update2 = [x*eps/deno1 for x in update]
            
            new_point = [sum(x) for x in zip(self.cur_points[i], update2)]
            
            vec_ans = vec_ans + [new_point]
            
        self.cur_points = vec_ans
        

    # Performs iter_num times the gradient flow update
    
    def wgf_update_iter(self, var, eps, iter_num):
    
        for i in range(0 , iter_num):
            self.wgf_update(var, eps)
    


# Similarly we implement a class corresponding to the EM algorithm
# Works when the base distributions is a set of point masses, need to specify the number of clusters 


class EmAlgo:
    
    def __init__(self, num_clu, list_points):
        self.num_clu = num_clu
        self.list_points = list_points
        
        self.center_clu = [0]*self.num_clu
        self.var = 1
        self.proba_vec = [0]*self.num_clu
    
    # Function to initialize the parameters to be update by the EM algo,
    # namely the location of the center of the clusters, the probability of each cluster, and the variance of the model

    def init_para(self):
    
        num_points = len(self.list_points)
        dim_space = len(self.list_points[0])
    
        proba_vec = [1/self.num_clu]*self.num_clu
    
    # To initialize the location of the clusters, we just select num_clu points at random from the dataset
    
        loc_cluster = []
    
        for i in range(0, self.num_clu):
            int_rand = rd.randint(0, num_points-1)
            loc_cluster = loc_cluster + [self.list_points[int_rand]]
    
    # We initialize the variance as the variance of the first coordinate of all the points
    
        points_first_cor = np.transpose(self.list_points)[0]
    
        var_init = np.var(points_first_cor)
        
        self.proba_vec = proba_vec
        self.var = var_init
        self.center_clu = loc_cluster
    

    # Function to perform one step of update

    def em_update(self):
        
        num_points = len(self.list_points)
        dim_space = len(self.list_points[0])
    
        # For each point in the data set, we compute the probabiltiy that it is in the cluster k
        # Result stored in proba_pts_cluster
    
        proba_pts_cluster = []
    
        for i in range(0, num_points):
            
            proba_c = []
        
            for k in range(0, self.num_clu):
            
                proba_c = proba_c + [self.proba_vec[k]*gaussian_density(self.center_clu[k], self.var, self.list_points[i])]
            
            sum_c = np.sum(proba_c)
        
            proba_c = [i/sum_c for i in proba_c]
    
            proba_pts_cluster = proba_pts_cluster + [proba_c]
    
    # We now compute the new values of center_clu, var, and proba_vec
    
        proba_pts_clu_sum0 = [0]*self.num_clu
        proba_pts_clu_sum1 = []
        proba_pts_clu_sum2 = []
        
        for k in range(0, self.num_clu):
            
            new_sum1 = [0]*dim_space
            new_sum2 = [0]*dim_space
            
            for i in range(0, num_points):
                
                proba_pts_clu_sum0[k] = proba_pts_clu_sum0[k] + proba_pts_cluster[i][k]
                
                for d in range(0, dim_space):
                    
                    new_sum1[d] = new_sum1[d] + (proba_pts_cluster[i][k])*self.list_points[i][d]
                    new_sum2[d] = new_sum2[d] \
                        + (proba_pts_cluster[i][k])*((self.list_points[i][d] - self.center_clu[k][d])**2)
            
            new_sum1 = [new_sum1[i]/proba_pts_clu_sum0[k] for i in range(0, dim_space)]
            new_sum2 = [new_sum2[i]/proba_pts_clu_sum0[k] for i in range(0, dim_space)]
            
            proba_pts_clu_sum1 = proba_pts_clu_sum1 + [new_sum1]
            proba_pts_clu_sum2 = proba_pts_clu_sum2 + [new_sum2]

        
        proba_vec_out = [proba_pts_clu_sum0[k]/num_points for k in range(0, self.num_clu)]
        
        
        var_out = np.mean(np.transpose(proba_pts_clu_sum2)[0])
    
    
        self.center_clu = proba_pts_clu_sum1
        self.var =var_out
        self.proba_vec = proba_vec_out
        

    # Function to perform iter_nums iteration of the EM algo  

    def em_update_iter(self, iter_nums):
        
        for i in range(0, iter_nums):
            self.em_update()

     
        
# Function to compute log-likelihood for the base distribution corresponding to point masses in space

def log_likelihood(list_points, center_clu, var, proba_vec):
    
    num_points = len(list_points)
    num_clu = len(proba_vec)
    
    ans = 0
    
    for i in range(0, num_points):
        
        new_sum = 0
        
        for k in range(0, num_clu):
            new_sum = new_sum + proba_vec[k]*gaussian_density(center_clu[k], var, list_points[i])
            
        ans = ans + np.log(new_sum)
    
    return ans
   
 

