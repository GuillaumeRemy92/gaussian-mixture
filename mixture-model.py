
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math

def dist_sq(list1, list2):    
    
    ans = 0
        
    for i in range(0, len(list1)):
        ans = ans + (list1[i] - list2[i])**2
    return ans

def gaussian_density(mean, var, point):
    l = len(mean)
    dist_sq = 0
    for i in range(0,l):
        dist_sq = dist_sq + (mean[i] - point[i])**2
    ans = np.exp(- (0.5/var)* dist_sq)
    return ans


class BaseDistribution:
    
    def __init__(self, dim_space, num_points):
        self.dim_space = dim_space
        self.num_points = num_points
        self.list_points = [[0]*dim_space]*num_points
        
        self.center_clu = []
        self.proba_vec = []
    
    def print_points(self):
        print(self.list_points)
        
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
        
    
    def add_gauss_noise(self, var):
        
        std = np.sqrt(var)
        
        for i in range(0, self.num_points):
            vec_gauss = np.random.normal(0, std, self.dim_space)
            self.list_points[i] = [sum(x) for x in zip(self.list_points[i], vec_gauss)]
            
    
    def twod_plot(self, dim1, dim2):
        
        list_points_t = np.transpose(self.list_points)
        plt.scatter(list_points_t[dim1], list_points_t[dim2])
        plt.show()
        

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
        
    
    
    def wgf_update_iter(self, var, eps, iter_num):
    
        for i in range(0 , iter_num):
            self.wgf_update(var, eps)
    
        
 

