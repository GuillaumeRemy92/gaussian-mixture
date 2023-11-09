import numpy as np
import matplotlib.pyplot as plt
import random as rd
import math
import scipy

import models

# Generate three models

mod1 = models.BaseDistribution(2, 1000)
mod1.circle_dis()
mod1.add_gauss_noise(0.02)
mod1.twod_plot(0,1)

mod2 = models.BaseDistribution(2, 1000)
mod2.t_dis()
mod2.add_gauss_noise(0.015)
mod2.twod_plot(0,1)

mod3 = models.BaseDistribution(2, 1000)
mod3.set_of_points(5)
mod3.add_gauss_noise(0.5)
mod3.twod_plot(0,1)

mod1 = models.UpdateAlgo(2, 1000, mod1.list_points)
mod2 = models.UpdateAlgo(2, 1000, mod2.list_points)
mod3 = models.UpdateAlgo(2, 1000, mod3.list_points)

# Create plots

nrows = 3
ncols = 4

fig, axs = plt.subplots(nrows, ncols, figsize = (ncols*8, nrows*8), squeeze=False)

list_points_t = np.transpose(mod1.cur_points)
axs[0,0].scatter(list_points_t[0], list_points_t[1])
axs[0,0].axis("off")

for i in range(1, ncols):
    mod1.wgf_update_iter(0.02, 0.7, 40)
    list_points_t = np.transpose(mod1.cur_points)
    axs[0,i].scatter(list_points_t[0], list_points_t[1])
    axs[0,i].axis("off")
    
    
list_points_t = np.transpose(mod2.cur_points)
axs[1,0].scatter(list_points_t[0], list_points_t[1], c="red")
axs[1,0].axis("off")

for i in range(1, ncols):
    mod2.wgf_update_iter(0.015, 0.35, 30)
    list_points_t = np.transpose(mod2.cur_points)
    axs[1,i].scatter(list_points_t[0], list_points_t[1], c="red")
    axs[1,i].axis("off")
    
list_points_t = np.transpose(mod3.cur_points)
axs[2,0].scatter(list_points_t[0], list_points_t[1], c="green")
axs[2,0].axis("off")

for i in range(1, ncols):
    mod3.wgf_update_iter(0.5, 0.7, 40)
    list_points_t = np.transpose(mod3.cur_points)
    axs[2,i].scatter(list_points_t[0], list_points_t[1], c="green")
    axs[2,i].axis("off")