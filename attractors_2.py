#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import math as m
import matplotlib.pyplot as plt


# In[18]:


def update_1(x,y,a=-1.5,b=1.6,c=1.2,d=0.7):
    # takes floats, returns updated floats
    xnew = m.sin(a * y) + c * m.cos(a * x)
    ynew = m.sin(b * x) + d * m.cos(b * y)
    return xnew, ynew

def update_2(x,y,p):
    # p is a list of parameters
    xnew = p[0][0] + p[0][1] * x + p[0][2] * y + p[0][3] * x * x + p[0][4] * x * y + p[0][5] * y*y
    ynew = p[1][0] + p[1][1] * x + p[1][2] * y + p[1][3] * x * x + p[0][4] * x * y + p[0][5] * y*y

def quad_funding(grid):
    return sum(np.sqrt(grid.flatten())) ** 2 / (grid.shape[0] * grid.shape[1])

def display(grid,id_num):
    # lovely_cmaps = ["YlGn","rainbow", "gnuplot2", "gray", "bone"] # you can loop through these...
    cmap = "bone"
    plt.figure(figsize=(8,8))
    plt.imshow(np.log(grid + 1), cmap=cmap)
    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labelleft=False,
        left=False,
        right=False) # labels along the bottom edge are off
    plt.axis("off") # take off grid
        
    plt.title("a={:.7} b={:.7} c={:.7} d={:.7}\nid={} quality={:.5}".format(a,b,c,d,id_num,quad_funding(grid)),fontsize=12)
    print("Bingo")
    plt.show()
    
    
def display_1(id_num,a,b,c,d,grid_len_px=1024):
    # first we find how thick the grid should be
    max_iter_test_bounds = 10**4
    xarr,yarr = np.zeros(max_iter_test_bounds),np.zeros(max_iter_test_bounds)
    x,y = 0,0
    for i in range(max_iter_test_bounds):
        x,y = update_1(x,y,a,b,c,d)
        xarr[i] = x
        yarr[i] = y
    
    # set the grid bounds
    left, right, up, down = min(xarr),max(xarr),max(yarr),min(yarr)
    width, height = right - left, up - down
    # add padding
    left = left - 0.1 * width
    right = right + 0.1 * width
    up = up + 0.1 * height
    down = down - 0.1 * height
    width,height = right - left , up - down
    
    # initialize the grid
    grid = np.zeros((grid_len_px,grid_len_px))
    max_iter = 10**6 # number of iterations of the generator function
    x,y = 0,0
    for i in range(max_iter):
        x,y = update_1(x,y,a,b,c,d)
        posx = int((x - left) * grid_len_px / width)
        posy = int((y - down) * grid_len_px / height)
        if posx >= 0 and posx < grid_len_px and posy >= 0 and posy < grid_len_px:
            grid[posx][posy] += 1
        else:
            print("something went wrong...",end=" ")
    
    # id_num is for the title
    # lovely_cmaps = ["rainbow","YlGn","gnuplot2"]
    cmap = "YlGn"
    plt.figure(figsize=(8,8))
    plt.imshow(np.log(np.log(grid + 1) + 1), cmap=cmap)
    plt.tick_params(
        axis='both',
        which='both',
        bottom=False,
        top=False,
        labelbottom=False,
        labelleft=False,
        left=False,
        right=False)
    plt.axis('off')
    # plt.suptitle("a={:.4} b={:.4} c={:.4} d={:.4}\nid_num={}".format(a,b,c,d,id_num),fontsize=13)
    print("a={:.4} b={:.4} c={:.4} d={:.4}\nid_num={}".format(a,b,c,d,id_num))
    plt.savefig("figures/id_{}_a_{:.4}_b_{:.4}_c_{:.4}_d_{:.4}.png".format(id_num,a,b,c,d))
    plt.show()
    
def display_2(id_num,a,b,c,d):
    # first we find how thick the grid should be
    max_iter_test_bounds = 10**4
    xarr,yarr = np.zeros(max_iter_test_bounds),np.zeros(max_iter_test_bounds)
    x,y = 0,0
    for i in range(max_iter_test_bounds):
        x,y = update_1(x,y,a,b,c,d)
        xarr[i] = x
        yarr[i] = y
    
    # set the grid bounds
    left, right, up, down = min(xarr),max(xarr),max(yarr),min(yarr)
    width, height = right - left, up - down
    # add padding
    left = left - 0.1 * width
    right = right + 0.1 * width
    up = up + 0.1 * height
    down = down - 0.1 * height
    width,height = right - left , up - down
    
    # initialize the grid
    grid_len_px = 4096
    grid = np.zeros((grid_len_px,grid_len_px))
    max_iter = 5*10**6 # number of iterations of the generator function
    x,y = 0,0
    for i in range(max_iter):
        x,y = update_1(x,y,a,b,c,d)
        posx = int((x - left) * grid_len_px / width)
        posy = int((y - down) * grid_len_px / height)
        if posx >= 0 and posx < grid_len_px and posy >= 0 and posy < grid_len_px:
            grid[posx][posy] += 1
        else:
            print("something went wrong...",end=" ")
    
    # id_num is for the title
    lovely_cmaps = ["rainbow","YlGn","gnuplot2"]
    plt.subplots(figsize=(42,16))
    for idx,cmap in enumerate(lovely_cmaps):
        plt.subplot(1,3,idx+1)
        plt.imshow(np.log(np.log(grid + 1) + 1), cmap=cmap)
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False,
            labelleft=False,
            left=False,
            right=False)
        # plt.axis('off')
    # plt.suptitle("a={:.4} b={:.4} c={:.4} d={:.4}\nid_num={}".format(a,b,c,d,id_num),fontsize=13)
    print("a={:.4} b={:.4} c={:.4} d={:.4}\nid_num={}".format(a,b,c,d,id_num))
    plt.savefig("id_num_{}_a_{:.4}_b_{:.4}_c_{:.4}_d_{:.4}.png".format(id_num,a,b,c,d))
    plt.show()
    
def pretty_display(grid):
    lovely_cmaps = ["YlGn","rainbow", "gnuplot2"] # you can loop through these...
    for cmap in lovely_cmaps:
        plt.figure(figsize=(15,15))
        plt.imshow(np.log(grid + 1), cmap=cmap)
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom=False,      # ticks along the bottom edge are off
            top=False,         # ticks along the top edge are off
            labelbottom=False,
            labelleft=False,
            left=False,
            right=False) # labels along the bottom edge are off

        plt.axis("off")
        plt.title("a={} b={} c={} d={}".format(a,b,c,d))
        print("convergence_orbweaver_{}.png".format(cmap))
        plt.show()


# In[62]:


def plot_2d_grid(c,d):
    ab = np.zeros((30,30))
    for ia,a in enumerate(np.linspace(-6,6,30)):
        print(ia)
        for ib,b in enumerate(np.linspace(-6,6,30)):
            g = small_iter_grid(a,b,c,d)
            ab[ia][ib] = quad_funding(g)
            plt.figure(figsize=(6,6))
            plt.imshow(g)
            plt.show()
    plt.figure(figsize=(10,10))
    plt.imshow(ab)
    plt.xlabel("a or b")
    plt.ylabel("b or a")
    plt.savefig("2dslice_c_{:.2}_d_{:.2}.png".format(c,d))
    plt.show()
    
plot_2d_grid(-1.08,1.207)


# In[48]:


# returns small grid after small amount of iterations
def small_iter_grid(a,b,c,d):
    x,y = 0,0
    grid = np.zeros((256,256))

    for i in range(3*10**3):
        x,y = update_1(x,y,a,b,c,d)
        try: grid[int(x * 40) + 128][int(y * 50 + 128)] += 1
        except: pass
    return grid

def search_and_display(seed_a,seed_b,seed_c,seed_d,grid_len_px=1024):
    trace = []
    a,b,c,d = seed_a,seed_b,seed_c,seed_d
    params = [a,b,c,d]
    stepsize = 0.01
    while True:
        # find the optmial step
        print("Searching for the optimal step to make")
        new_params = params.copy()
        best_params = params.copy()
        best_score = quad_funding(small_iter_grid(a,b,c,d))
        for idx in range(4):
            new_params[idx] += stepsize
            score = quad_funding(small_iter_grid(new_params[0],new_params[1],new_params[2],new_params[3]))
            if score > best_score:
                best_score = score
                best_params = new_params.copy()
            new_params[idx] -= 2*stepsize
            score = quad_funding(small_iter_grid(new_params[0],new_params[1],new_params[2],new_params[3]))
            if score > best_score:
                best_score = score
                best_params = new_params.copy()
            new_params[idx] += stepsize
            
        if params == best_params:
            stepsize *= 1.1
            print("Local maximum")
            continue
        else:
            stepsize /= 1.02
        params = best_params
        a,b,c,d = params
        print("parametrs, QFF", params, quad_funding(small_iter_grid(a,b,c,d))) # best parameters
            
            
        # first we find how thick the grid should be
        max_iter_test_bounds = 3*10**3
        xarr,yarr = np.zeros(max_iter_test_bounds),np.zeros(max_iter_test_bounds)
        x,y = 0,0
        for i in range(max_iter_test_bounds):
            x,y = update_1(x,y,a,b,c,d)
            xarr[i] = x
            yarr[i] = y
    
        # set the grid bounds
        left, right, up, down = min(xarr),max(xarr),max(yarr),min(yarr)
        width, height = right - left, up - down
        # add padding
        left = left - 0.1 * width
        right = right + 0.1 * width
        up = up + 0.1 * height
        down = down - 0.1 * height
        width,height = right - left , up - down
    
        # initialize the grid
        grid = np.zeros((grid_len_px,grid_len_px))
        max_iter = 2*10**5 # number of iterations of the generator function
        x,y = 0,0
        for i in range(max_iter):
            x,y = update_1(x,y,a,b,c,d)
            posx = int((x - left) * grid_len_px / width)
            posy = int((y - down) * grid_len_px / height)
            if posx >= 0 and posx < grid_len_px and posy >= 0 and posy < grid_len_px:
                grid[posx][posy] += 1
            else:
                print("something went wrong...",end=" ")
    
        # id_num is for the title
        # lovely_cmaps = ["rainbow","YlGn","gnuplot2"]
        cmap = "YlGn"
        plt.figure(figsize=(8,8))
        plt.imshow(np.log(np.log(grid + 1) + 1), cmap=cmap)
        plt.tick_params(
            axis='both',
            which='both',
            bottom=False,
            top=False,
            labelbottom=False,
            labelleft=False,
            left=False,
            right=False)
        plt.axis('off')
        plt.show()


# In[46]:


search_and_display(-1.95,-1.269,1.018,0.084)


# In[14]:


def test_run_1(id_num,a,b,c,d):
    # a,b,c,d are the update parameters (floats)
    sidelength = 512
    center = (sidelength // 2 , sidelength // 2)
    grid = np.zeros((sidelength,sidelength))

    outofbounds = False
    x,y = 0,0
    max_iter = 10**5
    for i in range(max_iter):
        x,y = update_1(x,y,a,b,c,d)
        posx = int(x * sidelength / 5) + center[0]
        posy = int(y * sidelength / 5) + center[1]
        if posx < sidelength and posx >= 0 and posy < sidelength and posy >= 0:
            grid[posx][posy] += 1
        elif x**2 + y**2 > (10**4)**2:
            outofbounds = True
            break
        else:
            print(".",end="")
            # print("out of range x={}, y={}".format(posx,posy),end="")
    if outofbounds: 
        print("out_of_bounds",end="")
    else:
        if quad_funding(grid) > 500: # the threshold
            display(grid,id_num)
            return True
        else:
            print("-",end="")
    return False
        
def pretty_display_run(a,b,c,d):
    sidelength = 8192
    center = (sidelength // 2 , sidelength // 2)
    grid = np.zeros((sidelength,sidelength))

    outofbounds = False
    x,y = 0,0
    for i in range(10**8):
        x,y = update_1(x,y,a,b,c,d)
        posx = int(x * sidelength / 5) + center[0]
        posy = int(y * sidelength / 5) + center[1]
        if posx < sidelength and posx >= 0 and posy < sidelength and posy >= 0:
            grid[posx][posy] += 1
        
    pretty_display(grid)
    return grid


# In[28]:


a = 1.9317
b = 1.9248
c = -0.713
d = -1.6801
pretty_display_run(a,b,c,d)


# # Random search

# In[8]:


import pandas as pd


# In[9]:


df = pd.read_csv("params_att_1.csv",header=None)


# In[19]:


aarr = np.linspace(-1.956,-1.698,50)
barr = np.linspace(-1.269,-1.468,50)
carr = np.linspace(1.018,-0.105,50)
darr = np.linspace(0.08399,1.091,50)
for id_n,(a,b,c,d) in enumerate(zip(aarr,barr,carr,darr)):
    display_1("series01-"+str(id_n+10),a,b,c,d)


# In[ ]:


for i in range(df.shape[0]):
    row = df.iloc[i]
    id_num = int(row[0])
    if id_num > 3571: # 5885 and id_num < 5990:
        a,b,c,d = row[1],row[2],row[3],row[4]
        display_2(id_num,a,b,c,d)


# In[11]:


f = open("params_att_2.csv" , "w")
for i in range(10000):
    a = np.random.uniform(-2,2)
    b = np.random.uniform(-2,2)
    c = np.random.uniform(-2,2)
    d = np.random.uniform(-2,2)
    is_attractor = test_run_1(i,a,b,c,d)
    if is_attractor:
        f.write("{},{},{},{},{}\n".format(i,a,b,c,d))

f.close()


# In[8]:


np.random.uniform(1,2)


# In[9]:


np.random.uniform(1,2)


# In[ ]:




