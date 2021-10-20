#!/usr/bin/env python
# coding: utf-8

# http://paulbourke.net/fractals/clifford/?curius=373 

# In[13]:


import numpy as np
import math as m
import matplotlib.pyplot as plt


# In[65]:


a = -1.5
b = 1.6
c = 1.2
d = 0.7


# In[66]:


def update(x,y):
    # takes floats, returns updated floats
    xnew = m.sin(a * y) + c * m.cos(a * x)
    ynew = m.sin(b * x) + d * m.cos(b * y)
    return xnew, ynew


# In[ ]:





# In[77]:


sidelength = 8192
center = (sidelength // 2 , sidelength // 2)
grid = np.zeros((sidelength,sidelength))

x,y = 0,0
for i in range(30000000):
    x,y = update(x,y)
    posx = int(x * sidelength / 5) + center[0]
    posy = int(y * sidelength / 4) + center[1]
    if posx < sidelength and posx >= 0 and posy < sidelength and posy >= 0:
        grid[posx][posy] += 2
    else:
        print(posx, posy)
#     print(x,y)


# In[74]:


max(grid.flatten()), max(np.log(grid.flatten() + 1))


# In[88]:


lovely_cmaps = ["YlGn","rainbow", "gnuplot2"]
for cmap in lovely_cmaps:
    plt.figure(figsize=(20,20))
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
    plt.savefig("convergence_orbweaver_{}.png".format(cmap))
    print("convergence_orbweaver_{}.png".format(cmap))
    plt.show()


# In[ ]:




