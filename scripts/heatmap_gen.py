import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd

#POINT DATASET
x=[297, 297, 297, 297, 296, 292, 289, 295, 297, 298, 298, 300, 300, 301, 300, 299, 300, 302, 304, 305, 305, 305, 306]
y=[254, 254, 254, 254, 255, 250, 247, 253, 254, 255, 258, 258, 258, 259, 258, 257, 258, 260, 262, 262, 263, 263, 263]

#DEFINE GRID SIZE AND RADIUS(h)
grid_size=1
h=10

#GETTING X,Y MIN AND MAX
x_min=min(x)
x_max=max(x)
y_min=min(y)
y_max=max(y)

#CONSTRUCT GRID
x_grid=np.arange(x_min-h,x_max+h,grid_size)
y_grid=np.arange(y_min-h,y_max+h,grid_size)
x_mesh,y_mesh=np.meshgrid(x_grid,y_grid)

#GRID CENTER POINT
xc=x_mesh+(grid_size/2)
yc=y_mesh+(grid_size/2)

#FUNCTION TO CALCULATE INTENSITY WITH QUARTIC KERNEL
def kde_quartic(d,h):
    dn=d/h
    P=(15/16)*(1-dn**2)**2
    return P

#PROCESSING
intensity_list=[]
for j in range(len(xc)):
    intensity_row=[]
    for k in range(len(xc[0])):
        kde_value_list=[]
        for i in range(len(x)):
            #CALCULATE DISTANCE
            d=math.sqrt((xc[j][k]-x[i])**2+(yc[j][k]-y[i])**2) 
            if d<=h:
                p=kde_quartic(d,h)
            else:
                p=0
            kde_value_list.append(p)
        #SUM ALL INTENSITY VALUE
        p_total=sum(kde_value_list)
        intensity_row.append(p_total)
    intensity_list.append(intensity_row)

#HEATMAP OUTPUT   
img = plt.imread("arroz.jpg") 
#intensity=np.array(intensity_list)
#plt.pcolormesh(x_mesh,y_mesh,intensity)
#plt.plot(x,y,'ro')
#plt.colorbar()
fig, ax = plt.subplots()
ax.imshow(img, extent=[200, 400, 200, 400])
ax.scatter(x, y, color="b")
plt.show()