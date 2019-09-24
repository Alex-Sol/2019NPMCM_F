import pandas as pd
import numpy as np

#计算点在投影上的排序位置、离AB线段的距离
def Coordinate_projection(loc_a,loc_b,x,y,z):
    t = ((loc_b.X-loc_a.X)*(x-loc_a.X)+(loc_b.Y-loc_a.Y)*(y-loc_a.Y)+(loc_b.Z-loc_a.Z)*(z-loc_a.Z)) \
            /(np.square(loc_b.X-loc_a.X)+np.square(loc_b.Y-loc_a.Y)+np.square(loc_b.Z-loc_a.Z))
    #t越小，代表离A点越小
    x0 = t*(loc_b.X-loc_a.X)+loc_a.X
    y0 = t * (loc_b.Y - loc_a.Y) + loc_a.Y
    z0 = t * (loc_b.Z - loc_a.Z) + loc_a.Z
    d = np.sqrt(np.square(x-x0)+np.square(y-y0)+np.square(z-z0))
    return t,d

#建立半球胞体
def distance_(data_order):
    distance_dict = {}
    for i in range(len(data_order)):
        distance_list = []
        x = data_order['X'].values[i]
        y = data_order['Y'].values[i]
        z = data_order['Z'].values[i]
        for j in range(i+1,len(data_order)):
            x1 = data_order['X'].values[j]
            y1 = data_order['Y'].values[j]
            z1 = data_order['Z'].values[j]
            distance = np.sqrt(np.square(x-x1)+np.square(y1-y)+np.square(z1-z))
            distance_list.append((data_order['num'].values[j],data_order['type'].values[j],distance))
            distance_list.sort(key=lambda tup:tup[2])
        distance_list = [d for d in distance_list if d[2]<=20000]
        distance_dict[data_order['num'].values[i]] = distance_list

    return distance_dict


