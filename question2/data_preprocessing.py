import pandas as pd
import numpy as np
import math

#计算点在投影上的排序位置、离AB线段的距离
def Coordinate_projection(loc_a,loc_b,x,y,z):
    t = ((loc_b.X-loc_a.X)*(x-loc_a.X)+(loc_b.Y-loc_a.Y)*(y-loc_a.Y)+(loc_b.Z-loc_a.Z)*(z-loc_a.Z)) \
            /(np.square(loc_b.X-loc_a.X)+np.square(loc_b.Y-loc_a.Y)+np.square(loc_b.Z-loc_a.Z))
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

#由三点坐标求经过三点的平面
def three_point_flat(loc_1,loc_2,loc_3):
    x1 = loc_1[0]
    y1 = loc_1[1]
    z1 = loc_1[2]
    x2 = loc_2[0]
    y2 = loc_2[1]
    z2 = loc_2[2]
    x3 = loc_3[0]
    y3 = loc_3[1]
    z3 = loc_3[2]
    A = (y2-y1)*(z3-z1)-(z2-z1)*(y3-y1)
    B = (z2-z1)*(x3-x1)-(x2-x1)*(z3-z1)
    C = (x2-x1)*(y3-y1)-(x3-x1)*(y2-y1)

    return A,B,C

#由当前运动方向和当前点、目标点坐标求两校正点之间的最短路径长度
def route_distance(v,loc_1,loc_2):
    vx = v[0]
    vy = v[1]
    vz = v[2]
    x1 = loc_1[0]
    y1 = loc_1[1]
    z1 = loc_1[2]
    x2 = loc_2[0]
    y2 = loc_2[1]
    z2 = loc_2[2]
    if loc_1[0]==0:
        d = np.sqrt(np.square(x2-x1)+np.square(y2-y1)+np.square(z2-z1))
        theta1 = 0
    else:
        x3 = x1 + vx
        y3 = y1 + vy
        z3 = z1 + vz
        #A,B,C = three_point_flat(loc_1,loc_2,[x3,y3,z3])
        #过三点平面方程：A(x-x1)+B(y-y1)+C(z-z1) = 0
        #与v垂直，过A点的平面：vx*(x-x1)+vy*(y-y1)+vz*(z-z1) = 0
        d_1_2_vertical = abs(vx*(x2-x1)+vy*(y2-y1)+vz*(z2-z1))/np.sqrt(np.square(vx)+np.square(vy)+np.square(vz))
        d_1_2 = np.sqrt(np.square(x2-x1)+np.square(y2-y1)+np.square(z2-z1))
        x_2 = np.sqrt(np.square(d_1_2)-np.square(d_1_2_vertical))
        y_2 = d_1_2_vertical
        x_1 = 0
        y_1 = 0
        x_O = 200
        y_O = 0
        r = 200
        d_BO = np.sqrt(np.square(x_2-x_O)+np.square(y_2-y_O))
        d2 = np.sqrt(np.square(d_BO)-np.square(r))
        theata2 = math.acos(r/d_BO)
        theata = math.pi-math.asin(y_2/d_BO)
        theta1 = theata-theata2
        d1 = theta1*r
        d = d1+d2


    return d,theta1

#由四元数法，已知当前运动方向和当前点、目标点坐标，求到达目标点时的运动方向
def direction(v,loc_1,loc_2):
    vx = v[0]
    vy = v[1]
    vz = v[2]
    x1 = loc_1[0]
    y1 = loc_1[1]
    z1 = loc_1[2]
    x2 = loc_2[0]
    y2 = loc_2[1]
    z2 = loc_2[2]
    if loc_1[0]==0:
        v = [x2-x1,y2-y1,z2-z1]
    else:
        x3 = x1 + vx
        y3 = y1 + vy
        z3 = z1 + vz
        r = 200
        A, B, C = three_point_flat(loc_1, loc_2, [x3, y3, z3])
        abs_v = np.sqrt(np.square(vx)+np.square(vy)+np.square(vz))
        d,theta1 = route_distance(v,loc_1,loc_2)
        d0 = 2*r*math.sin(theta1/2)
        v0 = [vx*d0/abs_v,vy*d0/abs_v,vz*d0/abs_v]
        xp = x1+v0[0]
        yp = y1+v0[1]
        zp = z1+v0[2]
        u = [A,B,C]
        theta = theta1/2
        q0 = math.cos(theta/2)
        q1 = A*math.sin(theta/2)
        q2 = B*math.sin(theta/2)
        q3 = C*math.sin(theta/2)
        xq = (np.square(q0)+np.square(q1)-np.square(q2)-np.square(q3))*xp+(2*q1*q2-2*q0*q3)*yp+(2*q0*q2+2*q1*q3)*zp
        yq = (2*q0*q3+2*q1*q2)*xp+(np.square(q0)-np.square(q1)+np.square(q2)-np.square(q3))*yp+(2*q2*q3-2*q0*q1)*zp
        zq = (2*q1*q3-2*q0*q2)*xp+(2*q0*q1+2*q2*q3)*yp+(np.square(q0)-np.square(q1)-np.square(q2)+np.square(q3))*zp
        v = [x2-xq,y2-yq,z2-zq]

    return v

#另外一种求运动方向的方法，假设路径是以圆弧运动
def direction0(v,loc_1,loc_2):
    vx = v[0]
    vy = v[1]
    vz = v[2]
    x1 = loc_1[0]
    y1 = loc_1[1]
    z1 = loc_1[2]
    x2 = loc_2[0]
    y2 = loc_2[1]
    z2 = loc_2[2]
    ux = x2-x1
    uy = y2-y1
    uz = z2-z1

    if loc_1[0]==0:
        v = [x2-x1,y2-y1,z2-z1]
    else:
        cos_theta = (vx*ux+vy*uy+vz*uz)/(np.sqrt(np.square(vx)+np.square(vy)+np.square(vz)) \
                                         *np.sqrt(np.square(ux)+np.square(uy)+np.square(uz)))
        l_u = np.sqrt(np.square(vx)+np.square(vy)+np.square(vz))*cos_theta*2
        l_u0 = np.sqrt(np.square(ux)+np.square(uy)+np.square(uz))
        xp = ux*l_u/l_u0+x1
        yp = uy * l_u / l_u0 + y1
        zp = uz * l_u / l_u0 + z1
        x0 = vx+x1
        y0 = vy+y1
        z0 = vz+z1
        v = [xp-x0,yp-y0,zp-z0]

    return v

#由当前运动方向和当前点、目标点坐标判断目标点是否在环形管道外，以判定目标点是否符合最小转弯约束要求
def judge_feasibility(v,loc_1,loc_2):
    if loc_1[0]!=0:
        vx = v[0]
        vy = v[1]
        vz = v[2]
        x1 = loc_1[0]
        y1 = loc_1[1]
        z1 = loc_1[2]
        x2 = loc_2[0]
        y2 = loc_2[1]
        z2 = loc_2[2]
        d_1_2_vertical = abs(vx * (x2 - x1) + vy * (y2 - y1) + vz * (z2 - z1)) / np.sqrt(np.square(vx) + np.square(vy) + np.square(vz))
        d_1_2 = np.sqrt(np.square(x2 - x1) + np.square(y2 - y1) + np.square(z2 - z1))
        x_2 = np.sqrt(np.square(d_1_2) - np.square(d_1_2_vertical))
        y_2 = d_1_2_vertical
        x_O = 200
        y_O = 0
        r = 200
        d_BO = np.sqrt(np.square(x_2 - x_O) + np.square(y_2 - y_O))

        if d_BO>=r:
            F = True
        else:
            F = False
    else:
        F = True

    return F



