# -*-coding:utf-8 -*-
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

#初始化生成chromosome_length大小的population_size个个体的二进制基因型种群
def species_origin(num_index,distance_dict,population_size,chromosome_length):
    population = []
    alpha1 = 20
    alpha2 = 10
    beta1 = 15
    beta2 = 20
    theta = 20
    delta = 0.001
    i = 0
    while i<population_size:
        node_list = []
        node_list.append((0, 'A', 0))
        node_candidate = []
        error_level = 0
        error_vertical = 0

        for n in distance_dict[str(node_list[0][0])]:
            if type(n) is not float:
                n = eval(n)
                if (n[1] == '0')&(n[2]<=beta1/delta):
                    node_candidate.append(n)
                elif(n[1] == '1')&(n[2]<=alpha2/delta):
                    node_candidate.append(n)
        r = random.randint(0, len(node_candidate) - 1)
        node_list.append(node_candidate[r])
        temp_node = node_list[-1]
        error_level += temp_node[2]*delta
        error_vertical += temp_node[2]*delta
        if temp_node[1]=='0':
            error_level = 0
        elif temp_node[1]=='1':
            error_vertical=0
        while temp_node[1]!='B':
            node_candidate = []
            for n in distance_dict[str(temp_node[0])]:
                if type(n) is not float:
                    n = eval(n)
                    if(n[1]=='0')&(n[2]<=min((beta1-error_vertical)/delta,(beta2-error_level)/delta)):
                        node_candidate.append(n)
                    elif(n[1]=='1')&(n[2]<=min((alpha1-error_vertical)/delta,(alpha2-error_level)/delta)):
                        node_candidate.append(n)
                    elif(n[1]=='B')&(n[2]<=min((theta-error_level)/delta,(theta-error_vertical)/delta)):
                        node_candidate.append(n)
            if len(node_candidate) == 0:
                break
            elif len(node_candidate) == 1:
                node_list.append(node_candidate[0])
            else:
                r = random.randint(0, len(node_candidate) - 1)
                node_list.append(node_candidate[r])
            temp_node = node_list[-1]
            error_level += temp_node[2] * delta
            error_vertical += temp_node[2] * delta
            if (temp_node[1] == '0'):
                if temp_node[3]==1:
                    error_level = 1
                else:
                    error_level = 0
            elif (temp_node[1] == '1') :
                if temp_node[3] == 1:
                    error_vertical = 1
                else:
                    error_vertical = 0
        if len(node_candidate) == 0:
            continue
        population.append([0 for _ in range(chromosome_length)])
        for j in range(len(node_list)-2):
            population[i][num_index[node_list[j+1][0]]-1] = 1
        i += 1

    return population

# 目标函数相当于环境 对染色体进行筛选，这里是总路径长度/当代种群里最长距离
def function(loc_b,loc_a,loc_cal,type_cal,population,chromosome_length,q3type_cal):
    sum_fitness = 0
    fitness = []
    alpha1 = 20
    alpha2 = 10
    beta1 = 15
    beta2 = 20
    theta = 20
    delta = 0.001
    length_list = []
    num_node_list = []
    length0_list = []
    P_list = []       #每个个体成功走到终点的概率
    num_node0_list = []
    zx = []
    zx1 = []
    zx2 = []
    for i in range(len(population)):
        P = 0        #能飞到B点的概率
        issue_point_num = 0
        for q in range(chromosome_length):
            if (population[i][q] == 1) & (q3type_cal[q]==1):
                issue_point_num += 1   #计算问题点的数目
        k = [0 for _ in range(issue_point_num)]
        for h in range(np.power(2,issue_point_num)):     #遍历所有的组合可能性
            pk = 1
            h = str(bin(h))[2:]          #将十进制转二进制以代表组合状态
            t = 0
            for ki in range(len(h)):
                k[ki] = int(h[ki])
                pk  *= 0.2*k[ki]+0.8*(1-k[ki])          #此种组合出现的概率
            length = 0
            num_node = 0
            error_level = 0
            error_vertical = 0
            temp_loc = loc_a.values
            Fi = True     # 为True符合条件,为False不符合条件
            for j in range(chromosome_length):
                if population[i][j] == 1:
                        distance = np.sqrt(
                            np.square(loc_cal[0][j] - temp_loc[0]) + np.square(loc_cal[1][j] - temp_loc[1]) \
                            + np.square(loc_cal[2][j] - temp_loc[2]))
                        length += distance
                        temp_loc = [loc_cal[0][j], loc_cal[1][j], loc_cal[2][j]]
                        num_node += 1
                        error_level += distance * delta
                        error_vertical += distance * delta
                        error_level_adjusted = min(error_level,5)
                        error_vertical_adjusted = min(error_vertical,5)
                        if (type_cal[j] == '0') & (error_vertical <= beta1) & (error_level <= beta2):
                            if q3type_cal[j]==1:     #是问题点
                                if k[t]==1:
                                    error_level = error_level_adjusted
                                else:
                                    error_level = 0
                                t += 1
                            else:
                                error_level = 0
                        elif (type_cal[j] == '1') & (error_vertical <= alpha1) & (error_level <= alpha2):
                            if q3type_cal[j] == 1:
                                if k[t] == 1:
                                    error_vertical = error_vertical_adjusted
                                else:
                                    error_vertical = 0
                                t += 1
                            else:
                                error_vertical = 0
                        else:
                            Fi = False

            distance = np.sqrt(np.square(loc_b[0] - temp_loc[0]) + np.square(loc_b[1] - temp_loc[1]) \
                               + np.square(loc_b[2] - temp_loc[2]))
            length += distance
            error_level += distance * delta
            error_vertical += distance * delta
            if (error_level > theta) | (error_vertical > theta):
                Fi = False
            if Fi == True:
                print('已找到')
            if Fi == True:
                P += pk   #若此种组合能成功到达B点，则把其出现的概率加入到此航迹成功到达B点的概率P中

        length = 0
        num_node = 0
        error_level = 0
        error_vertical = 0
        temp_loc = loc_a.values
        F = True #为True符合条件,为False不符合条件
        for j in range(chromosome_length):
            if population[i][j] == 1:
                distance = np.sqrt(np.square(loc_cal[0][j]-temp_loc[0])+np.square(loc_cal[1][j]-temp_loc[1]) \
                                  +np.square(loc_cal[2][j]-temp_loc[2]))
                length += distance
                temp_loc = [loc_cal[0][j],loc_cal[1][j],loc_cal[2][j]]
                num_node += 1
                error_level += distance*delta
                error_vertical += distance*delta
                if (type_cal[j] == '0') & (error_vertical<=beta1) & (error_level<=beta2):
                        error_level = 0
                elif (type_cal[j] == '1') & (error_vertical<=alpha1) & (error_level<=alpha2):
                        error_vertical = 0
                else:
                    F = False

        distance = np.sqrt(np.square(loc_b[0]-temp_loc[0])+np.square(loc_b[1]-temp_loc[1]) \
                                  +np.square(loc_b[2]-temp_loc[2]))
        length += distance
        error_level += distance * delta
        error_vertical += distance * delta
        length0_list.append(length)
        num_node0_list.append(num_node)
        if (error_level>theta) | (error_vertical>theta):
            F = False
        if (F == False) | (P<0.5):    #根据航迹的到达B点的概率大小来决策是否惩罚
            length = random.uniform(100,500)*length
            num_node = random.uniform(100,500)*num_node
            if num_node <= 6:
                num_node = np.power(100,6-num_node)*num_node
        length_list.append(length)
        num_node_list.append(num_node)
        P_list.append(P)
    max_length = max(length_list)
    min_length = min(num_node_list)
    max_num_node = max(num_node_list)
    min_num_node = min(num_node_list)
    for i in range(len(population)):
        fitness.append((length_list[i]-min_length)/(max_length-min_length) \
                       +(num_node_list[i]-min_num_node)/(max_num_node-min_num_node+1e-5))
        zx.append(fitness[i])
        fitness[i] = 1/fitness[i]
        zx1.append(fitness[i])
        sum_fitness += fitness[i]
    for i in range(len(fitness)):
        fitness[i] = fitness[i]/sum_fitness
        zx2.append(fitness[i])

    df = pd.DataFrame({'length0':length0_list,
                       'num_node0':num_node0_list,
                       'f1_length':length_list,
                       'f2_num_node':num_node_list,
                       'zx':zx,
                       '1/zx':zx1,
                       '1/zx/sum':zx2,
                       'P':P_list})

    return fitness,df

def cumsum(fitness):
    for i in range(len(fitness) - 2, -1, -1):
        # range(start,stop,[step])
        # 倒计数
        total = 0
        j = 0
        while (j <= i):
            total += fitness[j]
            j += 1

        fitness[i] = total
        fitness[len(fitness) - 1] = 1

    return fitness

# 选择种群中个体适应度最大的个体
def selection(pop_len, population,fitness):
    new_fitness = fitness.copy()
    # 将所有个体的适应度概率化,类似于softmax
    new_fitness = cumsum(new_fitness)
    # 将所有个体的适应度划分成区间
    fitness1 = [] #选择之后的种群对应的适应度
    ms = []
    # 存活的种群
    # 根据随机数确定哪几个能存活

    for i in range(pop_len):
        ms.append(random.random())
    # 产生种群个数的随机值
    ms.sort()
    # 存活的种群排序
    fitin = 0
    newin = 0
    #排序方式
    pop_fit = pd.DataFrame({'pop':population,'fitness':fitness})
    pop_fit = pop_fit.sort_values(by="fitness", ascending=False)
    new_pop = pop_fit['pop'].values[:pop_len].tolist()
    fitness1 = pop_fit['fitness'].values[:pop_len].tolist()
    # 轮盘赌方式
    '''new_pop = population[:pop_len]
    while newin<pop_len:
        if ms[newin]<new_fitness[fitin]:
            new_pop[newin] = population[fitin]
            fitness1.append(fitness[fitin])
            newin += 1
        else:
            fitin += 1'''

    return new_pop,fitness1

#交叉
def crossover(pop, pc):
    pop_len = len(pop)
    off_pop = []
    for i in range(pop_len - 1):
        if(random.random() < pc):
            cpoint = random.randint(0,len(pop[0]))
            temp1 = []
            temp2 = []
            temp1.extend(pop[i][0:cpoint])
            temp1.extend(pop[i+1][cpoint:len(pop[i])])
            temp2.extend(pop[i+1][0:cpoint])
            temp2.extend(pop[i][cpoint:len(pop[i])])
            off_pop.append(temp1)
            off_pop.append(temp2)

    return off_pop


#变异
def mutation(pop,off_pop,pm):
    # pm是概率阈值
    px = len(pop)
    # 求出种群中所有种群/个体的个数
    py = len(pop[0])
    # 染色体/个体基因的个数
    for i in range(px):
        if (random.random() < pm):
            temp = pop[i].copy()
            mpoint = random.randint(0, py - 1)
            if (temp[mpoint] == 1):
                # 将mpoint个基因进行单点随机变异，变为0或者1
                temp[mpoint] = 0
            else:
                temp[mpoint] = 1
            off_pop.append(temp)
    pop.extend(off_pop)

    return off_pop

#多重变异
def mutation1(distance_dict,data_order,num_index,pop,off_pop,pm):
    # pm是概率阈值
    px = len(pop)
    # 求出种群中所有种群/个体的个数
    py = len(pop[0])
    # 染色体/个体基因的个数
    for i in range(px):
        if (random.random() < pm):
            node_candidate = []
            temp = pop[i].copy()
            mpoint = random.randint(0, py - 1)
            node_num = [key for key,value in num_index.items() if value==mpoint+1][0]
            type_node = data_order[data_order['num']==node_num]['type'].values[0]
            for n in distance_dict[str(node_num)]:
                if type(n) is not float:
                    n = eval(n)
                    if (n[1] == type_node) & (n[2] <= 15 / 0.001):
                        node_candidate.append(n)

            if len(node_candidate) == 0:
                pass
            elif len(node_candidate) == 1:
                temp[num_index[node_candidate[0][0]] - 1] = temp[mpoint]
            else:
                r = random.randint(0, len(node_candidate) - 1)
                temp[num_index[node_candidate[r][0]] - 1] = temp[mpoint]
            if (temp[mpoint] == 1):
                # 将mpoint个基因进行单点随机变异，变为0或者1
                temp[mpoint] = 0
            else:
                temp[mpoint] = 1
            off_pop.append(temp)

    pop.extend(off_pop)

    return pop

# 寻找最好的适应度和个体
def best(population, fitness):
    px = len(population)
    bestindividual = population[0]
    bestfitness = fitness[0]

    for i in range(1, px):
        # 循环找出最大的适应度，适应度最大的也就是最好的个体
        if (fitness[i] > bestfitness) & (population[i].count(1)!=0):
            bestfitness = fitness[i]
            bestindividual = population[i]

    return [bestindividual, bestfitness]

def length_calculate(loc_b,loc_a,loc_cal,individual):
    length = 0
    temp_loc = loc_a.values
    for i in range(len(individual)):
        if individual[i] == 1:
            length += np.sqrt(np.square(loc_cal[0][i] - temp_loc[0]) + np.square(loc_cal[1][i] - temp_loc[1]) \
                              + np.square(loc_cal[2][i] - temp_loc[2]))
            temp_loc = [loc_cal[0][i], loc_cal[1][i], loc_cal[2][i]]
    length += np.sqrt(np.square(loc_b[0] - temp_loc[0]) + np.square(loc_b[1] - temp_loc[1]) \
                              + np.square(loc_b[2] - temp_loc[2]))

    return length

def loc_calculate(loc_a,loc_cal,data_order,individual):
    best_loc = []
    best_loc.append((loc_a.values[0],loc_a.values[1],loc_a.values[2]))
    best_num = []
    best_num.append(0)
    type_list = ['出发点A']

    for i in range(len(individual)):
        if individual[i] == 1:
            best_loc.append((loc_cal[0][i],loc_cal[1][i],loc_cal[2][i]))
            best_num.append(data_order['num'].values[i+1])
            type_list.append(data_order['type'].values[i + 1])

    best_num.append(data_order['num'].values[-1])
    type_list.append('终点B')

    return best_loc,best_num,type_list


def num_node_calculate(individual):
    num_node = individual.count(1)

    return num_node

def best_error(loc_a,loc_b,loc_cal,type_cal,bestindividual,q3type_cal):
    error_level = 0
    error_vertical = 0
    temp_loc = loc_a.values
    alpha1 = 20
    alpha2 = 10
    beta1 = 15
    beta2 = 20
    theta = 20
    delta = 0.001
    error_level_list = [error_level]
    error_vertical_list = [error_vertical]
    for i in range(len(bestindividual)):
        if bestindividual[i] == 1:
            distance = np.sqrt(np.square(loc_cal[0][i]-temp_loc[0])+np.square(loc_cal[1][i]-temp_loc[1]) \
                                  +np.square(loc_cal[2][i]-temp_loc[2]))
            error_level += distance * delta
            error_vertical += distance * delta
            error_level_list.append(error_level)
            error_vertical_list.append(error_vertical)
            # print(error_level)
            # print(error_vertical)
            if (type_cal[i] == '0') & (error_vertical <= beta1) & (error_level <= beta2):
                error_level = 0
            elif (type_cal[i] == '1') & (error_vertical <= alpha1) & (error_level <= alpha2):
                error_vertical = 0

            temp_loc = [loc_cal[0][i],loc_cal[1][i],loc_cal[2][i]]
    distance = np.sqrt(np.square(loc_b[0] - temp_loc[0]) + np.square(loc_b[1] - temp_loc[1]) \
                              + np.square(loc_b[2] - temp_loc[2]))
    error_level += distance * delta
    error_vertical += distance * delta
    error_level_list.append(error_level)
    error_vertical_list.append(error_vertical)
    print('error_level_list:',error_level_list)
    print('error_level_list',error_vertical_list)

    return error_level_list, error_vertical_list

