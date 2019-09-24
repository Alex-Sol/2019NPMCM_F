import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from data_preprocessing import *
from genetic_optimize import *

if __name__ == '__main__':
    data = pd.read_csv('数据集1.csv', encoding='gbk')    #读取数据
    data.columns = ['num', 'X', 'Y', 'Z', 'type', 'type_q3']
    loc_a = data.loc[0, ['X', 'Y', 'Z']]           #提取起始点、终点坐标
    loc_b = data.iloc[-1, 1:4]
    x_cal = data.X.values               #提取所有校正点坐标
    y_cal = data.Y.values
    z_cal = data.Z.values
    loc_cal = [x_cal, y_cal, z_cal]
    order_pro, distance = Coordinate_projection(loc_a, loc_b, x_cal, y_cal, z_cal)        #投影后的顺序、校正点与AB线的距离
    data_order = data[['num', 'X', 'Y', 'Z', 'type']]
    data_order['order_pro'] = order_pro           #大小代表离A点的远近
    data_order['distance'] = distance            #距离
    data_order = data_order.sort_values(by="order_pro", ascending=True)               #对在AB上的投影进行排序
    data_order = data_order[(data_order['order_pro'] <= 1) & (data_order['order_pro'] >= 0)].reset_index()   #剔除投影在AB线段以外的点
    d_max = max(data_order['distance'].values)
    data_order = data_order[data_order['distance'].values <= d_max / 3]      #取出距离较近的点作为待选
    del data_order['index']
    distance_dict = distance_(data_order)           #建立半球胞体字典
    pd.DataFrame(dict([(k, pd.Series(v)) for k, v in distance_dict.items()])).to_csv('distance_dict.csv')


    num_index = {}
    for i in range(len(data_order)-2):      #建立染色体中对应序号和对应校正点的编号
        num_index[data_order['num'].values[i+1]] = i+1

    distance_dict = pd.read_csv('distance_dict.csv').to_dict(orient='series')
    x_cal = data_order.X.values[1:-1]     #待选点的坐标
    y_cal = data_order.Y.values[1:-1]
    z_cal = data_order.Z.values[1:-1]
    type_cal = data_order.type.values[1:-1]        #待选点的类型
    loc_cal = [x_cal, y_cal, z_cal]
    primary_population_size = 100         #初代种群规模
    chromosome_length = len(loc_cal[0])          #个体染色体长度
    population_size = 100           #种群规模
    population = species_origin(num_index, distance_dict, primary_population_size, chromosome_length)      #建立初代种群
    pc = 0.25      #交叉概率
    pm = 0.02         #变异概率
    best_length = []         #每一代的最优个体的航迹长度
    best_num_node = []        #每一代的最优个体的校正点数目
    best_loc_list = []         #每一代最优个体的校正点坐标序列
    best_num_list = []         #每一代最优个体的校正点的编号序列
    X = []
    bestindividual_list = []     #每一代的最优个体染色体
    bestfitness_list = []       #每一代的最优个体的适应度
    for i in range(100):
        off_population = crossover(population, pc)    #交叉
        off_population = mutation(population,off_population,pm)   #变异
        population = mutation1(distance_dict, data_order, num_index, population, off_population, pm)
        fitness,df = function(loc_b,loc_a,loc_cal,type_cal,population,chromosome_length)   #计算适应度
        population,fitness = selection(population_size, population, fitness)    #选择
        [bestindividual, bestfitness] = best(population, fitness)     #在每一代中选择适应度最大的最优个体
        bestindividual_list.append(bestindividual)
        bestfitness_list.append(bestfitness)
        best_length.append(length_calculate(loc_b,loc_a,loc_cal,bestindividual))
        best_num_node.append(num_node_calculate(bestindividual))
        X.append(i)
        print('第',i,'轮迭代')

    best_loc, best_num, best_type_list = loc_calculate(loc_a, loc_cal, data_order, bestindividual)   #计算最优个体的坐标、校正点数目、类型
    best_loc_list.append(best_loc)
    error_level_list, error_vertical_list = best_error(loc_a, loc_b, loc_cal, type_cal, bestindividual)   #计算最优个体的每个校正点校正之前垂直误差，水平误差
    result = pd.DataFrame({'校正点编号': best_num,
                           '校正前垂直误差': error_vertical_list,
                           '校正前水平误差': error_level_list,
                           '校正点类型': best_type_list})
    result.to_csv('result1_1终稿.csv', encoding='gbk')

    fig = plt.figure()      #结果可视化
    ax = Axes3D(fig)
    data['type'].values[0] = 0
    data['type'].values[-1] = 0
    x1 = data[data['type'].apply(int).values == 0]['X'].values
    y1 = data[data['type'].apply(int).values == 0]['Y'].values
    z1 = data[data['type'].apply(int).values == 0]['Z'].values
    x2 = data[data['type'].apply(int).values == 1]['X'].values
    y2 = data[data['type'].apply(int).values == 1]['Y'].values
    z2 = data[data['type'].apply(int).values == 1]['Z'].values
    ax.scatter(x1, y1, z1, c='r', label='0')
    ax.scatter(x2, y2, z2, c='y', label='1')
    ax.plot([loc_a.X, loc_b.X], [loc_a.Y, loc_b.Y], [loc_a.Z, loc_b.Z], c='r')
    x = []
    y = []
    z = []
    for i in range(len(best_loc_list[0])):
        x.append(best_loc_list[0][i][0])
        y.append(best_loc_list[0][i][1])
        z.append(best_loc_list[0][i][2])
    x.append(loc_b[0])
    y.append(loc_b[1])
    z.append(loc_b[2])
    ax.plot(x, y, z, linewidth=1)
    plt.savefig('result1_1路径.jpg')
    plt.show()

    pd.DataFrame(columns=['best_num_node'], data=best_num_node).to_csv('best_num_node.csv')
    pd.DataFrame(columns=['best_length'], data=best_length).to_csv('best_length.csv')

    plt.plot(X, best_length)
    plt.savefig('best_length.jpg')
    plt.show()
    plt.plot(X, best_num_node)
    plt.savefig('best_num_node.jpg')
    plt.show()

