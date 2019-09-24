import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from data_preprocessing import *
from genetic_optimize import *

if __name__ == '__main__':
    data = pd.read_csv('数据集1.csv',encoding='gbk')
    data.columns = ['num','X','Y','Z','type','type_q3']
    loc_a = data.loc[0,['X','Y','Z']]
    loc_b = data.iloc[-1,1:4]
    x_cal = data.X.values
    y_cal = data.Y.values
    z_cal = data.Z.values
    loc_cal = [x_cal, y_cal, z_cal]
    order_pro, distance = Coordinate_projection(loc_a, loc_b, x_cal, y_cal, z_cal)
    data_order = data[['num', 'X', 'Y', 'Z', 'type']]
    data_order['order_pro'] = order_pro
    data_order['order_pro'] = order_pro
    data_order['d'] = distance
    coor = []
    for i in range(len(data)):
        coor.append([data['X'].values[i],data['Y'].values[i],data['Z'].values[i]])
    data['coor'] = coor

    data_order = data_order.sort_values(by="order_pro", ascending=True)
    data_order = data_order[(data_order['order_pro'] <= 1) & (data_order['order_pro'] >= 0)].reset_index()
    d_max = max(data_order['d'].values)
    data_order = data_order[data_order['d'].values <= d_max / 3]
    del data_order['index']

    distance_dict = distance_(data_order)
    pd.DataFrame(dict([(k, pd.Series(v)) for k, v in distance_dict.items()])).to_csv('distance_dict.csv')

    distance_dict = pd.read_csv('distance_dict.csv').to_dict(orient='series')

    num_index = {}
    for i in range(len(data_order) - 2):
        num_index[data_order['num'].values[i + 1]] = i + 1



    x_cal = data_order.X.values[1:-1]
    y_cal = data_order.Y.values[1:-1]
    z_cal = data_order.Z.values[1:-1]
    type_cal = data_order.type.values[1:-1]
    loc_cal = [x_cal, y_cal, z_cal]
    primary_population_size = 10
    chromosome_length = len(loc_cal[0])
    population_size = 10
    population = species_origin(data, loc_a, num_index, distance_dict, population_size, chromosome_length)
    pc = 0.25
    pm = 0.02
    best_length = []
    best_num_node = []
    best_loc_list = []
    best_num_list = []
    X = []
    bestindividual_list = []
    bestfitness_list = []
    for i in range(500):
        off_population = crossover(population, pc)
        off_population = mutation(population, off_population, pm)
        population = mutation1(distance_dict, data_order, num_index, population, off_population, pm)
        fitness, df = function(loc_b, loc_a, loc_cal, type_cal, population, chromosome_length)
        population, fitness = selection(population_size, population, fitness)
        [bestindividual, bestfitness] = best(population, fitness)
        best_length.append(length_calculate(loc_b, loc_a, loc_cal, bestindividual))
        best_num_node.append(num_node_calculate(bestindividual))
        X.append(i)
        print('第', i, '轮迭代')

    best_loc, best_num, best_type_list = loc_calculate(loc_a, loc_cal, data_order, bestindividual)
    best_loc_list.append(best_loc)
    error_level_list, error_vertical_list = best_error(loc_a, loc_b, loc_cal, type_cal, bestindividual)

    result = pd.DataFrame({'校正点编号': best_num,
                           '校正前垂直误差': error_vertical_list,
                           '校正前水平误差': error_level_list,
                           '校正点类型': best_type_list})
    result.to_csv('result2_1终稿.csv', encoding='gbk')

    fig = plt.figure()
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
    plt.savefig('result2_1路径终稿.jpg')
    plt.show()

    pd.DataFrame(columns=['best_num_node'], data=best_num_node).to_csv('best_num_node.csv')
    pd.DataFrame(columns=['best_length'], data=best_length).to_csv('best_length.csv')

    plt.plot(X, best_length)
    plt.savefig('best_length.jpg')
    plt.show()
    plt.plot(X, best_num_node)
    plt.savefig('best_num_node.jpg')
    plt.show()

