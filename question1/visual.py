import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

#可视化所有校正点的空间分布和AB线段
data = pd.read_csv('数据集1.csv',encoding='gbk')
data.columns = ['num','X','Y','Z','type','q3_type']
loc_a = data.loc[0, ['X', 'Y', 'Z']]
loc_b = data.iloc[-1, 1:4]
data['type'].values[0] = 0
data['type'].values[-1] = 0
x1 = data[data['type'].apply(int).values == 0]['X'].values
y1 = data[data['type'].apply(int).values == 0]['Y'].values
z1 = data[data['type'].apply(int).values == 0]['Z'].values
x2 = data[data['type'].apply(int).values == 1]['X'].values
y2 = data[data['type'] .apply(int).values== 1]['Y'].values
z2 = data[data['type'].apply(int).values == 1]['Z'].values
fig = plt.figure()
ax = Axes3D(fig)
ax.scatter(x1, y1, z1, c='r', label='0')
ax.scatter(x2,y2,z2, c='y', label='1')
ax.plot([loc_a.X, loc_b.X], [loc_a.Y, loc_b.Y], [loc_a.Z, loc_b.Z], c='r')
plt.show()