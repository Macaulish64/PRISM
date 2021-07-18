#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import csv
from matplotlib.font_manager import FontProperties
import numpy as np
from matplotlib.ticker import FuncFormatter

# plt.style.use('seaborn-darkgrid')

fig = plt.gcf()
# fig.set_size_inches(7.0/3,7.0/3) #dpi = 300, output = 700*700 pixels
fig.set_size_inches(9, 6)  # dpi = 300, output = 700*700 pixels
path = 'C:/Users\FlyFF\Desktop\program\data/'

filename = 'test'

out_png_path = path + filename + '.pdf'

font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=24)

methods_list = []
g5 = []
g10 = []
g15 = []
g20 = []
g25 = []
g30 = []
prism = []
# 打开example.txt 并且以读的方式打开
with open('C:/Users\FlyFF\Desktop\program\data/example.txt', 'r') as file:
	# 用csv去读文件 有关csv文件的格式请自行科谱
	# csv去读取文件并不只是读取以.csv结尾的文件，它只要满足是分隔数据格式就可以了，以逗号进行分隔的数据
	plots = csv.reader(file, delimiter='\t')
	for row in plots:
		methods_list.append(row[0])
		g5.append(float(row[1]))
		g10.append(float(row[2]))
		g15.append(float(row[3]))
		g20.append(float(row[4]))
		g25.append(float(row[5]))
		g30.append(float(row[6]))
		prism.append(float(row[7]))

total_width, n = 0.8, 7  # 有多少个类型，只需更改n即可，比如这里我们对比了四个，那么就把n设成4
width = total_width / n
x =list(range(len(methods_list)))

plt.bar(x, g5, width=width, color='darkcyan',tick_label = methods_list, label='g=5')
for i in range(len(x)):x[i] = x[i] + width
plt.bar(x, g10, width=width, color='olive',tick_label = methods_list, label='g=10')
for i in range(len(x)):x[i] = x[i] + width
plt.bar(x, g15, width=width, color='darkslateblue',tick_label = methods_list, label='g=15')
for i in range(len(x)):x[i] = x[i] + width
plt.bar(x, g20, width=width, color='seagreen',tick_label = methods_list, label='g=20')
for i in range(len(x)):x[i] = x[i] + width
plt.bar(x, g25, width=width, color='silver',tick_label = methods_list, label='g=25')
for i in range(len(x)):x[i] = x[i] + width
plt.bar(x, g30, width=width, color='grey',tick_label = methods_list, label='g=30')
for i in range(len(x)):x[i] = x[i] + width
plt.bar(x, prism, width=width, color='red',tick_label = methods_list, label='PRISM')

# plt.plot(x, g5, marker='+', markersize=15,lw=3,color='blue', label='g=5')
# plt.plot(x, g10, marker='*', markersize=15,lw=3,color='cyan', label='g=10')
# plt.plot(x, g15, marker='.', markersize=15,lw=3,color='coral', label='g=15')
# plt.plot(x, g20, marker='^', markersize=15,lw=3,color='orange', label='g=20')
# plt.plot(x, g25, marker='v', markersize=15,lw=3,color='seagreen', label='g=25')
# plt.plot(x, g30, marker='x', markersize=15,lw=3,color='greenyellow', label='g=30')
# plt.plot(x, prism, marker='d', markersize=15,lw=3,color='red', label='PRISM')

# def formatnum(x, pos):
#     return '$%.1f$x$10^{4}$' % (x/10000)
# formatter = FuncFormatter(formatnum)
# plt.gca().xaxis.set_major_formatter(formatter)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=20)

font2 = {'family': 'Times New Roman',
		 'weight': 'normal',
		 'size': 28, }
plt.xlabel('vol(q)', font2)
# plt.ylabel('MAE')

plt.yscale("log")
# plt.title(u'测试从文件加载数据', FontProperties=font)
# plt.figlegend(*fig.gca().get_legend_handles_labels(), ncol= 8)
# fig.savefiglegend('legend.png', format='png', transparent=True, dpi=300, bbox_inches = 'tight',pad_inches = 0)
# legend=plt.legend(loc='center',ncol=8,bbox_to_anchor=(0.5, 1))
# fig.savefig('legend.png', dpi="figure", bbox_inches='tight',bbox_to_anchor=(1.05, 1))
# legend=plt.legend()
fig.savefig(out_png_path, format='pdf', dpi=300, bbox_inches='tight', pad_inches=0)

plt.show()
# plt.close()
#
plt.figure()
plt.figlegend(*fig.gca().get_legend_handles_labels(),frameon=False, ncol=4)
plt.savefig(path + 'legend_g' + ".png", format='png',dpi=300, bbox_inches='tight')
plt.show()
plt.close()