# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import csv
from matplotlib.font_manager import FontProperties
import numpy as np
from matplotlib.ticker import FuncFormatter

plt.style.use('seaborn-darkgrid')

fig = plt.gcf()
fig.set_size_inches(8,4.5) #dpi = 300, output = 700*700 pixels
path='C:/Users\FlyFF\Desktop\program\data/'

filename='laplace_4_n'

out_png_path=path+filename+'.pdf'
font = FontProperties(fname=r"C:\Windows\Fonts\simhei.ttf", size=24)

x = []
hio = []
calm = []
hdg = []
min = []
max = []
prism_olh= []
prism_cube=[]
prism = []
# 打开example.txt 并且以读的方式打开
with open('C:/Users\FlyFF\Desktop\program\data/example.txt', 'r') as file:
    # 用csv去读文件 有关csv文件的格式请自行科谱
    # csv去读取文件并不只是读取以.csv结尾的文件，它只要满足是分隔数据格式就可以了，以逗号进行分隔的数据
    plots = csv.reader(file, delimiter='\t')
    for row in plots:
        x.append(row[0])
        hio.append(float(row[1]))
        calm.append(float(row[2]))
        hdg.append(float(row[3]))
        min.append(float(row[4]))
        max.append(float(row[5]))
        prism_olh.append(float(row[6]))
        prism_cube.append(float(row[7]))
        prism.append(float(row[8]))

plt.plot(x, hio, marker='+', markersize=15,lw=3,color='skyblue', label='HIO')
plt.plot(x, calm, marker='*', markersize=15,lw=3,color='blue', label='CALM')
plt.plot(x, hdg, marker='.', markersize=15,lw=3,color='cyan', label='HDG')
plt.plot(x, min, marker='^', markersize=15,lw=3,color='coral', label='MIN')
plt.plot(x, max, marker='v', markersize=15,lw=3,color='orange', label='MAX')
plt.plot(x, prism_olh, marker='x', markersize=15,lw=3,color='seagreen', label='NON-RRR')
plt.plot(x, prism_cube, marker='X', markersize=15,lw=3,color='greenyellow', label='NON-GPS')
plt.plot(x, prism, marker='d', markersize=15,lw=3,color='red', label='PRISM')

# def formatnum(x, pos):
#     return '$%.1f$x$10^{4}$' % (x/10000)
# formatter = FuncFormatter(formatnum)
# plt.gca().xaxis.set_major_formatter(formatter)
plt.tick_params(axis='both', which='major', labelsize=20)
plt.tick_params(axis='both', which='minor', labelsize=20)

font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 28,
}

plt.xlabel('$n$',font2)
# plt.ylabel('MAE')

plt.yscale("log")
# plt.title(u'测试从文件加载数据', FontProperties=font)
# plt.figlegend(*fig.gca().get_legend_handles_labels(), ncol= 8)
# fig.savefiglegend('legend.png', format='png', transparent=True, dpi=300, bbox_inches = 'tight',pad_inches = 0)
# legend=plt.legend(loc='center',ncol=8,bbox_to_anchor=(0.5, 1))
# fig.savefig('legend.png', dpi="figure", bbox_inches='tight',bbox_to_anchor=(1.05, 1))

fig.savefig(out_png_path, format='pdf',  dpi=300, bbox_inches = 'tight',pad_inches = 0)
# plt.close()
#
# plt.figure()
# plt.figlegend(*fig.gca().get_legend_handles_labels(),frameon=False, ncol=8)
# plt.savefig(path + 'legend' + ".pdf", format='pdf',dpi=300, bbox_inches='tight')
plt.show()
# plt.close()