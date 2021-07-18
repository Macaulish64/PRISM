import numpy as np
import random
import math
import pylab
import itertools
import matplotlib.pyplot as plt
def dictionairy():
    # 声明字典
    key_value = {}

    # 初始化
    key_value[2] = 56
    key_value[1] = 2
    key_value[5] = 12
    key_value[4] = 24
    key_value[6] = 18
    key_value[3] = 323
    print(key_value)
    print("按值(value)排序:")
    print(sorted(key_value.items(), key=lambda kv: (kv[1], kv[0])))
def export_legend(legend, filename="legend.png", expand=[-5, -5, 5, 5]):
    fig = legend.figure
    fig.canvas.draw()

    fig.savefig(filename, dpi="figure", bbox_inches='tight')
class cartesian(object):
    def __init__(self):
        self._data_list=[]

    def add_data(self,data=[]): #添加生成笛卡尔积的数据列表
        self._data_list.append(data)

    def build(self): #计算笛卡尔积
        print(self._data_list)
        for item in itertools.product(*self._data_list):
            print(item)
def main():
    df = pd.DataFrame(index=['A', 'B', 'C', 'D'], columns=['Values'])
    df['Values'] = [0.45, 0.28, 0.21, 0.3]

    fig = plt.figure(figsize=(8, 8))
    figlegend = plt.figure(figsize=(3, 2))

    # fig.subplot adds subplot to fig instead of to the 'current figure' like plt.subplot
    ax1 = fig.subplot(121, aspect='equal')
    df['Values'].dropna().plot(kind='pie', autopct='%1.0f%%', startangle=220, labels=None,
                               colors=['#002c4b', '#392e2c', '#92847a', '#ccc2bb', '#6b879d'])

    patches, labels = ax1.get_legend_handles_labels()
    # Get rid of the legend on the first plot, so it is only drawn on the separate figure
    ax1.get_legend().remove()
    figlegend.legend(patches, labels=df.index)

    fig.savefig('image.png')
    figlegend.savefig('legend.png')
    plt.close(fig)
    plt.close(figlegend)
if __name__ == "__main__":
    main()