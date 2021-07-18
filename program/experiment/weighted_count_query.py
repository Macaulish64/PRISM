from attribute_module import attribute
import math

def convert_data(datas, measure,k):
    n=len(datas)
    convert_table=[]
    count_len=[]
    for i in range(n):
        m=int(round(measure.data[i]/k))
        for j in range(m):
            convert_table.append(datas[i])
            count_len.append(m)
    return convert_table, count_len
