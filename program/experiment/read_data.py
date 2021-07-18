import csv
import numpy as np
def readcsv(csvpath):
    with open(csvpath,'r',encoding='UTF-8') as f:
        reader=csv.reader(f)
        result=list(reader)
        return(result)

def readinfo(infopath):
    with open(infopath,'r',encoding='UTF-8') as f:
        reader=csv.reader(f)
        info=list(reader)
        return info
def merge(attributes):
    datas=[]
    n=len(attributes[0].data)
    for i in range(n):
        data=[]
        for attri in attributes:
            data.append(attri.data[i])
        datas.append(data)
    return datas
