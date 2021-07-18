from attribute_module import attribute
import read_data
import copy
def convert_data(datas, measure):
    data_dict={}
    n=len(datas)
    for i in range(n):
        m=int(measure.data[i])
        if m in data_dict:
            data_dict[m].append(datas[i])
        else:
            data_dict[m]=[]
            data_dict[m].append(datas[i])
    return data_dict

def spilt_table(table2, measure):
    table=copy.deepcopy(table2)
    table_dict={}
    title=table[0]
    del table[0]
    sub_table=[]
    sub_table.append(title)
    n = len(table)
    for i in range(n):
        m = int(measure.data[i])
        if m in table_dict:
            table_dict[m].append(table[i])
        else:
            table_dict[m] = []
            table_dict[m].append(title)
            table_dict[m].append(table[i])
    #print(table_dict)
    return table_dict

if __name__ == "__main__":
    path = "E:/test.csv"
    infopath = "E:\info.csv"
    result = read_data.readcsv(path)
    info = read_data.readinfo(infopath)
    measure='V'
    #print(result)
    mea = attribute(measure, result, info,0)
    spilt_table(result,mea)