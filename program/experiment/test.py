import read_data
import control_group
import prefix_sum
import utils
import numpy as np
import itertools
from attribute_module import attribute
import weighted_count_query
import weighted_frequency_query
from estimation import Synthesizer
import query_module
import copy
def my_solution(attributes,measure,epsilon, noisy):
    datas=read_data.merge(attributes)
    #print("the attributes to be collected:")
    #print(datas)
    mea=attribute(measure,result2,info,0)
    table,len_list =weighted_count_query.convert_data(datas,mea,1)
    #print(table)
    pfs=prefix_sum.prefixsum(attributes,measure,epsilon)
    pfs.collection(table, len_list, noisy)
    #print(pfs.result)
    return pfs.result
def my_solution2(attributes,measure,epsilon, noisy):
    datas=read_data.merge(attributes)
    #print("the attributes to be collected:")
    #print(datas)
    mea=attribute(measure,result,info,0)
    table=datas
    len_list=np.ones(len(table),dtype=int)
    #print(table)
    pfs=prefix_sum.prefixsum(attributes,measure,epsilon)
    pfs.collection(table, len_list, noisy)
    #print(pfs.result)
    return pfs.result
def my_solution3(attributes,measure,epsilon, noisy):
    for attri in attributes:
        print('domain is ::::::::::::::',attri.name)
    datas=read_data.merge(attributes)
    #print("the attributes to be collected:")
    #print(datas)
    mea=attribute(measure,result,info,0)
    table=datas
    len_list=np.ones(len(table),dtype=int)
    #print(table)
    pfs1=prefix_sum.prefixsum(attributes,measure,epsilon)
    pfs1.collection_with_olh(table, len_list, noisy)
    #print(pfs.result)
    return pfs1.result
if __name__ == '__main__':
    #path="E:/test.csv"
    #infopath="E:\info.csv"
    print('loading data...')
    #path = "C:/Users\FlyFF\Desktop\学习\论文\program\dataset_low\Adult/test05.csv"
    #infopath = "C:/Users\FlyFF\Desktop\学习\论文\program\dataset_low\Adult\info.csv"
    path = "E:\PyWorkSpace\program\dataset_low\IPUMS/test02.csv"
    infopath = "E:\PyWorkSpace\program\dataset_low\IPUMS\info.csv"
    result=read_data.readcsv(path)
    result2=copy.deepcopy(result)
    info=read_data.readinfo(infopath)
    #print(type(result))
    print('done.')


    attris = ['MONTH', 'AGE']
    #attris=['Age','Eduction','Hours']
    #measure='Count'
    measure='FAMINC'
    #attris = ['A', 'B']
    #measure = 'V'
    #get orginal distribution
    domain=[]
    attributes = []
    for attri in attris:
        new_attri=attribute(attri, result, info,1)
        attributes.append(new_attri)
        domain.append(new_attri.domain)
    datas = read_data.merge(attributes)
    mea = attribute(measure, result, info,0)
    print('computing true distribution...')
    table_dict=weighted_frequency_query.convert_data(datas, mea)
    #print(table_dict)
    table ,xx = weighted_count_query.convert_data(datas, mea,1)
    #print(table_dict)
    original_data=np.zeros(domain, dtype=float)
    for data in table:
        index=tuple(data)
        original_data[index]+=1
    #print('original_data:',original_data)
    original_data/=len(table)
    print('done.')
    print('generating queries...')
    #set query
    n=100
    epsilon=1
    query_volume=0.25
    data_size=200000
    r_list=[]
    for i in range(n):
        r_list.append(query_module.get_range(domain,query_volume))
    query_list=[]
    for r in r_list:
        query_list.append(query_module.range_query2(attris,r,epsilon,data_size))
    print('done.')
    print('executing PRISM...')
    test1=1
    ERROR_PFS = 0
    if test1:
        #print('result:',result)
        tables_dict=weighted_frequency_query.spilt_table(result, mea)
        marginal_PFS = {}

        for weight in tables_dict.keys():
            print('collecting single dimensions for weight:',weight,'...')
            single_ps = {}
            sub_table = tables_dict[weight]
            for attri in attris:
                attributes=[]
                attributes.append(attribute(attri,sub_table,info,1))
                single_ps[attri]=my_solution2(attributes, measure,epsilon, 1)
            attri_dict = {}
            i = 0
            print('selecting valuable dimensions...')
            for key in single_ps.keys():
                attri_dict[key] = i
                i = i + 1
            value_ps = Synthesizer.value_attribute(single_ps)
            print(value_ps)
            pairs = utils.get_pair(value_ps)
            single_list = single_ps.values()
            print('collecting joint dimensions...')
            double_dict = {}
            attributes = []
            print(pairs)
            for pair in pairs:
                print(pair)
                pair_num = []
                for attri in pair:
                    new_attri = attribute(attri, result, info, 1)
                    attributes.append(new_attri)
                    pair_num.append(attri_dict[attri])
                key_num = tuple(pair_num)
                double_ps = my_solution2(attributes, measure, epsilon, 1)
                double_dict[key_num] = double_ps
                # print("double_dict:",double_dict)
            #print('single_list:',single_list)
            #marginal_PFS[weight] = Synthesizer.Maximum_entropy(single_list, double_dict, domain)
            print('estimating entire model...')
            marginal_PFS[weight] = Synthesizer.Maximum_entropy(single_list, double_dict, domain)
        ERROR_PFS = 0
        for query in query_list:
            answer_OLH = 0
            for weight in marginal_PFS.keys():
                answer_OLH += query.range_query(marginal_PFS[weight]) * len(table_dict[weight]) * weight * mea.weight
            standard_OLH = query.range_query(original_data) * len(table) * mea.weight
            ERROR_PFS += utils.RE(standard_OLH, answer_OLH)
        print('RE_PFS:', ERROR_PFS / n)
    print('done.')

    test1=1
    ERROR_PFS_OLH = 0
    if test1:
        tables_dict=weighted_frequency_query.spilt_table(result, mea)
        marginal_PFS = {}
        for weight in tables_dict.keys():
            single_ps = {}
            sub_table = tables_dict[weight]
            for attri in attris:
                attributes=[]
                attributes.append(attribute(attri,sub_table,info,1))
                print(attribute(attri,sub_table,info,1).domain)
                single_ps[attri]=my_solution3(attributes, measure,epsilon, 1)
            attri_dict = {}
            i = 0
            for key in single_ps.keys():
                attri_dict[key] = i
                i = i + 1
            value_ps = Synthesizer.value_attribute(single_ps)
            pairs = utils.get_pair(value_ps)
            single_list = single_ps.values()

            double_dict = {}
            attributes = []
            for pair in pairs:
                pair_num = []
                for attri in pair:
                    new_attri = attribute(attri, result, info, 1)
                    attributes.append(new_attri)
                    pair_num.append(attri_dict[attri])
                key_num = tuple(pair_num)
                double_ps = my_solution3(attributes, measure, epsilon, 1)
                double_dict[key_num] = double_ps
                # print("double_dict:",double_dict)
            #print('single_list:', single_list)
            #marginal_PFS[weight] = Synthesizer.Maximum_entropy(single_list, double_dict, domain)
            marginal_PFS[weight] = Synthesizer.Maximum_entropy(single_list,None, domain)
        ERROR_PFS_OLH = 0
        for query in query_list:
            answer_OLH = 0
            for weight in marginal_PFS.keys():
                answer_OLH += query.range_query(marginal_PFS[weight]) * len(table_dict[weight]) * weight * mea.weight
            standard_OLH = query.range_query(original_data) * len(table) * mea.weight
            ERROR_PFS_OLH += utils.RE(standard_OLH, answer_OLH)
        print('RE_PFS_olh:', ERROR_PFS_OLH / n)
    #collect 1-way marginals,return as dictionary
    test2=0
    if test2:
        single_ps={}
        for attri in attris:
            attributes=[]
            attributes.append(attribute(attri,result2,info,1))
            single_ps[attri]=my_solution(attributes, measure,epsilon, 1)
        attri_dict={}
        i=0
        for key in single_ps.keys():
            attri_dict[key]=i
            i=i+1
        value_ps=Synthesizer.value_attribute(single_ps)
        pairs=utils.get_pair(value_ps)
        single_list=single_ps.values()


        double_dict={}
        attributes = []
        for pair in pairs:
            pair_num=[]
            for attri in pair:
                new_attri = attribute(attri, result2, info,1)
                attributes.append(new_attri)
                pair_num.append(attri_dict[attri])
            key_num=tuple(pair_num)
            double_ps = my_solution(attributes, measure,epsilon, 1)
            double_dict[key_num]=double_ps
            #print("double_dict:",double_dict)


        marginal=Synthesizer.Maximum_entropy(single_list, double_dict,domain)
        #print(np.sum(marginal))
        ERROR=0
        for query in query_list:
            answer=query.range_query(marginal)*len(table)*mea.weight* 1
            standard=query.range_query(original_data)*len(table)*mea.weight
            ERROR+=utils.RE(standard,answer)
        print('RE:',ERROR/n)


    #OLH
    marginal_OLH={}
    for weight in table_dict.keys():
        list=table_dict[weight]
        olh=control_group.OLH(domain,epsilon)
        marginal_OLH[weight]=olh.collection(list)
    ERROR_OLH = 0
    for query in query_list:
        answer_OLH = 0
        for weight in marginal_OLH.keys():
            answer_OLH+=query.range_query(marginal_OLH[weight]) * len(table_dict[weight]) * weight *mea.weight
        standard_OLH = query.range_query(original_data) * len(table) * mea.weight
        ERROR_OLH += utils.RE(standard_OLH, answer_OLH)
    print('RE_OLH:', ERROR_OLH / n)

    #LHI
    hi_dict={}
    for weight in table_dict.keys():
        list=table_dict[weight]
        hi=control_group.HI(domain,epsilon)
        hi.collection(list)
        hi_dict[weight]=hi
    ERROR_HI=0
    for query in query_list:
        answer_HI=0
        for weight in hi_dict.keys():
            answer_HI+=query.range_query_hi(hi_dict[weight])* len(table_dict[weight]) * weight *mea.weight
        standard_HI = query.range_query(original_data) * len(table) * mea.weight
        ERROR_HI += utils.RE(standard_HI, answer_HI)
    print('RE_HI:', ERROR_HI / n)

    #MG
    attributes = []
    marginal_MG = {}
    for attri in attris:
        new_attri = attribute(attri, result2, info,1)
        attributes.append(new_attri)
    for weight in table_dict.keys():
        list=table_dict[weight]
        clam=control_group.CLAM(attributes,domain,epsilon)
        marginal_MG[weight]=clam.collection(list)
    ERROR_MG = 0
    for query in query_list:
        answer_MG = 0
        for weight in marginal_MG.keys():
            answer_MG += query.range_query(marginal_MG[weight]) * len(table_dict[weight]) * weight * mea.weight
        standard_MG = query.range_query(original_data) * len(table) * mea.weight
        ERROR_MG += utils.RE(standard_MG, answer_MG)
    print('RE_MG:', ERROR_MG / n)
    print('RE_PFS:', ERROR_PFS / n)
    print('RE_HI:', ERROR_HI / n)
    print('RE_OLH:', ERROR_OLH / n)
    print('marginal:',len(attris),'epsilon:',epsilon,'query_volume:',query_volume,'data_size:',data_size)