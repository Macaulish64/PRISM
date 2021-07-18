import numpy as np
import random
import itertools
import math
import utils
import read_data
import attribute_module
from estimation import Synthesizer
from cvxopt import solvers, blas, matrix, spmatrix, spdiag, log, div

class bi:
    def __init__(self, j,v):
        self.hash = j
        self.value = v
class OLH:
    def __init__(self, domain, eps):
        self.family_num=10
        self.domain=self.get_domain(domain)#拍平
        self.weight=self.get_weight(domain)
        self.raw_domain=domain
        l=1
        self.ee = np.exp(eps / l)
        self.g=int(self.ee+1)
        self.p = self.ee / (self.ee + self.g - 1)
        self.q = 1 / (self.ee + self.g - 1)
        self.hash_family=self.get_hash_family()
    def get_domain(self,domain):
        d=1
        for i in range(len(domain)):
            d*=domain[i]
        return [d]
    def get_weight(self,domain):
        weight=np.ones(len(domain),dtype=int)
        for i in range(len(domain)):
            for j in range(i+1,len(domain)):
                weight[i]=weight[i]*domain[j]
        return weight
    def convert_data(self,record):
        data=0
        #print('record:',record,'domain',self.domain,'weight', self.weight)
        for i in range(len(record)):

            #print(int(record[i]),self.weight[i])
            data=data+int(record[i])*self.weight[i]
        #print('final data:',data)
        return data
    def get_hash_family(self):
        hash_family = []
        for i in range(len(self.domain)):
            d=self.domain[i]
            if d>=self.g:
                g=self.g
            else:
                g=d
            hash_group=[]
            j=0
            while j<self.family_num:
                hash = []
                for i in range(d):
                    colume = list(np.zeros(g, dtype=int))
                    k = random.randint(0, g - 1)
                    colume[k] = 1
                    hash.append(colume)
                hash = np.asmatrix(hash)
                if np.linalg.matrix_rank(hash)==g:
                    hash_group.append(hash)
                    j+=1
            hash_family.append(hash_group)
        return hash_family
    def grr(self,data):
        domain=len(data)
        x=0
        for i in range(domain):
            if data[i]==1:
                x=i
        n = len(data)
        y = x
        p_sample = np.random.random_sample()

        if p_sample > self.p and domain>1:
            y = np.random.randint(0, domain - 1)
            if y >= x:
                y += 1
        if p_sample > self.p and domain==1:
            y = x
        return y

    def collection(self,datas):
        perturb_datas=[]
        for data in datas:
            perturb_data=[]
            v = np.zeros(self.domain, dtype=int)
            #print('the data when excuted OLH',data,self.convert_data(data))
            v[self.convert_data(data)] = 1
            v = np.matrix(v)
            j = random.randint(0, self.family_num - 1)
            x = np.dot(v, self.hash_family[0][j])
            x = np.asarray(x)
            y = self.grr(x[0])
            newbi = bi(j, y)
            perturb_data.append(newbi)
            perturb_datas.append(perturb_data)
        marginal=np.zeros(self.domain,dtype=float)

        for data in perturb_datas:
            index=[]
            for i in range(len(data)):
                j=0
                xx=[]
                for line in self.hash_family[i][data[i].hash]:
                    line=np.asarray(line)
                    if line[0][data[i].value]==1:
                        xx.append(j)
                    j=j+1
                index.append(xx)
            items=[]
            for item in itertools.product(*index):
                items.append(item)
            for item in items:
                item=tuple(item)
                marginal[item]+=1/len(items)
        marginal=np.reshape(marginal,self.raw_domain)
        return marginal/marginal.sum()

    def collection_for_pfs(self,datas):
        perturb_datas=[]
        for data in datas:
            perturb_data=[]

            v = np.zeros(self.domain, dtype=int)
            # print('v:',len(v),self.convert_data(data))
            v[self.convert_data(data)] = 1
            v = np.matrix(v)
            j = random.randint(0, self.family_num - 1)
            x = np.dot(v, self.hash_family[0][j])
            x = np.asarray(x)
            y = self.grr(x[0])
            newbi = bi(j, y)
            perturb_data.append(newbi)
            perturb_datas.append(perturb_data)
            # for i in range(len(data)):# len=1
            #     v=np.zeros(self.domain[i],dtype=int)#v.size=12
            #     v[data[i]]=1#将一条记录的对应位置设1
            #     v=np.matrix(v)#转换为one-hot的矩阵
            #     j=random.randint(0, self.family_num - 1)#从哈希簇中选择一个哈希函数
            #     x=np.dot(v,self.hash_family[i][j])
            #     x=np.asarray(x)#映射为domain=2的原始数据
            #     y=self.grr(x[0])
            #     newbi=bi(j,y)
            #     perturb_data.append(newbi)
            # perturb_datas.append(perturb_data)#以上过程将所有记录映射为binary数据，并通过grr扰动
        marginal=np.zeros(self.domain,dtype=float)
        for data in perturb_datas:
            index=[]
            for i in range(len(data)):
                j=0
                xx=[]
                for line in self.hash_family[i][data[i].hash]:
                    line=np.asarray(line)
                    if line[0][data[i].value]==1:
                        xx.append(j)
                    j=j+1
                index.append(xx)
            items=[]
            for item in itertools.product(*index):
                items.append(item)
            for item in items:
                item=tuple(item)
                marginal[item]+=1/len(items)
        marginal = np.reshape(marginal, self.raw_domain)
        if len(self.raw_domain)== 1:
            prefixsum=marginal
            for i in range(self.raw_domain[0]-1):
                prefixsum[i+1]+=prefixsum[i]
        else:
            prefixsum=np.zeros(self.raw_domain,dtype=int)
            expended_marginal=np.zeros((self.raw_domain[0]+1,self.raw_domain[1]+1),dtype=int)
            for i in range(self.raw_domain[0]):
                for j in range(self.raw_domain[1]):
                    expended_marginal[i+1,j+1]=marginal[i,j]
            for i in range(self.raw_domain[0]):
                for j in range(self.raw_domain[1]):
                    expended_marginal[i+1,j+1]=expended_marginal[i,j+1]+expended_marginal[i+1,j]-expended_marginal[i,j]+expended_marginal[i+1,j+1]
                    prefixsum[i,j]=expended_marginal[i+1,j+1]
        return prefixsum


class HI:
    def __init__(self, domain, eps,g):
        self.g=g
        self.b=2
        self.indexs = []
        domain=np.array(domain)//self.g
        self.domain=domain

        self.l=self.get_l()
        #self.epsilon = self.get_eps(eps)
        self.epsilon = eps
        self.model=self.get_model()
    def get_l(self):
        l=[]
        print(self.domain)
        for d in self.domain:
            l.append(math.ceil(math.log(d,self.b))+1)
        return l
    def get_eps(self,epsilon):
        ll=1
        for l in self.l:
            ll*=l
        #eps=epsilon / len(self.domain)/ll
        eps = epsilon / len(self.domain)
        return eps
    def get_model(self):
        model={}
        s = utils.cartesian()
        for i in self.l:
            s.add_data(range(1, i+1))
        for index in itertools.product(*s._data_list):
            self.indexs.append(index)
            shape=[]
            for i in range(len(self.l)):
                shape.append(self.b**(self.l[i]-index[i]))
            shape=tuple(shape)
            model[index]=np.zeros(shape,dtype=float)
        return model

    def get_matrixB(self, double_dict, domain):
        A2 = []
        b2 = []
        for record in double_dict:
            k = record
            v = double_dict.get(record)
            s = utils.cartesian()
            for i in v.shape:
                s.add_data(range(0, i))
            for index1 in itertools.product(*s._data_list):
                a = np.zeros(domain, dtype=float)
                ss = utils.cartesian()
                for i in domain:
                    ss.add_data(range(0, i))
                for index2 in itertools.product(*ss._data_list):
                    if (index2[k[0]] == index1[0] and index2[k[1]] == index1[1]):
                        a[index2] = 1
                A2.append(a.flatten())
            for i in v.flatten():
                b2.append(i)
        A2 = np.asarray(A2)
        A2 = matrix(A2)
        b2 = np.asarray(b2)
        b2 = matrix(b2)
        return A2, b2

    def Max_entropy(self, double_dict, raw_domain):
        domain = []
        for i in range(len(raw_domain)):
            domain.append(math.ceil(raw_domain[i] / self.g))

        solvers.options['show_progress'] = False
        n = 1
        for d in domain:
            n = n * d
        A2, b2 = self.get_matrixB(double_dict, domain)
        A3 = matrix(1, (1, n), 'd')
        b3 = matrix([1.0])
        A = matrix([A2, A3])
        b = matrix([b2, b3])

        def F(x=None, z=None):
            # print(x)
            if x is None: return 0, matrix(1.0 / n, (n, 1))
            # implicit constraint that x should be non-negative
            if min(x) <= 0: return None

            f = x.T * log(x)
            grad = 1.0 + log(x)
            if z is None: return f, grad.T
            H = spdiag(z[0] * x ** -1)
            return f, grad.T, H

        # print(A, b)
        sol = solvers.cp(F, G=A, h=b, A=A3, b=b3)
        p = sol['x']
        p_array = np.array(p).reshape(domain)
        result = np.zeros(raw_domain, dtype=float)
        s = utils.cartesian()
        for i in raw_domain:
            s.add_data(range(0, i))
        for index in itertools.product(*s._data_list):
            i = tuple(np.asarray(index) // self.g)
            result[index] = p_array[i] / (self.g ** len(domain))
        return result

    def estimation(self,lhio,views):
        for index in self.indexs:
            index = np.asarray(index)
            domain = np.asarray(self.domain)
            domain = np.array((domain / (self.b ** (index - 1)) + 1), dtype=int)
            selected_2_way={}
            for key in views.keys():
                i=[]
                for k in list(key):
                    i.append(index[k])
                selected_2_way[key]=lhio.model_dict[views[key]][tuple(i)]
            self.model[tuple(index)]=self.Max_entropy(selected_2_way,domain)
        return 0

    def collection(self,datas):
        for index in self.indexs:
            index = np.asarray(index)
            domain=np.asarray(self.domain)
            domain=np.array((domain/(2**(index-1))+1),dtype=int)
            olh=OLH(domain,self.epsilon)
            record=[]
            for data in datas:
                data=np.asarray(data)
                v=np.array((data/(2**(index-1))),dtype=int)
                record.append(v)
            #print('index:',index,'record[0]',record[0])
            m=olh.collection(record)
            lst=np.where(m>0)
            for n in range(len(lst[0])):
                pos=[]
                for i in range(len(lst)):
                    pos.append(lst[i][n])
                pos=tuple(pos)
                self.model[tuple(index)][pos]=m[pos]
class LHIO:
    def __init__(self, attributes, domain, eps,g):
        self.g=g
        self.b=2
        self.attris=attributes

        domain=np.array(domain)
        self.all_domain = domain//self.g
        for attri in attributes:
            attri.domain=attri.domain//self.g
            data=[]
            for record in attri.data:
                data.append(record//self.g)
            attri.data=data

        self.epsilon = eps
        self.all_domain = domain

        self.views, self.domain, self.p = self.get_view()

        self.index_dict={}
        self.l_dict={}
        self.model_dict={}
        for view in self.views:
            self.l_dict[view]=self.get_l(view)
            self.model_dict[view],self.index_dict[view]=self.get_model(view)

    def get_view(self):
        views = []
        view_domain = {}
        p = []
        attri_name=[]
        tmp_dict = {}
        for i in range(len(self.attris)):
            tmp_dict[self.attris[i].name] = i  # vp_dict中是属性名和序号的映射
        for view in itertools.product(self.attris, self.attris):
            if view[0].name < view[1].name:
                views.append(view)
                p.append((tmp_dict[view[0].name], tmp_dict[view[1].name]))
                attri_name.append((view[0].name,view[1].name))
                view_domain[view]=[view[0].domain, view[1].domain]
        return views, view_domain,p
    def get_l(self,view):
        l = []
        for d in self.domain[view]:
            l.append(math.ceil(math.log(d, self.b)) + 1)
        return l
    def get_model(self,view):
        model = {}
        indexs=[]
        l=self.l_dict[view]
        s = utils.cartesian()
        for i in l:
            s.add_data(range(1, i + 1))
        for index in itertools.product(*s._data_list):
            indexs.append(index)
            shape = []
            for i in range(len(l)):
                shape.append(self.b ** (l[i] - index[i]))
            shape = tuple(shape)
            model[index] = np.zeros(shape, dtype=float)
        return model,indexs

    def collection(self):
        #根据view从table中取出数据datas
        for view in self.views:
            entire_datas = read_data.merge(view)
            #print(entire_datas)
            #print(len(entire_datas),len(self.views))
            entire_datas = random.sample(entire_datas, len(entire_datas) // len(self.views))
            #print(len(entire_datas))
            for index in self.index_dict[view]:
                datas = random.sample(entire_datas, len(entire_datas) // len(self.index_dict[view]))
                #print(len(datas))
                index = np.asarray(index)
                domain = np.asarray(self.domain[view])
                domain = np.array((domain / (self.b ** (index - 1)) + 1), dtype=int)
                olh = OLH(domain, self.epsilon)
                if (len(datas)==0):
                    record=np.array([],dtype=int)
                else:
                    record=np.array((datas / (self.b ** (index - 1))), dtype=int)
                m = olh.collection(record)
                lst = np.where(m > 0)
                for n in range(len(lst[0])):
                    pos = []
                    for i in range(len(lst)):
                        pos.append(lst[i][n])
                    pos = tuple(pos)
                    self.model_dict[view][tuple(index)][pos] = m[pos]

class HDG:
    def __init__(self, attris, eps, partition,g):
        self.epsilon=eps
        self.g1=g
        self.g2=g*5
        self.pairs=self.get_pairs(attris)
        self.TwoD=self.get_TwoD(partition)
        self.OneD=self.get_OneD(partition,attris)
    def get_pairs(self,attris):
        pairs=[]
        for pair in itertools.product(attris,attris):
            if pair[0].name<pair[1].name:
                pairs.append(pair)
        return pairs
    def get_TwoD(self,partition):
        pair_data={}
        for pair in self.pairs:
            domain=[pair[0].domain//self.g2,pair[1].domain//self.g2]
            datas=read_data.merge(pair)
            datas=np.array(random.sample(datas, len(datas)//partition))
            datas=datas//self.g2
            #print(datas[0])
            olh = OLH(domain, self.epsilon)
            pair_data[(pair[0].name,pair[1].name)] = olh.collection(datas)
            #print(pair_data[(pair[0].name,pair[1].name)])
        return pair_data
    def get_OneD(self,partition,attris):
        one_data={}
        for attri in attris:
            domain=[]
            domain.append(attri.domain//self.g1)
            datas = read_data.merge([attri])
            datas=np.array(random.sample(datas, len(datas)//partition))
            datas=datas//self.g1
            olh = OLH(domain, self.epsilon)
            one_data[attri.name] = olh.collection(datas)
        return one_data
    def expend_OneD(self,attri):
        return self.OneD[attri]

    def expend_TwoD(self,pair,d):
        d=d//self.g1
        Two_way=np.zeros([d,d],dtype=float)
        for i in range(d):
            for j in range(d):
                Two_way[i,j]=self.TwoD[pair][i//5,j//5]/5
        return Two_way


class CALM:
    def __init__(self, attributes, domain, eps,g):
        self.g=g
        self.attris = attributes
        domain=np.array(domain)
        self.all_domain = domain//self.g
        for attri in attributes:
            attri.domain=attri.domain//self.g
            data=[]
            for record in attri.data:
                data.append(record//self.g)
            attri.data=data
        self.views, self.domain,self.attri_name,self.p = self.get_view(self.attris)
        self.epsilon = eps
        self.partition=len(self.views)
    def get_view(self, attris):
        views = []
        view_domain = []
        p = []
        attri_name=[]
        vp_dict = {}
        for i in range(len(attris)):
            vp_dict[attris[i].name] = i  # vp_dict中是属性名和序号的映射
        for view in itertools.product(attris, attris):
            if view[0].name < view[1].name:
                views.append(view)
                p.append((vp_dict[view[0].name], vp_dict[view[1].name]))
                attri_name.append((view[0].name,view[1].name))
                view_domain.append([view[0].domain, view[1].domain])
        return views, view_domain,attri_name,p

    def get_matrixB(self, double_dict, domain):
        A2 = []
        b2 = []
        for record in double_dict:
            k = record
            v = double_dict.get(record)
            s = utils.cartesian()
            for i in v.shape:
                s.add_data(range(0, i))
            for index1 in itertools.product(*s._data_list):
                a = np.zeros(domain, dtype=float)
                ss = utils.cartesian()
                for i in domain:
                    ss.add_data(range(0, i))
                for index2 in itertools.product(*ss._data_list):
                    if (index2[k[0]] == index1[0] and index2[k[1]] == index1[1]):
                        a[index2] = 1
                A2.append(a.flatten())
            for i in v.flatten():
                b2.append(i)
        A2 = np.asarray(A2)
        A2 = matrix(A2)
        b2 = np.asarray(b2)
        b2 = matrix(b2)
        return A2, b2

    def Max_entropy(self, double_dict, raw_domain):
        domain = []
        for i in range(len(raw_domain)):
            domain.append(math.ceil(raw_domain[i] / self.g))
        solvers.options['show_progress'] = False
        n = 1
        for d in domain:
            n = n * d
        A2, b2 = self.get_matrixB(double_dict, domain)
        A3 = matrix(1, (1, n), 'd')
        b3 = matrix([1.0])
        A = matrix([A2, A3])
        b = matrix([b2, b3])

        def F(x=None, z=None):
            # print(x)
            if x is None: return 0, matrix(1.0 / n, (n, 1))
            # implicit constraint that x should be non-negative
            if min(x) <= 0: return None

            f = x.T * log(x)
            grad = 1.0 + log(x)
            if z is None: return f, grad.T
            H = spdiag(z[0] * x ** -1)
            return f, grad.T, H

        # print(A, b)
        sol = solvers.cp(F, G=A, h=b, A=A3, b=b3)
        p = sol['x']
        p_array = np.array(p).reshape(domain)

        result = np.zeros(raw_domain, dtype=float)
        s = utils.cartesian()
        for i in raw_domain:
            s.add_data(range(0, i))
        for index in itertools.product(*s._data_list):
            i = tuple(np.asarray(index) // self.g)
            result[index] = p_array[i] / (self.g ** len(domain))
        return result

    def collection(self, all_datas):
        #print(all_datas[0])

        view_data = {}
        for i in range(len(self.views)):
            #print(self.views[i])
            datas = random.sample(all_datas,len(all_datas)//self.partition)
            datas = np.matrix(datas)
            datas = datas[:, self.p[i]]
            datas=np.array(datas)//self.g
            datas = datas.tolist()
            olh = OLH(self.domain[i], self.epsilon)
            #print(datas[0])
            view_data[self.attri_name[i]] = olh.collection(datas)
        return view_data

if __name__ == "__main__":
    # attris=['A','B','C']
    # domain=[6,6,6]
    # datas=[[0,5,1],[1,2,4],[2,3,2],[3,5,5],[4,4,1],[5,3,2],[4,2,4],[3,2,3],[2,1,3],[1,0,0]]
    domain=[3,3]
    datas=[[0,0],[0,1],[1,0],[1,1],[1,2],[2,1],[2,2],[0,2],[2,0]]
    marginal=0
    olh=OLH(domain,100)
    marginal=olh.collection_for_pfs(datas)
    print(marginal)
    # attributes=[]
    # for i in range(len(attris)):
    #     attri=attribute_module.easy_attribute(attris[i],domain,datas,i)
    #     attributes.append(attri)
    # epsilon = 1
	#
    # lhio=LHIO(attributes,domain,epsilon)
    # lhio.collection()
    # for view in lhio.views:
    #     print(lhio.model_dict[view])
    # attri_dict={}
    # attri_dict['A']=0
    # attri_dict['B']=1
    # attri_dict['C']=2
    # involved_view = {}
    # for view in lhio.views:  # view is a pair of class
    #     pair = (view[0].name, view[1].name)
    #     if not (set(pair) - set(attris)):
    #         pair_num = []
    #         for attri in list(pair):
    #             pair_num.append(attri_dict[attri])
    #         involved_view[tuple(pair_num)] = view
    # hio = HI(domain, epsilon)
    # hio.estimation(lhio, involved_view)
    # print(hio.model)

