import numpy as np
import itertools
import read_data
import utils
import control_group
import attribute_module
import random


class prefixsum:
	def __init__(self, attributes, measure, eps,g):
		# 当执行用户分组时无需分配隐私预算
		# self.eps=eps / len(attributes)
		self.g = g
		self.eps = eps
		self.ee = np.exp(self.eps)
		self.p = self.ee / (self.ee + 1)
		self.q = 1 / (self.ee + 1)
		self.measure = measure
		self.attributes = []
		raw_domain=self.get_shape(attributes)
		self.domain=[]
		for i in range(len(raw_domain)):
			self.domain.append(int(raw_domain[i]//g))
		self.result = np.zeros(self.domain, dtype=np.int)


	def get_shape(self, attributes):
		domain = []
		for attri in attributes:
			domain.append(attri.domain)
		return np.asarray(domain)


	def perturb(self, datas, bound, eps):
		# print(datas)
		if datas == []:
			print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
		l = len(datas[0])
		ee = np.exp(eps)

		p = ee / (ee + 1)

		datas = np.asarray(datas)
		bound = np.asarray(bound, dtype=int)
		datas = datas <= bound
		datas = np.sum(datas, axis=1)
		datas[datas < l] = 0
		datas[datas == l] = 1
		r = np.random.binomial(1, 1 - p, size=datas.shape)
		datas = np.abs(r - datas)
		return sum(datas)


	def unperturb(self, datas, bound):
		n = len(datas)
		l = len(datas[0])
		perturbed_datas = 0
		for i in range(n):
			y = 1
			for j in range(l):
				if datas[i][j] >= bound[j] + 1:
					y = 0
			perturbed_datas += y  # 这一步改为权重相加即可
		return perturbed_datas


	def consistency(self, n):
		arr = utils.arr_extend(self.result, 0)
		steps = utils.one_steps(self.domain)
		l = len(self.domain)
		minus_list = np.zeros(l, dtype=int)
		for i in range(len(minus_list)):
			minus_list[i] = -1
		minus_step = tuple(minus_list)
		maxindex = utils.t_add(self.domain, minus_step)
		s = utils.cartesian()
		for i in self.domain:
			s.add_data(range(0, i))
		for index in itertools.product(*s._data_list):
			if arr[index] < 0:
				arr[index] = 0
			if arr[index] > n:
				arr[index] = n
			for step in steps:
				index2 = utils.t_add(index, step)
				if arr[index] > arr[index2]:
					arr[index2] = arr[index]
			if index == maxindex:
				arr[index] = n
			# print(maxindex)
		for index in itertools.product(*s._data_list):
			self.result[index] = arr[index]


	def calibration(self, n):
		s = utils.cartesian()
		for i in self.domain:
			s.add_data(range(0, i))
		for index in itertools.product(*s._data_list):
			self.result[index] = int((self.result[index] - n * self.q) / (self.p - self.q))


	# def init_collection(self,):

	def collection(self, datas, len_list, noisy):
		n = len(datas)
		datas = np.asarray(datas)
		datas=datas//self.g
		# print(n)
		s = utils.cartesian()
		for i in self.domain:
			s.add_data(range(0, i))
		# print("iterate the bound")
		bound_num = 0
		for bound in itertools.product(*s._data_list):
			bound_num += 1
		for bound in itertools.product(*s._data_list):
			# sample_datas=random.sample(datas,len(datas)//bound_num)
			# print('collect ',bound)
			if noisy:
				# data = prefixsum.perturb(self, sample_datas, bound,self.eps)
				data = prefixsum.perturb(self, datas, bound, self.eps / bound_num)
			else:
				data = prefixsum.unperturb(self, datas, bound)
			self.result[bound] = data

		if noisy:
			# prefixsum.calibration(self,n)
			# print(self.result)
			prefixsum.consistency(self, n)
		return self.result


	def collection_with_olh(self, datas, len_list, noisy):
		n = len(datas)
		datas = np.asarray(datas)
		datas = datas // self.g
		olh = control_group.OLH(self.domain, self.eps)
		marginal = olh.collection_for_pfs(datas)
		self.result=marginal
		# print(n)
		# s = utils.cartesian()
		# for i in self.domain:
		# 	s.add_data(range(0, i))
		# # print("iterate the bound")
		# for bound in itertools.product(*s._data_list):
		# 	olh = control_group.OLH(self.domain, self.eps)
		# 	marginal = olh.collection_for_pfs(datas)
		# 	b = bound[0]
		# 	for i in range(b):
		# 		self.result[bound] += marginal[i]
		# if noisy:
		# 	# prefixsum.calibration(self,n)
		# 	prefixsum.consistency(self, n)
		# # print('its result of another ips:',self.result,'domain:',self.domain)
		return self.result


if __name__ == '__main__':
	attris = ['A', 'B']
	domain = [4, 4]
	datas=[]
	s = utils.cartesian()
	for i in domain:
		s.add_data(range(0, i))
	for index in itertools.product(*s._data_list):
		datas.append(list(index))
	print(datas)
	attributes = []
	for i in range(len(attris)):
		attri = attribute_module.easy_attribute(attris[i], domain, datas, i)
		attributes.append(attri)
	epsilon = 1
	pfs=prefixsum(attributes, None, epsilon,2)
	print(pfs.collection(datas,None,1))