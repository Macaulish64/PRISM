import numpy as np
import random
import itertools
from attribute_module import attribute


class range_query:
	def __init__(self):
		self.attributes = []
		self.epsilon = 0
		self.predicate = []
		self.data_size = 0
		self.answer = 0

	# new version
	def __init__(self, attris, predicate, k, epsilon, data_size):
		self.attributes = []
		self.predicate = []
		attri = random.sample(attris, k)
		for i in range(len(attri)):
			self.attributes.append(attri[i])
			self.predicate.append(predicate[i])
		self.epsilon = epsilon
		self.data_size = data_size
		self.answer = 0

	def add_predicate(self, attri, predicate):
		for i in range(len(attri)):
			self.attributes.append(attri[i])
			self.predicate.append(predicate[i])

	def add_answer(self, marginal):
		self.answer = self.range_query(marginal)

	def range_query(self, marginal):
		l = len(marginal.shape)
		c = marginal
		r = self.predicate
		for i in range(l):
			axis = []
			for j in range(l):
				if j == 0:
					axis.append(i)
				elif j == i:
					axis.append(0)
				else:
					axis.append(j)
			axis = tuple(axis)
			c = c.transpose(axis)
			c = c[r[i]]
			c = c.transpose(axis)
		return np.sum(c)

	def range_query_hi(self, hi):
		hi_list = []
		hi_dict = []
		for i in range(len(hi.l)):
			l0 = hi.l[i]
			r = set(self.predicate[i])
			l_list = []
			l_dict = {}

			for l in range(l0, 0, -1):
				b = False
				l_dict[l] = []
				for k in range(np.asarray((hi.domain[i] / (2 ** (l - 1)) + 1), dtype=int)):
					r0 = set(range(2 ** (l - 1) * k, 2 ** (l - 1) * (k + 1)))
					if r0 in r:
						b = True
						l_dict[l].append(k)
						r = r - r0
				if b:
					l_list.append(l)
			hi_dict.append(l_dict)
			hi_list.append(l_list)
			hi_index = []
			for index in itertools.product(*hi_list):
				hi_index.append(index)
			query_dict = {}
			for index in hi_index:
				query_dict[index] = []
				list = []
				for i in len(hi_index[0]):
					dict = hi_dict[i]
					list.append(dict[hi][index])
				for i in itertools.product(*list):
					query_dict[index].append(i)
			sum = 0
			for index in query_dict.keys():
				model = hi.model[index]
				for i in query_dict[index]:
					sum += model[i]
			return sum


class range_query2(range_query):
	def __init__(self, attri, predicate, epsilon, data_size):
		self.attributes = []
		self.predicate = []
		for i in range(len(attri)):
			self.attributes.append(attri[i])
			self.predicate.append(predicate[i])
		self.epsilon = epsilon
		self.data_size = data_size


def get_range(domains, query_volume):
	rang = []
	for domain in domains:
		max = int(domain * (1.0 - query_volume))
		if max > 0:
			l = random.randint(0, max - 1)
		else:
			l = 0
		r = l + domain - max
		rang.append(range(l, r))
	print(rang)
	return rang


if __name__ == '__main__':
	domain = [6, 6]
	qv = 0.1
	for i in range(10):
		get_range(domain, qv)
