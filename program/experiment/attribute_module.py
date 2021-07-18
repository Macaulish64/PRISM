import csv


# 从文件获取属性
class attribute:
	def __init__(self, attri, result, info, correct):
		position = attribute.get_position(self, result, attri)

		self.name = attri
		# print(self.name,'position::::',position)
		# print(int(info[0][position]))
		domain = int(info[0][position])
		self.domain = int(info[0][position])
		self.weight = int(info[1][position])
		self.data = self.get_data(result, position, correct)

	# print(self.data)
	def get_position(self, result, attri):
		num = 0
		result_attribute = result[0]
		# print(result_attribute)
		for i in range(len(result_attribute)):
			if attri == result[0][i]:
				num = i
		return num

	def get_data(self, result, position, correct):
		result_data = result[:]
		data = []
		del result_data[0]
		# print(self.name)
		i = 0
		for row in result_data:
			if correct:
				data.append(int((int(row[position]) / self.weight) - 1))
			else:
				print(i)
				i += 1
				print(position,row[position])
				data.append(int(int(row[position]) / self.weight))
		return data


class easy_attribute:  # 从变量获取属性，主要用于测试
	def __init__(self, attri, domain, datas, position):
		self.name = attri
		self.domain = domain[position]
		self.weight = 1
		self.data = []
		for data in datas:
			self.data.append(data[position])
		# print(self.data)


if __name__ == '__main__':
	attris = ['A', 'B', 'C']
	domain = [6, 6, 6]
	datas = [[0, 5, 1], [1, 2, 4], [2, 3, 2], [3, 5, 5], [4, 4, 1], [5, 3, 2], [4, 2, 4], [3, 2, 3], [2, 1, 3],
			 [1, 0, 0]]
	for i in range(len(attris)):
		attri = easy_attribute(attris[i], domain, datas, i)
		print(attri.name, attri.domain, attri.data)
