import os


out = open('out.txt', 'w')
inf = open('test.txt', 'r')

sample_list = []

for sample in inf.readlines():
	target = sample.split('/')[9]
	sample_list.append(target)

sample_list = list(set(sample_list))
print(sample_list)

for target in sample_list:
	print(target)
	same_group = []
	
	
	inf = open('test.txt', 'r')
	for sample in inf.readlines():
		#print(sample)

		if target in sample:
			print('yes', sample)
			files = os.listdir(sample.rstrip())
			for file in files:
				same_group.append(sample.rstrip()+file)

	out.write('{')
	for i in range(len(same_group)):

		if i==len(same_group)-1:
			out.write('"'+same_group[i]+'"}\n')
		else:
			out.write('"'+same_group[i]+'", ')



out.close()


