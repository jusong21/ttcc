import os

path = '/pnfs/iihe/cms/store/user/jusong/topNanoAODv9/2018UL'
path = '/pnfs/iihe/cms/ph/sc4/store/mc/'
dirs = ['RunIISummer20UL16NanoAODAPVv9/','RunIISummer20UL16NanoAODv9/','RunIISummer20UL17NanoAODv9/','RunIISummer20UL18NanoAODv9/']

data = []
f = open('mc_global.txt', 'r')
for line in f.readlines():
	name = line.split('/')[1]
	data.append(name)

print(data)
print(len(data))

out = open('samples_list.txt', 'w')

for dir in dirs:
	p = path+dir
	file_list = os.listdir(p)
	
	for file in file_list:
		for n in data:
			if n in file:
				out.write(p+file+'\n')
out.close()



