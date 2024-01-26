import os



inf = open('samples.txt', 'r')

out = open("subdirs.txt", 'w')
for sample in inf.readlines():
	dir_list = os.listdir(sample.rstrip())

	for dir in dir_list:

		sub_dir_list = os.listdir(sample.rstrip()+'/'+dir)
		
		for sub in sub_dir_list:

			path = sample.rstrip()+'/'+dir+'/'+sub+'/\n'
			out.write(path)

out.close()
