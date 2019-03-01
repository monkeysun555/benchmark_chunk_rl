import os
import numpy as np

DATA_DIR = '../bw_traces/'

def loadBandwidth(data_dir = DATA_DIR):
	datas = os.listdir(data_dir)
	time_traces = []
	throughput_traces = []
	data_names = []

	a = 0
	for data in datas:
		file_path = data_dir + data
		time_trace = []
		throughput_trace = []
		time = 0.0
		# print(data)
		with open(file_path, 'rb') as f:
			for line in f:
				# parse = line.split(',')
				parse = line.strip('\n')
				# print(parse)
				time_trace.append(time)
				# throughput_trace.append(float(parse[4]))
				throughput_trace.append(float(parse))
				time += 1.0
		# print(throughput_trace)
		time_traces.append(time_trace)
		throughput_traces.append(throughput_trace)
		data_names.append(data)

	return time_traces, throughput_traces, data_names




if __name__ == '__main__':
	loadBandwidth()