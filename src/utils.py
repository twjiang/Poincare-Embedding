import os, sys
import logging
import logging.config
import json
import math
import random
import torch
import pathlib

from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def checkFile(filename):
	return pathlib.Path(filename).is_file()

def update(current, max_epochs, START_LR=0.001, FINAL_LR=0.00001):
	lr = 0
	r = float(current) / max_epochs
	lr = (1 - r) * START_LR + r * FINAL_LR
	return lr

def transitive_isometry(t1, t0):
	(x1, y1), (x0, y0) = t1, t0
	def to_h(z):
		return (1 + z)/(1 - z) * complex(0,1)
	def from_h(h):
		return (h - complex(0,1)) / (h + complex(0,1))
	z1 = complex(x1, y1)
	z0 = complex(x0, y0)
	h1 = to_h(z1)
	h0 = to_h(z0)
	def f(h):
		assert( h0.imag > 0 )
		assert( h1.imag > 0 )
		return h0.imag/h1.imag * (h - h1.real) + h0.real
	def ret(z):
		z = complex(z[0], z[1])
		h = to_h(z)
		h = f(h)
		z = from_h(h)
		return z.real, z.imag
	return ret

def nodes_plot(embedding_file, target_file, fig_file, center_name):
	if embedding_file == '' or target_file == '' or fig_file == '': return
	targets = []
	fr = open(target_file,'r')
	for line in fr:
		arr = line.strip('\r\n').split('\t')
		targets.append(arr[0])
	fr.close()
	targets = list(set([x for x in targets]))
	embeddings = pd.read_csv(embedding_file, header=None, sep="\t", index_col=0)
	fig = plt.figure(figsize=(10,10))
	ax = plt.gca()
	ax.cla()
#	ax.set_xlim((-1.1, 1.1))
#	ax.set_ylim((-1.1, 1.1))
	ax.set_xlim((-1.00, 1.00))
	ax.set_ylim((-1.00, 1.00))
	circle = plt.Circle((0,0), 1., color='black', fill=False)
	#circle = plt.Circle((0,0), 0.60, color='black', fill=False)
	ax.add_artist(circle)
	z = embeddings.ix[center_name]
	isom = transitive_isometry((z[1], z[2]), (0, 0))
	for n in targets:
		z = embeddings.ix[n]
		x, y = isom((z[1], z[2]))
#		x, y = z[1], z[2]
		if n == center_name:
			ax.plot(x, y, 'o', color='g')
			ax.text(x+0.001, y+0.001, n, color='r', fontsize=12, alpha=0.8)
		else:
			ax.plot(x, y, 'o', color='y')
			ax.text(x+0.001, y+0.001, n, color='b', fontsize=12, alpha=0.8)
#	plt.show()
	plt.savefig(fig_file)

def getLogger(name, log_dir, config_dir):
	config_dict = json.load(open(config_dir + '/log_config.json'))

	if os.path.isdir(log_dir) == False: 				# Make log_dir if doesn't exist
		os.system('mkdir {}'.format(log_dir))

	config_dict['handlers']['file_handler']['filename'] = log_dir + '/' + name + '.txt'
	logging.config.dictConfig(config_dict)
	logger = logging.getLogger(name)

	std_out_format = '%(asctime)s - [%(levelname)s] - %(message)s'
	consoleHandler = logging.StreamHandler(sys.stdout)
	consoleHandler.setFormatter(logging.Formatter(std_out_format))
	logger.addHandler(consoleHandler)

	return logger

def emb2tsv(filename, embedding, words):
	fw = open(filename,'w')
	dim = embedding.shape[-1]
	for i in range(len(embedding)):
		s = words[i]
		for d in range(dim):
			s += '\t'+str(embedding[i][d].item())
		fw.write(s+'\n')
	fw.close()

def tsv2emb(filename, embedding):
	fr = open(filename,'r')
	dim = embedding.shape[-1]

	for i, line in enumerate(fr):
		_list = line.strip().split('\t')
		assert len(_list) == (dim+1)
		emb = _list[1:]
		embedding[i] = torch.Tensor([float(ele) for ele in emb])

	fr.close()

def neg_sample(sample_list, except_id, n):
	results = random.sample(sample_list, n)
	if except_id in results:
		results.remove(except_id)
	while len(results) != n:
		results = random.sample(sample_list, n)
		if except_id in results:
			results.remove(except_id)
	assert (len(results) == n) and (except_id not in results)
	return results

def train(args, dc, model, current):
	hypernymidx = shuffle(dc.hypernymidx)
	lr = update(current, args.max_epochs)

	_, whole_right_indx = list(zip(*hypernymidx))

	batches = math.ceil(len(hypernymidx)/args.batch_sz)
	for index in range(batches):
		if (index+1) % 1000 == 0:
			dc.logger.info(f'EPOCH [{current+1}/{args.max_epochs}], batch: [{index+1}/{batches}], ')
		hypernymidx_batch = hypernymidx[index*args.batch_sz: (index+1)*args.batch_sz]
		
		left_indx, right_indx = list(zip(*hypernymidx_batch))
		left_indx_batch = [[i]*(args.num_neg_samp+1) for i in left_indx]
		right_indx_batch = [[i]+neg_sample(whole_right_indx, i, args.num_neg_samp) for i in right_indx]
		left_indx_batch = torch.LongTensor(left_indx_batch).to(args.device)
		right_indx_batch = torch.LongTensor(right_indx_batch).to(args.device)
		# print(left_indx_batch.size(), right_indx_batch.size())

		result_tuple, dists = model(left_indx_batch, right_indx_batch)
		Z = torch.sum(torch.exp(-1 * dists), -1).unsqueeze(-1)
		# print(Z.size())

		left_grad_pos,right_grad_pos = model.backward(left_indx_batch[:,0:1], right_indx_batch[:,0:1], 1-torch.exp(-1 * dists[:,0:1])/Z, (ele[:,0:1].unsqueeze(-1) for ele in result_tuple))
		left_grad_neg, right_grad_neg = model.backward(left_indx_batch[:,1:], right_indx_batch[:,1:], -1*torch.exp(-1 * dists[:,1:])/Z, (ele[:,1:].unsqueeze(-1) for ele in result_tuple))
		left_grad = torch.cat([left_grad_pos, left_grad_neg], 1)
		right_grad = torch.cat([right_grad_pos, right_grad_neg], 1)
	
		model.step(left_indx_batch, left_grad, lr)
		model.step(right_indx_batch, right_grad, lr)
