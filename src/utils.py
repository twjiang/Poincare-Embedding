import os, sys
import logging
import logging.config
import json
import math
import random
import torch

from sklearn.utils import shuffle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
	ax.set_xlim((-0.60, 0.60))
	ax.set_ylim((-0.60, 0.60))
#	circle = plt.Circle((0,0), 1., color='black', fill=False)
	circle = plt.Circle((0,0), 0.60, color='black', fill=False)
	ax.add_artist(circle)
	z = embeddings.ix[center_name]
	isom = transitive_isometry((z[1], z[2]), (0, 0))
	for n in targets:
		z = embeddings.ix[n]
		x, y = isom((z[1], z[2]))
#		x, y = z[1], z[2]
		if n == center_name:
			ax.plot(x, y, 'o', color='g')
			ax.text(x+0.001, y+0.001, n, color='r', fontsize=10, alpha=0.8)
		else:
			ax.plot(x, y, 'o', color='y')
			ax.text(x+0.001, y+0.001, n, color='b', fontsize=10, alpha=0.8)
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
			s += '\t'+str(embedding[i][d])
		fw.write(s+'\n')
	fw.close()

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

	batches = math.ceil(len(hypernymidx)/args.batch_sz)
	for index in range(batches):
		dc.logger.info(f'batch: [{index+1}/{batches}]')
		hypernymidx_batch = hypernymidx[index*args.batch_sz: (index+1)*args.batch_sz]
		
		left_indx, right_indx = list(zip(*hypernymidx_batch))
		left_indx_batch = [[i]*(args.num_neg_samp+1) for i in left_indx]
		right_indx_batch = [[i]+neg_sample(right_indx, i, args.num_neg_samp) for i in right_indx]
		left_indx_batch = torch.LongTensor(left_indx_batch).to(args.device)
		right_indx_batch = torch.LongTensor(right_indx_batch).to(args.device)
		print(left_indx_batch.size(), right_indx_batch.size())

		result_tuple, dists = model(left_indx_batch, right_indx_batch)
		print(dists.size())
		Z = torch.sum(torch.exp(-1 * dists), -1).view(-1,1)
		print(Z.size())
		left_grad_pos,right_grad_pos = model.backward(left_indx_batch[:,0:1], right_indx_batch[:,0:1], 1-torch.exp(-1 * dists[:,0:1])/Z, (ele[:,0:1].view(ele[:,0:1].size(0), ele[:,0:1].size(1), 1) for ele in result_tuple))
		left_grad_neg, right_grad_neg = model.backward(left_indx_batch[:,1:], right_indx_batch[:,1:], -1*torch.exp(-1 * dists[:,1:])/Z, (ele[:,1:].view(ele[:,1:].size(0), ele[:,1:].size(1), 1) for ele in result_tuple))
		left_grad = torch.cat([left_grad_pos, left_grad_neg], 1)
		right_grad = torch.cat([right_grad_pos, right_grad_neg], 1)
		print(left_grad.size(), right_grad.size())
		

	# start = time.clock()
	# left_indx = [0 for i in range(NEG+1)]
	# right_indx = [0 for i in range(NEG+1)]
	# left_grad = np.zeros(((NEG+1),DIM))
	# right_grad = np.zeros(((NEG+1),DIM))
	# dists = np.zeros(NEG+1)
	# exp_neg_dist_value = np.zeros(NEG+1)
	# lr = update(current,START_LR,FINAL_LR)
	# nhypernym = len(hypernymidx)
	# random.shuffle(hypernymidx)
	# for itr in range(nhypernym):
	# 	left_indices[0] = hypernymidx[itr][0]
	# 	right_indices[0] = hypernymidx[itr][1]
	# 	dists[0] = poincare(embedding[left_indices[0]],embedding[right_indices[0]])
	# 	exp_neg_dist_value[0] = exp(-1 * dists[0])
	# 	for k in range(NEG):
	# 		left_indices[k+1] = hypernymidx[itr][0]
	# 		right_indices[k+1] = hypernymidx[random.randint(0,nhypernym-1)][1]
	# 		dists[k+1] = poincare(embedding[left_indices[k+1]],embedding[right_indices[k+1]])
	# 		exp_neg_dist_value[k+1] = exp(-1 * dists[k+1])
	# 	Z = sum(exp_neg_dist_value)
	# 	for k in range(NEG):
	# 		left_grad[k+1],right_grad[k+1] = backward(embedding[left_indices[k+1]],embedding[right_indices[k+1]], \
	# 			-1*exp_neg_dist_value[k+1]/Z,left_grad[k+1],right_grad[k+1])
	# 	left_grad[0],right_grad[0] = backward(embedding[left_indices[0]],embedding[right_indices[0]], \
	# 		1-exp_neg_dist_value[0]/Z,left_grad[0],right_grad[0])
	# 	for k in range(NEG+1):
	# 		embedding[left_indices[k]] += -lr * left_grad[k]
	# 		embedding[right_indices[k]] += -lr * right_grad[k]
	# end = time.clock()
	# print 'epoch ',current,'time:',end-start
	# return embedding
