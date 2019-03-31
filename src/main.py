# Hierarchy Embedding in Poincare Space	
#	by Meng Jiang 		mjiang2@nd.edu
#	by Qingkai Zeng 	qzeng@nd.edu
#	by Tianwen Jiang 	tjiang2@nd.edu 

import sys, os
import argparse
import torch
import random
import time

import numpy as np

from dataCenter import *
from models import *

parser = argparse.ArgumentParser(description='HiJointLearn: Joint Learning of Hierarchical Relations in Hyperbolic Spaces')
parser.add_argument('-data', dest='dataset', default='cs_toy', help='Dataset to run on')
parser.add_argument('-data_dir', dest='data_dir', default='./data', help='Data directory')
parser.add_argument('-out_dir', dest='out_dir', default='./results', help='Directory to store output')
parser.add_argument('-config_dir', dest='config_dir', default='./configs', help='Config directory')
parser.add_argument('-log_dir', dest='log_dir', default='./logs', help='Directory for dumping log files')
parser.add_argument('-reset', dest="reset", action='store_true', help='Clear the cached files (Start a fresh run)')
parser.add_argument('-name', dest='name', default=None, help='Assign a name to the run')

# Embedding hyper-parameters
parser.add_argument('-num_neg_samp', dest='num_neg_samp', default=20, type=int, help='Number of Negative Samples')
parser.add_argument('-batch_sz', dest='batch_sz', default=10000, type=int, help='batch size')
parser.add_argument('-max_epochs', dest='max_epochs', default=200, type=int, help='Maximum number of epoch')
parser.add_argument('-lr', dest='lr', default=0.001, type=float, help='Learning rate')
parser.add_argument('-init', dest='init', default=1e-3, type=float)
parser.add_argument('-EPS', dest='EPS', default=1e-5, type=float)
parser.add_argument('-no-norm', dest='normalize', action='store_false', help='Normalize embeddings after every epoch')
parser.add_argument('-embed_dims', dest='embed_dims', default=2, type=int, help='Embedding dimension')
parser.add_argument('-seed', dest='seed', default=824, type=int, help='Seed for random')
parser.add_argument('-gpu', default=0, type=int, help='Which GPU to run on (-1 for no gpu)')
parser.add_argument('-center_name', dest='center_name', default='data_science')

args = parser.parse_args()

device = torch.device('cuda:'+str(args.gpu) if args.gpu >= 0 else 'cpu')

if torch.cuda.is_available():
	if args.gpu == -1:
		print("WARNING: You have a GPU device, so you should probably run with --cuda")
	else:
		device_id = torch.cuda.current_device()
		print('using device', device_id, torch.cuda.get_device_name(device_id))

args.device = device

if __name__ == '__main__':
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)

	if args.name == None: 
		args.name = args.dataset + '_' + time.strftime("%d_%m_%Y") + '_' + time.strftime("%H:%M:%S")

	args.file_results	= '/results.json'		# Location for loading hyperparameters

	args.out_path  = args.out_dir  + '/' + args.name
	args.data_path = args.data_dir + '/' + args.dataset + '/' + args.dataset + '.tsv'  # Path to the dataset
	args.target_file = args.data_dir + '/' + args.dataset + '/target.tsv'
	if args.reset: os.system('rm -r {}'.format(args.out_path)) # Clear cached files if requeste
	if not os.path.isdir(args.out_path): os.system('mkdir -p ' + args.out_path)	# Create the output directory if doesn't exist

	dc = DataCenter(args)
	model = Poincare(args, dc, args.device).to(device)

	dc.logger.info('Concept Embedding Training')
	for epoch in range(args.max_epochs):
		dc.logger.info('\tEPOCH-' + str(epoch+1))
		embedding_file = args.out_path+'/Concept_Embeddings_EPOCH_'+str(epoch)+'.tsv'
		if checkFile(embedding_file):
			continue
		else:
			if epoch > 0:
				tsv2emb(embedding_file.replace('EPOCH_'+str(epoch), 'EPOCH_'+str(epoch-1)), model.Concept_Embeddings)
		train(args, dc, model, epoch)
		embedding = model.Concept_Embeddings
		emb2tsv(embedding_file, embedding, dc.words)
		nodes_plot(embedding_file, args.target_file, args.out_path+'/Concept_Embeddings_EPOCH_'+str(epoch)+'.png', args.center_name)
	emb2tsv(args.out_path+'/Concept_Embeddings.tsv', embedding, dc.words)
	nodes_plot(embedding_file, args.target_file, args.out_path+'/Concept_Embeddings.pdf', args.center_name)
	