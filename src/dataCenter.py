import sys
import os

from utils import *

''' *************************************** DATASET PREPROCESSING **************************************** '''
class DataCenter(object):
	"""docstring for DataCenter"""
	def __init__(self, args):
		super(DataCenter, self).__init__()
		self.args = args
		self.logger  = getLogger(args.name, args.log_dir, args.config_dir)
		self.logger.info('Running {}'.format(args.name))

		self.hypernyms = []
		self.hypernymidx = []
		self.words = []
		self.word2idx = {}
		self.widx = 0

		self.read_data(self.args.data_path)

	def read_data(self, filename):
		fr = open(filename,'r')
		for line in fr:
			hypernym = line.strip('\r\n').split('\t')
			self.hypernyms.append(hypernym)
			for word in hypernym:
				if not word in self.word2idx:
					self.words.append(word)
					self.word2idx[word] = self.widx
					assert len(self.words) == self.widx + 1
					self.widx += 1
			self.hypernymidx.append([self.word2idx[word] for word in hypernym])
		fr.close()
		assert len(self.words) == len(self.word2idx)
		assert len(self.hypernyms) == len(self.hypernymidx)

		self.logger.info('num of concepts: '+str(len(self.word2idx)))
		self.logger.info('num of hypernym pairs: '+str(len(self.hypernymidx)))





	# 	self.logger.info('Reading Triples')

	# 	fname = self.p.out_path + self.p.file_triples	# File for storing processed triples
	# 	self.triples_list = []				# List of all triples in the dataset
	# 	self.amb_ent 	  = ddict(int)			# Contains ambiguous entities in the dataset
	# 	self.amb_mentions = {} 				# Contains all ambiguous mentions
	# 	self.isAcronym    = {} 				# Contains all mentions which can be acronyms

	# 	if not checkFile(fname):
	# 		self.ent2wiki = ddict(set)
	# 		self.logger.info(self.p.data_path)
	# 		with codecs.open(self.p.data_path, encoding='utf-8', errors='ignore') as f:
	# 			for line in f:
	# 				origin_trp = json.loads(line.strip())
	# 				trp = {}

	# 				trp['src_sentences'] = origin_trp['src_sentences']
	# 				trp['raw_triple'] = origin_trp['triple']
	# 				sub, rel, obj     = map(str, origin_trp['triple'])

	# 				if sub.isalpha() and sub.isupper(): self.isAcronym[proc_ent(sub)] = 1		# Check if the subject is an acronym
	# 				if obj.isalpha() and obj.isupper(): self.isAcronym[proc_ent(obj)] = 1		# Check if the object  is an acronym

	# 				sub, rel, obj = proc_ent(sub), origin_trp['triple_norm'][1], proc_ent(obj)		# Get Morphologically normalized subject, relation, object

	# 				if len(sub) == 0  or len(rel) == 0 or len(obj) == 0: continue  			# Ignore incomplete triples

	# 				trp['triple'] 		= [sub, rel, obj]
	# 				trp['triple_unique']	= [sub+'|'+str(origin_trp['_id']), rel, obj+'|'+str(origin_trp['_id'])]
	# 				trp['true_sub_link']	= origin_trp['true_link']['subject']
	# 				trp['true_obj_link']	= origin_trp['true_link']['object']

	# 				self.triples_list.append(trp)
	# 				if len(self.triples_list) % 1000 == 0:
	# 					self.logger.info(str(len(self.triples_list))+' done.')

	# 		with open(fname, 'w') as f: 
	# 			f.write('\n'.join([json.dumps(triple) for triple in self.triples_list]))
	# 			self.logger.info('\tCached triples')
	# 	else:
	# 		self.logger.info('\tLoading cached triples')
	# 		with open(fname) as f: 
	# 			self.triples_list = [json.loads(triple) for triple in f.read().split('\n')]

	# def parsing_triples(self):
	# 	self.logger.info('Parsing Triples')

	# 	ent1List, relList, ent2List = [], [], []	# temp variables
	# 	for triple in self.triples_list:			# Get all subject, objects and relations
	# 		ent1List.append(triple['triple_unique'][0])
	# 		relList .append(triple['triple_unique'][1])
	# 		ent2List.append(triple['triple_unique'][2])

	# 	# Get unique list of subject, relations, and objects
	# 	self.rel_list	= list(set(relList))
	# 	self.ent_list	= list(set().union( list(set(ent1List)), list(set(ent2List))))
	# 	self.sub_list   = list(set(ent1List))

	# 	self.clean_ent_list = [ent.split('|')[0] for ent in self.ent_list]

	# 	self.ent2id 	= dict([(v,k) for k,v in enumerate(self.ent_list)])
	# 	self.rel2id 	= dict([(v,k) for k,v in enumerate(self.rel_list)])

	# 	# Creating inverse mapping as well
	# 	self.id2ent   	= invertDic(self.ent2id)
	# 	self.id2rel		= invertDic(self.rel2id)

	# 	self.isSub = {}
	# 	self.sub2id = {}
	# 	for sub in self.sub_list:
	# 		self.isSub[self.ent2id[sub]] = 1
	# 	for sub_id, eid in enumerate(self.isSub.keys()):
	# 		self.sub2id[eid] = sub_id
	# 	self.id2sub = invertDic(self.sub2id)

	# 	# Get frequency of occurence of entities and relations
	# 	self.ent_freq = {}
	# 	self.rel_freq = {}

	# 	for ele in ent1List:
	# 		ent = self.ent2id[ele]
	# 		self.ent_freq[ent] = self.ent_freq.get(ent, 0)
	# 		self.ent_freq[ent] += 1

	# 	for ele in ent2List:
	# 		ent = self.ent2id[ele]
	# 		self.ent_freq[ent] = self.ent_freq.get(ent, 0)
	# 		self.ent_freq[ent] += 1

	# 	for ele in relList:
	# 		rel = self.rel2id[ele]			
	# 		self.rel_freq[rel] = self.rel_freq.get(rel, 0)
	# 		self.rel_freq[rel] += 1

	# 	''' Identifying ambiguous entities '''
	# 	amb_clust = {}
	# 	for trp in self.triples_list:
	# 		sub = trp['triple'][0]
	# 		for tok in sub.split():
	# 			amb_clust[tok] = amb_clust.get(tok, set())
	# 			amb_clust[tok].add(sub)

	# 	for rep, clust in amb_clust.items():
	# 		if rep in clust and len(clust) >= 3:
	# 			self.amb_ent[rep] = len(clust)
	# 			for ele in clust: self.amb_mentions[ele] = 1

	# 	''' Ground truth clustering '''
	# 	self.true_ent2clust = ddict(set)
	# 	for trp in self.triples_list:
	# 		sub_u = trp['triple_unique'][0]
	# 		self.true_ent2clust[sub_u].add(trp['true_sub_link'])
	# 	# self.logger.info(len(self.true_ent2clust))
	# 	# for trp in self.triples_list:
	# 	# 	obj_u = trp['triple_unique'][2]
	# 	# 	self.true_ent2clust[obj_u].add(trp['true_obj_link'])
	# 	self.true_clust2ent = invertDic(self.true_ent2clust, 'm2os')
	# 	self.logger.info('num of true clusts: '+str(len(self.true_clust2ent)))

	# 	fname = self.p.out_path + self.p.file_gold_entClust
	# 	with open(fname, 'w') as f:
	# 		for rep, clust in self.true_clust2ent.items():
	# 			f.write(rep + '\n')
	# 			for ele in clust:
	# 				f.write('\t' + ele + '\n')

	# 	