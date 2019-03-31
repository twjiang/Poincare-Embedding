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
