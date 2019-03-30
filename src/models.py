# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch

import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Poincare(nn.Module):
	"""docstring for Poincare"""
	def __init__(self, args, dc):
		super(Poincare, self).__init__()
		self.args = args
		self.dc = dc
		self.Concept_Embeddings = nn.Embedding(len(dc.word2idx), args.embed_dims)
		self.Concept_Embeddings.weight = nn.init.uniform_(self.Concept_Embeddings.weight, a = -args.init, b = args.init)

	def forward(self, left_idx, right_idx):
		print(left_idx.size())
		print(right_idx.size())
		left_embeddings = self.Concept_Embeddings(left_idx)
		right_embeddings = self.Concept_Embeddings(right_idx)
		print(left_embeddings.size())
		print(right_embeddings.size())
		dists = self._poincare(left_embeddings, right_embeddings)
		print('DIST:', dists.size())

	def _arcosh(self, x):
		print(x.size())
		return torch.log(x + torch.sqrt(x * x - 1))

	def _prepare(self, u, v):
		print(u.size(), v.size())
		uu = torch.sum(u * u, -1)
		vv = torch.sum(v * v, -1)
		uv = torch.sum(u * v, -1)

		alpha = 1 - uu
		alpha[alpha<=0] = self.args.EPS
		beta = 1 - vv
		beta[beta<=0] = self.args.EPS

		gamma = 1 + 2 * (uu - 2 * uv + vv) / alpha / beta
		gamma[gamma<1.0] = 1.0
		print(uu.size(), uv.size(), vv.size(), alpha.size(), beta.size(), gamma.size())

		return uu, uv, vv, alpha, beta, gamma

	def _poincare(self, u, v):
		uu, uv, vv, alpha, beta, gamma = self._prepare(u, v)
		return self._arcosh(gamma)


		
		