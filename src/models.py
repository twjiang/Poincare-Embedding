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
	def __init__(self, args, dc, device):
		super(Poincare, self).__init__()
		self.args = args
		self.dc = dc
		
		self.Concept_Embeddings = nn.init.uniform_(torch.randn(len(dc.word2idx), args.embed_dims), a = -args.init, b = args.init).to(device)

	def forward(self, left_idx, right_idx):
		# print(left_idx.size())
		# print(right_idx.size())
		left_embeddings = self.Concept_Embeddings[left_idx]
		right_embeddings = self.Concept_Embeddings[right_idx]
		# print(left_embeddings.size())
		# print(right_embeddings.size())
		result_tuple, dists = self._poincare(left_embeddings, right_embeddings)
		# print('DIST:', dists.size())
		return result_tuple, dists

	def _arcosh(self, x):
		# print(x.size())
		return torch.log(x + torch.sqrt(x * x - 1))

	def _prepare(self, u, v):
		# print(u.size(), v.size())
		uu = torch.sum(u * u, -1)
		vv = torch.sum(v * v, -1)
		uv = torch.sum(u * v, -1)

		alpha = 1 - uu
		alpha[alpha<=0] = self.args.EPS
		beta = 1 - vv
		beta[beta<=0] = self.args.EPS

		gamma = 1 + 2 * (uu - 2 * uv + vv) / alpha / beta
		gamma[gamma<1.0] = 1.0
		# print(uu.size(), uv.size(), vv.size(), alpha.size(), beta.size(), gamma.size())

		return (uu, uv, vv, alpha, beta, gamma)

	def _poincare(self, u, v):
		result_tuple = self._prepare(u, v)
		gamma = result_tuple[-1]
		return result_tuple, self._arcosh(gamma)

	def backward(self, left_idx, right_idx, grad_output, result_tuple):
		u = self.Concept_Embeddings[left_idx]
		v = self.Concept_Embeddings[right_idx]
		c = grad_output.unsqueeze(-1)
		uu, uv, vv, alpha, beta, gamma = result_tuple

		c *= 4 / torch.sqrt(gamma * gamma - 1) / alpha / beta
		cu = c * alpha * alpha / 4
		cv = c * beta * beta / 4
		grad_u = (cu * (vv - 2 * uv + 1) / alpha) * u - cu * v
		grad_v = (cv * (uu - 2 * uv + 1) / beta) * v - cv * u
		grad_u[grad_u == float('inf')] = 0.0
		grad_v[grad_v == float('inf')] = 0.0
		grad_u[grad_u == float('-inf')] = 0.0
		grad_v[grad_v == float('-inf')] = 0.0
		grad_u[torch.isnan(grad_u)] = 0.0
		grad_v[torch.isnan(grad_v)] = 0.0
		return grad_u, grad_v

	def step(self, idx, grad, lr):
		self.Concept_Embeddings[idx] -= lr*grad



