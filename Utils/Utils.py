import torch as t
import torch.nn.functional as F
from tqdm import tqdm
import networkx as nx
from collections import defaultdict


def innerProduct(usrEmbeds, itmEmbeds):
	return t.sum(usrEmbeds * itmEmbeds, dim=-1)

def pairPredict(ancEmbeds, posEmbeds, negEmbeds):
	return innerProduct(ancEmbeds, posEmbeds) - innerProduct(ancEmbeds, negEmbeds)

def calcRegLoss(model):
	ret = 0
	for W in model.parameters():
		ret += W.norm(2).square()
	return ret

def contrastLoss(embeds1, embeds2, nodes, temp):
	embeds1 = F.normalize(embeds1, p=2)
	embeds2 = F.normalize(embeds2, p=2)
	pckEmbeds1 = embeds1[nodes]
	pckEmbeds2 = embeds2[nodes]
	nume = t.exp(t.sum(pckEmbeds1 * pckEmbeds2, dim=-1) / temp)
	deno = t.exp(pckEmbeds1 @ embeds2.T / temp).sum(-1)
	return -t.log(nume / deno)

def generate_index_maps(trnData):
	row_to_cols = defaultdict(list)
	col_to_rows = defaultdict(list)

	for r, c in zip(trnData.row, trnData.col):
		row_to_cols[r].append(c)
		col_to_rows[c].append(r)
	return row_to_cols, col_to_rows
