from torch import nn
import torch.nn.functional as F
import torch
from Diffusion import Denoise, GaussianDiffusion, build_denoise_model
from Params import args
from copy import deepcopy
import numpy as np
import math
import scipy.sparse as sp
from Utils.Utils import contrastLoss, calcRegLoss, pairPredict
import time
import torch_sparse
import dgl
from GraghACL import AclModel

init = nn.init.xavier_uniform_


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.uEmbeds = nn.Parameter(init(torch.empty(args.user, args.latdim)))  # [args.user, args.latdim]
        self.iEmbeds = nn.Parameter(init(torch.empty(args.item, args.latdim)))  # [args.item, args.latdim]
        self.gcnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gnn_layer)])
        self.acl_model = AclModel(hid_dim=args.latdim, temp=args.temp, num_MLP=args.acl_mlp_nums)

        # Diffusion modules (built when enabled)
        if getattr(args, 'use_diff_gcl', 0) == 1:
            self.user_denoise_model = build_denoise_model(args.latdim)
            self.item_denoise_model = build_denoise_model(args.latdim)
            self.diffusion_model = GaussianDiffusion(args.noise_scale, args.noise_min, args.noise_max,
                                                     args.diff_steps).cuda()


    def forward_gcn(self, adj):
        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)  # [args.user + args.item , args.latdim]

        embedsLst = [iniEmbeds]  # [1, args.user + args.item , args.latdim]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)  # [args.user + args.item , args.latdim]

        return mainEmbeds[:args.user], mainEmbeds[args.user:]

    def forward_graphcl(self, adj):
        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

        embedsLst = [iniEmbeds]
        for gcn in self.gcnLayers:
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
        mainEmbeds = sum(embedsLst)  # [args.user + args.item , args.latdim]

        return mainEmbeds

    def forward_graphcl_(self, generator):
        iniEmbeds = torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

        embedsLst = [iniEmbeds]
        count = 0
        for gcn in self.gcnLayers:
            with torch.no_grad():
                adj = generator.generate(x=embedsLst[-1], layer=count)
            embeds = gcn(adj, embedsLst[-1])
            embedsLst.append(embeds)
            count += 1
        mainEmbeds = sum(embedsLst)

        return mainEmbeds

    def split_user_item_diff_forward(self, diffusion_model, denoise_model, x, users, items):
        user_emb, item_emb = torch.split(x, [args.user, args.item], dim=0)
        batch_user_emb = user_emb[users]
        batch_item_emb = item_emb[items]
        # diffused_user_emb = self.diffusion_model.p_sample(self.user_denoise_model, batch_user_emb, args.diff_steps)
        # with torch.no_grad():
        diffused_item_emb = diffusion_model.p_sample(denoise_model, batch_item_emb, args.diff_steps)

        diff_user_emb = user_emb.clone()
        diff_item_emb = item_emb.clone()
        # diff_user_emb[users] = batch_user_emb
        diff_item_emb[items] = diffused_item_emb

        diff_x = torch.cat([diff_user_emb, diff_item_emb], dim=0)
        return diff_x, batch_user_emb, batch_item_emb

    def loss_graphcl_with_diff(self, diffusion_model, denoise_model, x1, x2, users, items):
        loss_cl = self.loss_graphcl(x1, x2, users, items)

        diff_x1, batch_user_emb1, batch_item_emb1 = self.split_user_item_diff_forward(diffusion_model, denoise_model,
                                                                                      x1, users, items)
        diff_x2, batch_user_emb2, batch_item_emb2 = self.split_user_item_diff_forward(diffusion_model, denoise_model,
                                                                                      x2, users, items)

        loss_cl_diff1 = self.loss_graphcl(x1, diff_x1, users, items)
        loss_cl_diff2 = self.loss_graphcl(x2, diff_x2, users, items)

        total_loss = loss_cl + args.diff_alpha * (loss_cl_diff1 + loss_cl_diff2)
        return total_loss

    def loss_graphcl(self, x1, x2, users, items):
        T = args.temp
        user_embeddings1, item_embeddings1 = torch.split(x1, [args.user, args.item], dim=0)
        user_embeddings2, item_embeddings2 = torch.split(x2, [args.user, args.item], dim=0)

        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)
        all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)

        all_embs1_abs = all_embs1.norm(dim=1)
        all_embs2_abs = all_embs2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs,
                                                                                    all_embs2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)

        return loss

    def loss_graphcl_acl(self, x1, x2, users, items, acl_graph):
        T = args.temp
        user_embeddings1, item_embeddings1 = torch.split(x1, [args.user, args.item], dim=0)
        user_embeddings2, item_embeddings2 = torch.split(x2, [args.user, args.item], dim=0)

        user_embeddings1 = F.normalize(user_embeddings1, dim=1)
        item_embeddings1 = F.normalize(item_embeddings1, dim=1)
        user_embeddings2 = F.normalize(user_embeddings2, dim=1)
        item_embeddings2 = F.normalize(item_embeddings2, dim=1)

        user_embs1 = F.embedding(users, user_embeddings1)
        item_embs1 = F.embedding(items, item_embeddings1)
        user_embs2 = F.embedding(users, user_embeddings2)
        item_embs2 = F.embedding(items, item_embeddings2)

        all_embs1 = torch.cat([user_embs1, item_embs1], dim=0)
        all_embs2 = torch.cat([user_embs2, item_embs2], dim=0)

        all_embs1_abs = all_embs1.norm(dim=1)
        all_embs2_abs = all_embs2.norm(dim=1)

        sim_matrix = torch.einsum('ik,jk->ij', all_embs1, all_embs2) / torch.einsum('i,j->ij', all_embs1_abs,
                                                                                    all_embs2_abs)
        sim_matrix = torch.exp(sim_matrix / T)
        pos_sim = sim_matrix[np.arange(all_embs1.shape[0]), np.arange(all_embs1.shape[0])]
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
        loss = - torch.log(loss)

        acl_loss = self.acl_model(x1, x2, acl_graph)
        # return loss + acl_loss
        return acl_loss

    def _spmm(self, support, x, order: int):
        # order=1: Ax ; order=2: A(Ax)
        out1 = torch.spmm(support, x)
        if order >= 2:
            out2 = torch.spmm(support, out1)
            return out1, out2
        return out1, None

    # ACL-lite, BYOL, and Proto branches removed

    def getEmbeds(self):
        self.unfreeze(self.gcnLayers)
        return torch.concat([self.uEmbeds, self.iEmbeds], axis=0)

    def unfreeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = True

    def getGCN(self):
        return self.gcnLayers


# 原始乘积的GCN
class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds, flag=True):
        if (flag):
            return torch.spmm(adj, embeds)
        else:
            return torch_sparse.spmm(adj.indices(), adj.values(), adj.shape[0], adj.shape[1], embeds)


class vgae_encoder(Model):
    def __init__(self):
        super(vgae_encoder, self).__init__()
        hidden = args.latdim  # user item的embed维度
        self.encoder_mean = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden))
        self.encoder_std = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, hidden),
                                         nn.Softplus())

    def forward(self, adj):
        x = self.forward_graphcl(adj)  # [args.user + args.item , args.latdim]

        x_mean = self.encoder_mean(x)  # [args.user + args.item , args.latdim]
        x_std = self.encoder_std(x)  # [args.user + args.item , args.latdim]
        gaussian_noise = torch.randn(x_mean.shape).cuda()
        x = gaussian_noise * x_std + x_mean
        return x, x_mean, x_std


class vgae_decoder(nn.Module):
    def __init__(self, hidden=args.latdim):
        super(vgae_decoder, self).__init__()
        self.decoder = nn.Sequential(nn.ReLU(inplace=False), nn.Linear(hidden, hidden), nn.ReLU(inplace=False),
                                     nn.Linear(hidden, 1)).cuda()
        self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCELoss(reduction='none')

    def forward(self, x, x_mean, x_std, users, items, neg_items, encoder):
        x_user, x_item = torch.split(x, [args.user, args.item], dim=0)

        edge_pos_pred = self.sigmoid(self.decoder(x_user[users] * x_item[items]))
        edge_neg_pred = self.sigmoid(self.decoder(x_user[users] * x_item[neg_items]))

        loss_edge_pos = self.bceloss(edge_pos_pred, torch.ones(edge_pos_pred.shape).cuda())
        loss_edge_neg = self.bceloss(edge_neg_pred, torch.zeros(edge_neg_pred.shape).cuda())
        loss_rec = loss_edge_pos + loss_edge_neg

        kl_divergence = - 0.5 * (1 + 2 * torch.log(x_std) - x_mean ** 2 - x_std ** 2).sum(dim=1)

        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
        regLoss = calcRegLoss(encoder) * args.reg

        beta = 0.1
        loss = (loss_rec + beta * kl_divergence.mean() + bprLoss + regLoss).mean()

        return loss


class vgae(nn.Module):
    def __init__(self, encoder, decoder):
        super(vgae, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data, users, items, neg_items):
        x, x_mean, x_std = self.encoder(data)
        loss = self.decoder(x, x_mean, x_std, users, items, neg_items, self.encoder)
        return loss

    def generate(self, data, edge_index, adj):
        x, _, _ = self.encoder(data)

        edge_pred = self.decoder.sigmoid(self.decoder.decoder(x[edge_index[0]] * x[edge_index[1]]))

        vals = adj._values()
        idxs = adj._indices()
        edgeNum = vals.size()
        edge_pred = edge_pred[:, 0]
        mask = ((edge_pred + 0.5).floor()).type(torch.bool)

        newVals = vals[mask]

        newVals = newVals / (newVals.shape[0] / edgeNum[0])
        newIdxs = idxs[:, mask]

        return torch.sparse.FloatTensor(newIdxs, newVals, adj.shape)


class DenoisingNet(nn.Module):
    def __init__(self, gcnLayers, features):
        super(DenoisingNet, self).__init__()

        self.features = features  # [user_embed + item_embed, embed_dim]

        self.gcnLayers = gcnLayers

        self.edge_weights = []
        self.nblayers = []
        self.selflayers = []
        self.relu = nn.ReLU(inplace=True)

        self.attentions = []
        self.attentions.append([])
        self.attentions.append([])

        hidden = args.latdim

        # import pdb; pdb.set_trace()
        if args.attention_type == 'gate':
            # gate attention
            print(f'use {args.attention_type}')
            self.nblayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
            self.nblayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

            self.selflayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
            self.selflayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

            self.gate_atte_user_0 = Gate(2 * hidden, hidden, nn.ReLU(inplace=True))
            self.gate_atte_user_1 = Gate(2 * hidden, hidden, nn.ReLU(inplace=True))

            self.gate_atte_item_0 = Gate(2 * hidden, hidden, nn.ReLU(inplace=True))
            self.gate_atte_item_1 = Gate(2 * hidden, hidden, nn.ReLU(inplace=True))

            self.attentions_0 = nn.Sequential(nn.Linear(4 * hidden, 1))
            self.attentions_1 = nn.Sequential(nn.Linear(4 * hidden, 1))

        elif args.attention_type == 'pure_gate':
            self.gate_atte_user_0 = Gate(2 * hidden, hidden, nn.ReLU(inplace=True))
            self.gate_atte_user_1 = Gate(2 * hidden, hidden, nn.ReLU(inplace=True))

            self.gate_atte_item_0 = Gate(2 * hidden, hidden, nn.ReLU(inplace=True))
            self.gate_atte_item_1 = Gate(2 * hidden, hidden, nn.ReLU(inplace=True))

            self.attentions_0 = nn.Sequential(nn.Linear(2 * hidden, 1))
            self.attentions_1 = nn.Sequential(nn.Linear(2 * hidden, 1))

        else:
            # original attention
            print(f'use {args.attention_type}')
            self.nblayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
            self.nblayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

            self.selflayers_0 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))
            self.selflayers_1 = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(inplace=True))

            self.attentions_0 = nn.Sequential(nn.Linear(2 * hidden, 1))
            self.attentions_1 = nn.Sequential(nn.Linear(2 * hidden, 1))

    # self.rel_cl = relation_contrast(temperature=0.07, num_neg=args.item)

    def freeze(self, layer):
        for child in layer.children():
            for param in child.parameters():
                param.requires_grad = False

    def get_attention(self, input1, input2, layer=0):
        if args.attention_type == 'gate':
            return self.get_gate_attention(input1, input2, layer)
        elif args.attention_type == 'pure_gate':
            return self.get_pure_gate_attention(input1, input2, layer)
        else:
            return self.get_attention_original(input1, input2, layer)

    def get_attention_deperate(self, input1, input2, layer=0):
        if layer == 0:
            # nb_layer = self.nblayers_0
            # selflayer = self.selflayers_0
            multi_head_attn = self.multi_head_attn
        if layer == 1:
            nb_layer = self.nblayers_1
            selflayer = self.selflayers_1

        # nb_layer_type = identify_layer(nb_layer)
        # nb_selftype = identify_layer(selflayer)

        # if nb_layer_type == 'Linear':
        # 	input1 = nb_layer(input1)
        # elif nb_layer_type == 'MultiheadAttention':
        # 	input1 = input1.unsqueeze(0)
        # 	output1 ,_ = nb_layer(input1, input1, input1)
        # 	input1 = self.relu(output1.squeeze())
        # elif nb_layer_type == 'BatchMultiheadAttention':
        # 	input1 = nb_layer(input1)

        # # input1 = nb_layer(input1)

        # if nb_selftype == 'Linear':
        # 	input2 = selflayer(input2)
        # elif nb_selftype == 'MultiheadAttention':
        # 	input2 = input2.unsqueeze(0)
        # 	output2 ,_ = selflayer(input2, input2, input2)
        # 	input2 = self.relu(output2.squeeze())
        # elif nb_selftype == 'BatchMultiheadAttention':
        # 	input2 = selflayer(input2)

        # input2 = selflayer(input2)

        if layer == 0:
            input10 = self.multi_head_attn(input1, input2)
        if layer == 1:
            input1 = nb_layer(input1)
            input2 = selflayer(input2)
            input10 = torch.concat([input1, input2], axis=1)

        # input10 = torch.concat([input1, input2], axis=1)

        if layer == 0:
            weight10 = self.attentions_multi_head(input10)
        if layer == 1:
            weight10 = self.attentions_1(input10)

        return weight10

    def get_gate_attention(self, input1, input2, layer=0):
        if layer == 0:
            nb_layer = self.nblayers_0
            selflayer = self.selflayers_0
        if layer == 1:
            nb_layer = self.nblayers_1
            selflayer = self.selflayers_1

        input1_linear = nb_layer(input1)
        input2_linear = selflayer(input2)

        # input10 = torch.concat([input1, input2], axis=1)

        if layer == 0:
            input_user = self.gate_atte_user_0(input1, input2)
            input_item = self.gate_atte_item_0(input2, input1)
            # input_user = self.nblayers_0(input_user)
            # input_item = self.selflayers_0(input_item)
            input10 = torch.concat([input_user, input_item, input1_linear, input2_linear], axis=1)
            weight10 = self.attentions_0(input10)
        if layer == 1:
            input_user = self.gate_atte_user_1(input1, input2)
            input_item = self.gate_atte_item_1(input2, input1)
            # input_user = self.nblayers_1(input_user)
            # input_item = self.selflayers_1(input_item)
            input10 = torch.concat([input_user, input_item, input1_linear, input2_linear], axis=1)
            weight10 = self.attentions_1(input10)

        return weight10

    def get_pure_gate_attention(self, input1, input2, layer=0):

        if layer == 0:
            input_user = self.gate_atte_user_0(input1, input2)
            input_item = self.gate_atte_item_0(input2, input1)
            input10 = torch.concat([input_user, input_item], axis=1)
            weight10 = self.attentions_0(input10)
        if layer == 1:
            input_user = self.gate_atte_user_1(input1, input2)
            input_item = self.gate_atte_item_1(input2, input1)
            # input_user = self.nblayers_1(input_user)
            # input_item = self.selflayers_1(input_item)
            input10 = torch.concat([input_user, input_item], axis=1)
            weight10 = self.attentions_1(input10)

        return weight10

    def get_attention_original(self, input1, input2, layer=0):
        if layer == 0:
            nb_layer = self.nblayers_0
            selflayer = self.selflayers_0
        if layer == 1:
            nb_layer = self.nblayers_1
            selflayer = self.selflayers_1

        input1 = nb_layer(input1)
        input2 = selflayer(input2)

        input10 = torch.concat([input1, input2], axis=1)

        if layer == 0:
            weight10 = self.attentions_0(input10)
        if layer == 1:
            weight10 = self.attentions_1(input10)

        return weight10

    def hard_concrete_sample(self, log_alpha, beta=1.0, training=True):
        gamma = args.gamma
        zeta = args.zeta

        if training:
            debug_var = 1e-7
            bias = 0.0
            np_random = np.random.uniform(low=debug_var, high=1.0 - debug_var,
                                          size=np.shape(log_alpha.cpu().detach().numpy()))
            random_noise = bias + torch.tensor(np_random)
            gate_inputs = torch.log(random_noise) - torch.log(1.0 - random_noise)
            gate_inputs = (gate_inputs.cuda() + log_alpha) / beta
            gate_inputs = torch.sigmoid(gate_inputs)
        else:
            gate_inputs = torch.sigmoid(log_alpha)

        stretched_values = gate_inputs * (zeta - gamma) + gamma
        cliped = torch.clamp(stretched_values, 0.0, 1.0)
        return cliped.float()

    def generate(self, x, layer=0):
        f1_features = x[self.row, :]  # 非零的user 坐标对应的向量
        f2_features = x[self.col, :]  # 非零的item 坐标对应的向量

        weight = self.get_attention(f1_features, f2_features, layer)

        mask = self.hard_concrete_sample(weight, training=False)

        mask = torch.squeeze(mask)
        adj = torch.sparse.FloatTensor(self.adj_mat._indices(), mask, self.adj_mat.shape).coalesce()

        ind = deepcopy(adj._indices())
        row = ind[0, :]
        col = ind[1, :]

        # 加 epsilon 防止度为 0 导致的无穷大
        rowsum = torch.sparse.sum(adj, dim=-1).to_dense() + 1e-6
        d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
        d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
        row_inv_sqrt = d_inv_sqrt[row]
        col_inv_sqrt = d_inv_sqrt[col]
        values = torch.mul(adj._values(), row_inv_sqrt)
        values = torch.mul(values, col_inv_sqrt)

        support = torch.sparse.FloatTensor(adj._indices(), values, adj.shape).coalesce()

        return support

    def l0_norm(self, log_alpha, beta):
        gamma = args.gamma
        zeta = args.zeta
        gamma = torch.tensor(gamma)
        zeta = torch.tensor(zeta)
        reg_per_weight = torch.sigmoid(log_alpha - beta * torch.log(-gamma / zeta))

        return torch.mean(reg_per_weight)

    def set_fea_adj(self, nodes, adj):
        # nodes = args.user+args.item, adj = deepcopy(self.handler.torchBiAdj
        self.node_size = nodes
        self.adj_mat = adj

        ind = deepcopy(adj._indices())  # 稀疏矩阵adj的非灵值坐标

        self.row = ind[0, :]
        self.col = ind[1, :]

    def call(self, inputs, training=None):
        if training:
            temperature = inputs
        else:
            temperature = 1.0

        self.maskes = []

        x = self.features.detach()
        layer_index = 0
        embedsLst = [self.features.detach()]

        for layer in self.gcnLayers:
            xs = []
            f1_features = x[self.row, :]
            f2_features = x[self.col, :]

            weight = self.get_attention(f1_features, f2_features, layer=layer_index)
            mask = self.hard_concrete_sample(weight, temperature, training)

            self.edge_weights.append(weight)
            self.maskes.append(mask)
            mask = torch.squeeze(mask)

            adj = torch.sparse.FloatTensor(self.adj_mat._indices(), mask, self.adj_mat.shape).coalesce()
            ind = deepcopy(adj._indices())
            row = ind[0, :]
            col = ind[1, :]

            rowsum = torch.sparse.sum(adj, dim=-1).to_dense() + 1e-6
            d_inv_sqrt = torch.reshape(torch.pow(rowsum, -0.5), [-1])
            d_inv_sqrt = torch.clamp(d_inv_sqrt, 0.0, 10.0)
            row_inv_sqrt = d_inv_sqrt[row]
            col_inv_sqrt = d_inv_sqrt[col]
            values = torch.mul(adj.values(), row_inv_sqrt)
            values = torch.mul(values, col_inv_sqrt)
            support = torch.sparse.FloatTensor(adj._indices(), values, adj.shape).coalesce()

            nextx = layer(support, x, False)
            xs.append(nextx)
            x = xs[0]
            embedsLst.append(x)
            layer_index += 1
        return sum(embedsLst)

    def lossl0(self, temperature):
        l0_loss = torch.zeros([]).cuda()
        for weight in self.edge_weights:
            l0_loss += self.l0_norm(weight, temperature)
        self.edge_weights = []
        return l0_loss

    def forward(self, users, items, neg_items, temperature):
        self.freeze(self.gcnLayers)
        x = self.call(temperature, True)
        x_user, x_item = torch.split(x, [args.user, args.item], dim=0)
        ancEmbeds = x_user[users]
        posEmbeds = x_item[items]
        negEmbeds = x_item[neg_items]
        scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
        bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
        regLoss = calcRegLoss(self) * args.reg

        # pos_score = torch.mm(posEmbeds, x_item.transpose(1,0))
        # neg_score = torch.mm(negEmbeds, x_item.transpose(1,0))
        # rel_loss =  self.rel_cl(F.normalize(pos_score, dim=1), F.normalize(neg_score, dim=1))

        lossl0 = self.lossl0(temperature) * args.lambda0
        return bprLoss + regLoss + lossl0


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            # SimCLR loss
            mask = torch.eye(batch_size).float().to(device)
        elif labels is not None:
            # Supconloss
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        # concat all contrast features at dim 0
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob

        # negative samples
        exp_logits = torch.exp(logits) * logits_mask

        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # avoid nan loss when there's one sample for a certain class, e.g., 0,1,...1 for bin-cls , this produce nan for 1st in Batch
        # which also results in batch total loss as nan. such row should be dropped
        pos_per_sample = mask.sum(1)  # B
        pos_per_sample[pos_per_sample < 1e-6] = 1.0
        mean_log_prob_pos = (mask * log_prob).sum(1) / pos_per_sample  # mask.sum(1)

        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss


def identify_layer(layer):
    # 先检查layer是不是可迭代的，比如nn.Sequential
    if hasattr(layer, '__getitem__'):
        # 检查第一个子层的类型
        first_sublayer = layer[0]
    else:
        first_sublayer = layer

    # 检查子层的类型
    if isinstance(first_sublayer, nn.Linear):
        return "Linear"
    elif isinstance(first_sublayer, nn.MultiheadAttention):
        return "MultiheadAttention"
    elif isinstance(first_sublayer, BatchMultiheadAttention):
        return "BatchMultiheadAttention"
    else:
        return "Unknown structure"


class BatchMultiheadAttention(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4):
        super(BatchMultiheadAttention, self).__init__()
        self.nb_layer = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.relu = nn.ReLU()

    def forward(self, input1, input2):
        # 假设input1的形状是[149490, 32]
        batch_size = 1000  # 你可以根据硬件限制来调整这个批次大小
        num_batches = (input1.size(0) + batch_size - 1) // batch_size  # 计算需要多少个批次

        outputs = []
        for i in range(num_batches):
            # 逐个批次处理
            batch_input1 = input1[i * batch_size:(i + 1) * batch_size, :]
            batch_input1 = batch_input1.unsqueeze(1)  # MultiheadAttention需要三维的输入 [batch_size, seq_len, embed_dim]

            # 逐个批次处理
            batch_input2 = input2[i * batch_size:(i + 1) * batch_size, :]
            batch_input2 = batch_input2.unsqueeze(1)  # MultiheadAttention需要三维的输入 [batch_size, seq_len, embed_dim]

            # 必须保证kv输入的序列长度一致，这里简单处理所有批次都是相同的input
            output1, _ = self.nb_layer(batch_input1, batch_input2, batch_input1)
            output1 = self.relu(output1.squeeze())

            outputs.append(output1)

        # 将所有输出按顺序拼回一个tensor
        final_output = torch.cat(outputs, dim=0)
        return final_output


class Gate(nn.Module):
    def __init__(self,
                 input_size,
                 output_size,
                 gate_activation=torch.sigmoid):
        super(Gate, self).__init__()
        self.output_size = output_size
        self.gate_activation = gate_activation
        self.g = nn.Linear(input_size, output_size)
        self.g1 = nn.Linear(output_size, output_size, bias=False)
        self.g2 = nn.Linear(input_size - output_size, output_size, bias=False)
        self.gate_bias = nn.Parameter(torch.zeros(output_size))

    def forward(self, x_ent, x_lit):
        x = torch.cat([x_ent, x_lit], x_lit.ndimension() - 1)
        g_embedded = torch.tanh(self.g(x))
        gate = self.gate_activation(self.g1(x_ent) + self.g2(x_lit) + self.gate_bias)
        output = (1 - gate) * x_ent + gate * g_embedded
        return output



