import torch
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Params import args
from Model import Model, vgae_encoder, vgae_decoder, vgae, DenoisingNet
from DataHandler import DataHandler
from Diffusion import Denoise, GaussianDiffusion, build_denoise_model
import numpy as np
from Utils.Utils import calcRegLoss, pairPredict
from Utils.example_util import *
import os
from pathlib import Path
from copy import deepcopy
import scipy.sparse as sp
import random
import datetime
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from negative_sample import NegativeSampleHard
from GraghACL import AclModel
import dgl


class Coach:
    def __init__(self, handler):
        self.handler = handler
        current_time = datetime.datetime.now().strftime("%m%d_%H%M%S")  # 月日_时分秒
        self.resume_ckpt = args.resume_ckpt.strip()
        resume_time = None
        if self.resume_ckpt:
            ckpt_path = Path(self.resume_ckpt)
            # expected: checkpoint/{exp}/{exp}_{time}/{data}/best.pth
            if ckpt_path.name == 'best.pth' and len(ckpt_path.parents) >= 2:
                exp_time_dir = ckpt_path.parents[1].name
                prefix = f"{args.exp}_"
                if exp_time_dir.startswith(prefix):
                    resume_time = exp_time_dir[len(prefix):]
        if resume_time:
            current_time = resume_time

        self.file_format = f"{args.exp}/{args.exp}_{current_time}/{args.data}/topk_{args.topk}_lr_{args.lr}_dim_{args.latdim}_" + \
                           f"gnn_layer_{args.gnn_layer}_gamma_{args.gamma}_ib_reg_{args.ib_reg}_epoch_{args.epoch}_acl_mlp_nums_{args.acl_mlp_nums}_" + \
                           f"acl_ratio_{args.acl_ratio}_attention_type_{args.attention_type}_cl_type_{args.cl_type}"

        self.result_dir = f"./result/{args.exp}/{args.exp}_{current_time}/{args.data}/"
        self.ckpt_dir = f"./checkpoint/{args.exp}/{args.exp}_{current_time}/{args.data}/"
        os.makedirs(self.result_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.result_path = f"./result/{self.file_format}.txt"
        if not (self.resume_ckpt and os.path.exists(self.result_path)):
            with open(self.result_path, "w") as f:
                f.write(f"{datetime.datetime.now()}\n")
        else:
            with open(self.result_path, "a") as f:
                f.write(f"resume training at {datetime.datetime.now()}\n")

        self.makeWrite(str(vars(args)))

        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', self.handler.trnLoader.dataset.__len__())
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

        self.best_ckpt_path = os.path.join(self.ckpt_dir, 'best.pth')

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def makeWrite(self, content):
        with open(self.result_path, 'a') as f:
            f.write(f"{content} \n")

    def run(self):
        self.prepareModel()
        log('Model Prepared')

        recallMax = 0
        ndcgMax = 0
        bestEpoch = 0
        stloc = 0
        if self.resume_ckpt:
            stloc, recallMax, ndcgMax, bestEpoch = self.load_checkpoint(self.resume_ckpt)
        log('Model Initialized')

        self.acl_graph = self.handler.dgl_graph_generator()

        start_training_time = datetime.datetime.now()
        self.mark_training(start_time=start_training_time, end_time=None)

        printContent = ''
        for ep in range(stloc, args.epoch):
            temperature = max(0.05, args.init_temperature * pow(args.temperature_decay, ep))
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch(temperature, ep)
            printContent = self.makePrint('Train', ep, reses, tstFlag)
            log(printContent)
            self.makeWrite(printContent)
            if tstFlag:
                reses = self.testEpoch()
                # track best performance
                if reses['Recall'] > recallMax:
                    recallMax = reses['Recall']
                    ndcgMax = reses['NDCG']
                    bestEpoch = ep
                    if getattr(args, 'save_best', 1) == 1:
                        self.save_checkpoint(ep)
                printContent = self.makePrint('Test', ep, reses, tstFlag)
                log(printContent)
                self.makeWrite(printContent)
            print()
        printContent = f"Best epoch : {bestEpoch} , Recall :  {recallMax}  , NDCG : {ndcgMax}"
        log(printContent)
        self.makeWrite(printContent)
        self.mark_training(start_time=start_training_time, end_time=datetime.datetime.now())

    def load_checkpoint(self, ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.model.load_state_dict(ckpt.get('model', {}), strict=False)
        self.generator_1.load_state_dict(ckpt.get('generator_1', {}), strict=False)
        self.generator_2.load_state_dict(ckpt.get('generator_2', {}), strict=False)

        start_epoch = ckpt.get('epoch', -1) + 1
        recallMax = 0
        ndcgMax = 0
        bestEpoch = 0
        metrics = ckpt.get('metrics', {})
        if isinstance(metrics, dict) and metrics.get('TestRecall'):
            recall_list = metrics.get('TestRecall', [])
            ndcg_list = metrics.get('TestNDCG', [])
            recallMax = max(recall_list) if recall_list else 0
            if recall_list:
                best_idx = recall_list.index(recallMax)
                bestEpoch = best_idx * args.tstEpoch
                if ndcg_list and len(ndcg_list) > best_idx:
                    ndcgMax = ndcg_list[best_idx]
        log(f"Resume from {ckpt_path}, start_epoch={start_epoch}, prev_best_epoch={bestEpoch}")
        self.makeWrite(f"Resume from {ckpt_path}, start_epoch={start_epoch}, prev_best_epoch={bestEpoch}")
        return start_epoch, recallMax, ndcgMax, bestEpoch

    def save_checkpoint(self, epoch: int):
        state = {
            'epoch': epoch,
            'args': vars(args),
            'model': self.model.state_dict(),
            'generator_1': self.generator_1.state_dict(),
            'generator_2': self.generator_2.state_dict(),
            'metrics': self.metrics,
        }
        torch.save(state, self.best_ckpt_path)

    def mark_training(self, start_time, end_time):
        if start_time is None:
            raise ValueError("start_time is None!")
        if start_time is not None and end_time is None:
            start_training_content = f"start training at {start_time}"
            log(start_training_content)
            self.makeWrite(start_training_content)
        elif start_time is not None and end_time is not None:
            end_training_content = f"end training at {end_time}"
            log(end_training_content)
            self.makeWrite(end_training_content)

            execution_time = end_time - start_time
            total_seconds = execution_time.total_seconds()
            hours, remainder = divmod(total_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            execution_time_content = f"Execution time: {int(hours)} hours, {int(minutes)} minutes, {seconds:.2f} seconds"
            log(execution_time_content)
            self.makeWrite(execution_time_content)

    def prepareModel(self):
        self.model = Model().cuda()  # uEmbeds, iEmbeds, gcnLayers

        encoder = vgae_encoder().cuda()  # 初始化mean和std，forward的时候构建高斯分布
        decoder = vgae_decoder().cuda()  # TODO
        self.generator_1 = vgae(encoder, decoder).cuda()
        self.generator_2 = DenoisingNet(self.model.getGCN(), self.model.getEmbeds()).cuda()
        self.generator_2.set_fea_adj(args.user + args.item, deepcopy(self.handler.torchBiAdj).cuda())

        self.opt = torch.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        self.opt_gen_1 = torch.optim.Adam(self.generator_1.parameters(), lr=args.lr, weight_decay=0)
        self.opt_gen_2 = torch.optim.Adam(filter(lambda p: p.requires_grad, self.generator_2.parameters()), lr=args.lr,
                                          weight_decay=0, eps=args.eps)
        # diffusion modules are constructed inside Model when enabled; avoid duplicates here

    def trainEpoch(self, temperature, ep):
        trnLoader = self.handler.trnLoader
        trnLoader.dataset.negSampling()
        generate_loss_1, generate_loss_2, bpr_loss, im_loss, ib_loss, reg_loss, supcon_loss = 0, 0, 0, 0, 0, 0, 0
        steps = trnLoader.dataset.__len__() // args.batch
        epDiLoss = 0

        for i, tem in enumerate(trnLoader):
            data = deepcopy(self.handler.torchBiAdj).cuda()

            data1 = self.generator_generate(self.generator_1)

            self.opt.zero_grad()
            self.opt_gen_1.zero_grad()
            self.opt_gen_2.zero_grad()

            ancs, poss, negs = tem
            ancs = ancs.long().cuda()
            poss = poss.long().cuda()
            negs = negs.long().cuda()

            out1 = self.model.forward_graphcl(data1)
            out2 = self.model.forward_graphcl_(self.generator_2)

            loss = self.model.loss_graphcl(out1, out2, ancs, poss).mean() * args.ssl_reg
            im_loss += float(loss)
            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

            # info bottleneck
            _out1 = self.model.forward_graphcl(data1)
            _out2 = self.model.forward_graphcl_(self.generator_2)

            loss_ib = None
            if args.cl_type == 'acl':
                # loss_ib = self.model.loss_graphcl(_out1, out1.detach(), ancs, poss) + self.model.loss_graphcl(_out2,out2.detach(), ancs, poss)
                loss_ib = self.model.loss_graphcl_acl(_out1, out1.detach(), ancs, poss,
                                                      self.acl_graph) * args.acl_ratio + self.model.loss_graphcl_acl(
                    _out2, out2.detach(), ancs, poss, self.acl_graph)
            elif args.cl_type == 'mix':
                loss_ib = self.model.loss_graphcl_acl(_out1, out1.detach(), ancs, poss,
                                                      self.acl_graph) * args.acl_ratio + self.model.loss_graphcl_acl(
                    _out2, out2.detach(), ancs, poss, self.acl_graph) + self.model.loss_graphcl(_out1, out1.detach(),
                                                                                                ancs,
                                                                                                poss) + self.model.loss_graphcl(
                    _out2, out2.detach(), ancs, poss)
            else:
                loss_ib = self.model.loss_graphcl(_out1, out1.detach(), ancs, poss) + self.model.loss_graphcl(_out2,
                                                                                                              out2.detach(),
                                                                                                              ancs,
                                                                                                              poss)
            loss = loss_ib.mean() * args.ib_reg
            ib_loss += float(loss)
            loss.backward()

            self.opt.step()
            self.opt.zero_grad()

            # BPR
            usrEmbeds, itmEmbeds = self.model.forward_gcn(data)
            ancEmbeds = usrEmbeds[ancs]
            posEmbeds = itmEmbeds[poss]
            negEmbeds = itmEmbeds[negs]

            scoreDiff = pairPredict(ancEmbeds, posEmbeds, negEmbeds)
            bprLoss = - (scoreDiff).sigmoid().log().sum() / args.batch
            regLoss = calcRegLoss(self.model) * args.reg
            loss = bprLoss + regLoss
            bpr_loss += float(bprLoss)
            reg_loss += float(regLoss)
            loss.backward()

            loss_1 = self.generator_1(deepcopy(self.handler.torchBiAdj).cuda(), ancs, poss, negs)
            loss_2 = self.generator_2(ancs, poss, negs, temperature)

            loss = loss_1 + loss_2
            generate_loss_1 += float(loss_1)
            generate_loss_2 += float(loss_2)
            loss.backward()

            self.opt.step()
            self.opt_gen_1.step()
            self.opt_gen_2.step()

            # DDR-style denoising regularizer on item embeddings (optional)
            if args.use_diff_gcl == 1 and getattr(args, 'lambda_ddr', 0.0) > 0:
                # DDR 正则仅在 warmup 之后启用，并用很小权重，避免训练前期漂移
                if ep >= getattr(args, 'ddr_warmup', 20):
                    with torch.no_grad():
                        _, itmEmbeds_ddr = self.model.forward_gcn(data)
                    self.opt.zero_grad()
                    ddr_loss = self.model.diffusion_model.training_losses_tmp(self.model.item_denoise_model, itmEmbeds_ddr).mean()
                    epDiLoss += float(ddr_loss)
                    (ddr_loss * getattr(args, 'lambda_ddr', 0.1)).backward()
                    self.opt.step()

            log('Step %d/%d: gen 1 : %.3f ; gen 2 : %.3f ; bpr : %.3f ; im : %.3f ; ib : %.3f ; reg : %.3f  ' % (
                i,
                steps,
                generate_loss_1,
                generate_loss_2,
                bpr_loss,
                im_loss,
                ib_loss,
                reg_loss,
            ), save=False, oneline=True)

        ret = dict()
        ret['Gen_1 Loss'] = generate_loss_1 / steps
        ret['Gen_2 Loss'] = generate_loss_2 / steps
        ret['BPR Loss'] = bpr_loss / steps
        ret['IM Loss'] = im_loss / steps
        ret['IB Loss'] = ib_loss / steps
        ret['Reg Loss'] = reg_loss / steps
        if args.use_diff_gcl == 1:
            ret['Diffusion Loss'] = epDiLoss / steps

        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epRecall, epNdcg = [0] * 2
        hintsEpRecall = [0] * len(args.hints)
        hintsEpNdcg = [0] * len(args.hints)
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            usrEmbeds, itmEmbeds = self.model.forward_gcn(self.handler.torchBiAdj)
            allPreds = torch.mm(usrEmbeds[usr], torch.transpose(itmEmbeds, 1, 0)) * (1 - trnMask) - trnMask * 1e8
            _, topLocs = torch.topk(allPreds, args.topk)
            recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            epRecall += recall
            epNdcg += ndcg
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False,
                oneline=True)
            for idx, hint in enumerate(args.hints):
                if hint == args.topk:
                    hintsEpRecall[idx] = epRecall
                    hintsEpNdcg[idx] = epNdcg
                    continue
                _, topLocs = torch.topk(allPreds, hint)
                hintRecall, hintNdcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
                hintsEpRecall[idx] += hintRecall
                hintsEpNdcg[idx] += hintNdcg
        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        for idx, hint in enumerate(args.hints):
            ret[f'Recall@{hint}'] = hintsEpRecall[idx] / num
            ret[f'NDCG@{hint}'] = hintsEpNdcg[idx] / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            if temTstLocs is None or len(temTstLocs) == 0:
                continue
            tstNum = len(temTstLocs)
            # standard denominator for Recall: |GT|
            denom = tstNum
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / max(denom, 1)
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def generator_generate(self, generator):
        edge_index = []
        edge_index.append([])
        edge_index.append([])
        adj = deepcopy(self.handler.torchBiAdj)
        idxs = adj._indices()

        with torch.no_grad():
            view = generator.generate(self.handler.torchBiAdj, idxs, adj)

        return view


def seed_it(seed):
    random.seed(seed)
    os.environ["PYTHONSEED"] = str(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.manual_seed(seed)


if __name__ == '__main__':
    # with torch.cuda.device(args.gpu):
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
    logger.saveDefault = True
    seed_it(args.seed)

    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')

    coach = Coach(handler)

    coach.run()
