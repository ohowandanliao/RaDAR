import math
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import networkx as nx
from tqdm import tqdm
import operator
import traceback


def ego_graph_dic_all(entities, G, length = 2):
    ego_dic = {}
    c = 0
    exp_key_dict = {}
    for entity in tqdm(entities):
        nodes_ego = {}
        for k in range(1,length):#1 2
            #H = nx.ego_graph(G, n = entity, radius = 3)
            
            #nodes_ego[str(k)] = list(H.nodes)
            #print(entity)
            try:
                edges = list(nx.bfs_edges(G, source=entity, reverse = False, depth_limit=k))
           
                depth_node = []
                for s, t in edges:
                    depth_node.append(t)
            except Exception as e:
                edges = entity[k+c:k+c+5]
                exp_key_dict[entity] = edges
                c = c+5
                # depth_node = []
                # for t in edges:
                #     depth_node.append(t)
                
            nodes_ego[str(k)] = list(set(depth_node))

        ego_dic[entity] = nodes_ego
    return ego_dic

def move_to_cuda(batchsample):
    if len(batchsample) == 0:
        return {}

    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return [_move_to_cuda(x) for x in maybe_tensor]
        else:
            return maybe_tensor

    return _move_to_cuda(batchsample)  

def dic_true_triple(all_triple):
    
    triple_dic = {}
    for triple in all_triple:
        head = triple.head_id
        relation = triple.relation
        tail = triple.tail_id
        triple_dic[head+'@'+relation] = []
    
    for triple in all_triple:
        head = triple.head_id
        relation = triple.relation
        tail = triple.tail_id
        tail_list = triple_dic[head+'@'+relation]
        if tail not in tail_list:
            tail_list.append(tail)
        triple_dic[head+'@'+relation] = tail_list
        
    return triple_dic

def dic_true_item(trnMat):
    user2item_dict = {}
    item2user_dict = {}
    for item in trnMat.col:
        item2user_dict[item] = []
    for user in trnMat.row:
        user2item_dict[user] = []
    for i, j in zip(trnMat.row, trnMat.col):
        item_list = user2item_dict[i]
        user_list = item2user_dict[j]
        if i not in user_list:
            user_list.append(i)
        if j not in item_list:
            item_list.append(j)
        user2item_dict[i] = item_list
        item2user_dict[j] = user_list
    return user2item_dict, item2user_dict

def dic_true_entities(trnMat):

    user2item_dict = defaultdict(set)
    item2user_dict = defaultdict(set)
    # 只遍历一遍组合，减少了重复的赋值操作
    for i, j in zip(trnMat.row, trnMat.col):
        # 直接添加到集合中，集合自己会处理重复元素的问题，无需手动检查
        user2item_dict[i].add(j)
        item2user_dict[j].add(i)

    # 转换回普通字典，并且将集合转换成列表
    user2item_dict = {user: list(items) for user, items in user2item_dict.items()}
    item2user_dict = {item: list(users) for item, users in item2user_dict.items()}

    return user2item_dict, item2user_dict
    
    

def true_entities(test_user, test_item, user2item_dict, item2user_dict):
    if len(test_user) > 0:
        return [user2item_dict[user.item()] for user in test_user]
    if len(test_item) > 0:
        return [item2user_dict[item.item()] for item in test_item]
    return []
