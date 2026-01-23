import torch.nn as nn
import torch
from random import sample
from Params import args
from Utils.example_util import move_to_cuda


class NegativeSampleHard(object):

    def __init__(self, batch_size=256, false_neg_size=4, hard_size=5, 
                 graph_stru=None, user2item_dict=None, item2user_dict = None):
        super(NegativeSampleHard, self).__init__()

        self.negative_hard_size = hard_size
        self.false_negative_size = false_neg_size
        self.graph_stru = graph_stru
        
       
        self.batch_size = batch_size
        self.user2item_dict, self.item2user_dict = user2item_dict, item2user_dict

    def hard_items_em(self, all_embed, users, items, batch_input_item_ids):

        x_user, x_item = torch.split(all_embed, [args.user, args.item], dim=0)
        gx_input_user = x_user[users]
        p_score = torch.matmul(gx_input_user, torch.transpose(x_item, 0, 1))
        p_score_exp = torch.exp(p_score)  # 1024*1
        for i in range(len(items)):
            # k = self.eneities_labels.index(true_item[i])
            k = items[i].item()
            p_score_exp[i][k] = 0
            other_true_ids = batch_input_item_ids[i]
            for j in other_true_ids:
                # k = other_true_ids[j]
                p_score_exp[i][j] = 0

        sorted_score, sorted_indices = torch.sort(p_score_exp, dim=-1, descending=True)

        item_id_list = []
        item_embed_list = []
        for i in range(self.batch_size):
            for j in range(self.negative_hard_size):
                k = sorted_indices[i][j].item()
                # print(k)
                # ent_label = self.eneities_labels[k]
                # en_example = self.entites_dataset_all[k]
                # item_input_id_list.append(ent_label)
                item_id_list.append(k)
                item_embed_list.append(x_item[k].unsqueeze(0))
        item_embed_all_tensor = torch.cat((item_embed_list),0)
        return item_embed_all_tensor, item_id_list

    def neg_sampling(self, users, items, all_embed, batch_input_item_ids):

        # entity_list = [entity_tensor.unsqueeze(0) for en_name, entity_tensor in em_bank.items()]
        # entity_tensor_all = torch.cat(entity_list, dim=0)
        
        x_user, x_item = torch.split(all_embed, [args.user, args.item], dim=0)
        
        
        gx_input_user = x_user[users]
        # print(gx_input.shape)

        true_items = items
        true_users = users
        # sort the entities based on similarity and return some hard negatives
        item_embed_all_tensor, hard_item_id_list = self.hard_items_em(all_embed, users, items, batch_input_item_ids)
        """
            em_bank是所有的embedding，k是sentence,v是embeddingtensor
            gx_input 是 head和relation处理之后的 #[batch_size, emb_dim]
            y_input 是hrt对应的sentence原文 #[3, batch_size]
            batch_input_true_ids 是 tail准确的sentenceId
        """
        false_score_list = []
        entity_input_list_lable = []
        # entites_dataset_all_token = []
        # entites_dataset_all_mask = []
        # entites_dataset_all_type = []
        # find the false negative from Neighbor
        for i in range(len(true_items)):
            cur_user = true_users[i].item()
            hard_tail_i = hard_item_id_list[i:i + self.negative_hard_size]
            false_neg_list1 = [hard_tail_i_each for hard_tail_i_each in hard_tail_i if
                               hard_tail_i_each in self.graph_stru[str(cur_user)]['2']]
            num2 = self.false_negative_size - len(false_neg_list1)
            if num2 > 0:
                try:
                    false_neg_list2 = sample(self.graph_stru[cur_user]['2'], num2)
                except:
                    other_true_entity = self.user2item_dict[cur_user] * num2
                    false_neg_list2 = sample(other_true_entity, num2)
                false_neg_list1.extend(false_neg_list2)
            false_neg_list = false_neg_list1[:self.false_negative_size]
            false_entity_list = []

            entity_input_list_lable.extend(false_neg_list)

            for neg_item_id in false_neg_list:
                false_entity_list.append(x_item[neg_item_id].unsqueeze(0))

                # en_example = self.entites_dataset_all[self.eneities_labels.index(neg_item_id)]
                # entites_dataset_all_token.append(torch.LongTensor(en_example['entity_token_ids']).unsqueeze(0))
                # entites_dataset_all_mask.append(torch.LongTensor(en_example['entity_token_mask']).unsqueeze(0))
                # entites_dataset_all_type.append(torch.LongTensor(en_example['entity_token_type_ids']).unsqueeze(0))

            # the false negatives and its probabiltiy
            false_entity_tensor = torch.cat(false_entity_list, dim=0)
            false_score_i = torch.matmul(gx_input_user[i], torch.transpose(false_entity_tensor, 0, 1))
            false_score_list.append(false_score_i.unsqueeze(0))

        # batch data, hard data, and false data to build all tails
        # y_tails = y_input['tail']+y_input['head']+hard_tails+entity_input_list_lable
        items_list = items.tolist()
        y_tails = items_list + hard_item_id_list
        entity_list = [x_item[en_name].unsqueeze(0) for en_name in y_tails]
        entity_tensor_all = torch.cat(entity_list, dim=0)

        # return their socre and assign it as zero if we know it is false tails
        p_score = torch.matmul(gx_input_user, torch.transpose(entity_tensor_all, 0, 1))
        # p_score_exp = torch.exp(p_score)#1024*1
        p_score_exp = nn.functional.normalize(p_score, dim=1)
        # print(p_score_exp)
        for i in range(self.batch_size):
            # k = eneities_labels.index(true_tails[i])
            p_score_exp[i][i] = -10
            other_true_ids = batch_input_item_ids[i]
            for j in other_true_ids:
                false_tail = j
                try:
                    # k_list = [k for k in range(len(a)) if a[k] == 'b']
                    k = true_items[false_tail].item()
                    p_score_exp[i][k] = -10
                except:
                    pass
        # false negative **data
        false_score_tensor = torch.cat(false_score_list, dim=0)
        false_score_tensor = nn.functional.normalize(false_score_tensor, dim=1)

        f_prob = nn.functional.softmax(false_score_tensor, dim=1)
        # print('f_prob',f_prob.shape)256*3

        prob = nn.functional.softmax(p_score_exp, dim=1)
        # print('prob',prob.shape)256*1280
        for l in range(self.batch_size):
            prob[l][l] = 1

        # return prob of tails, prob of false negative tails, false ngative data, false negative label, hard negative data, hard negative label
        return prob, f_prob, {}, entity_input_list_lable, item_embed_all_tensor, hard_item_id_list
