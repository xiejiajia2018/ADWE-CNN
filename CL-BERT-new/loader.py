import pickle
import random
import torch
import numpy as np
from torch.utils.data import Dataset,DataLoader
from transformers import BertTokenizer
import dgl
from dgl import DGLGraph
import dgl.function as fn
from tqdm import tqdm
import pdb
'''扁平式的图'''

def encode_sequence(tokens, max_len, span_loc):
    
    seg_id = torch.LongTensor([0 for i in range(span_loc[0], span_loc[0]+max_len)])


    if span_loc[1]-span_loc[0]==max_len-1:
        att_mask = torch.LongTensor([1 for i in range(span_loc[0], span_loc[0]+max_len)])
        tokens = torch.LongTensor(tokens)
        positions = torch.LongTensor([i for i in range(span_loc[0], span_loc[0]+max_len)])
    else:
        tokens_ = [101]+tokens[span_loc[0]:span_loc[1]+1]+[102]
        tokens = torch.LongTensor([0]*max_len)
        tokens[0:len(tokens_)] = torch.LongTensor(tokens_)

        att_mask_ = [1]+ [1 for i in range(span_loc[0], span_loc[1]+1)] + [1]
        att_mask = torch.LongTensor([0]*max_len)
        att_mask[0:len(att_mask_)] = torch.LongTensor(att_mask_)

        positions = torch.LongTensor([0]+[i for i in range(span_loc[0], span_loc[0]+max_len-1)])
    return tokens, att_mask, seg_id, positions



class custom_dataset(Dataset):  #train dataset
    def __init__(self, pathfile, tokenizer):
        
        with open(pathfile, 'rb') as f:
            raw_data = pickle.load(f) 
        self.data = self.preprocess(raw_data, tokenizer)
               
    @staticmethod
    def preprocess(raw_data, tokenizer):
        processed = []
        for d in raw_data:
            cons, tokens, labels = d['cons'],d['tokens'],d['label']
            whole_sent = [tokenizer.cls_token_id]
            attention_mask = [1]
            tmp = {'labels':[0],
                    'cls_mask':[0],
                    'transit':[0],
                    'lengths': 0,
                    'positions':[],
                    }
            cur = 1
            
            for i, t in enumerate(tokens):
                sub_words = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(t))
                length = len(sub_words)
                whole_sent+= sub_words                
                
                if (labels[i]==1) and length>1:
                    tmp['labels'] += ([1] + [2]*(length-1))
                else:
                    tmp['labels'] += [labels[i]]*length
                tmp['cls_mask'] += [1]*length
                attention_mask += [1]*length
                tmp['transit']+= [i]*length
                cur += length
           
            whole_sent+= [tokenizer.sep_token_id]
            attention_mask += [1]
            tmp['cls_mask' ]+= [0]
            tmp['labels']+= [0]
            tmp['transit']+=[999]
            tmp['lengths'] = cur-1 #subword length, no cls and sep
            total_length = cur+1

            rev_tra = {}
            for i, j in enumerate(tmp['transit']):                   
                    rev_tra[j]=i

            g = DGLGraph()
            g.add_nodes(len(cons))

            encoding_inputs, encoding_masks, encoding_ids, encoding_positions = encode_sequence(whole_sent, total_length, (0,total_length-1))
            g.nodes[0].data['encoding'] = encoding_inputs.unsqueeze(0)
            g.nodes[0].data['encoding_mask'] = encoding_masks.unsqueeze(0)
            g.nodes[0].data['segment_id'] = encoding_ids.unsqueeze(0)
            g.nodes[0].data['positions'] = encoding_positions.unsqueeze(0) 

            assert cons[0][1][0]==0 and cons[0][1][1]+1==len(tokens)
            for i_node in range(1, len(cons)):                
                cons_type, ori_span_loc = cons[i_node]
                
                span_loc = (rev_tra[ori_span_loc[0]],rev_tra[ori_span_loc[1]])
                encoding_inputs, encoding_masks, encoding_ids, encoding_positions = encode_sequence(whole_sent, total_length, span_loc)
                g.nodes[i_node].data['encoding'] = encoding_inputs.unsqueeze(0)
                g.nodes[i_node].data['encoding_mask'] = encoding_masks.unsqueeze(0)
                g.nodes[i_node].data['segment_id'] = encoding_ids.unsqueeze(0)
                g.nodes[i_node].data['positions'] = encoding_positions.unsqueeze(0)

            for x in range(1, len(cons)):
                g.add_edge(x, 0)

            # pdb.set_trace()
            tmp['graph'] = g
            processed.append(tmp)

        return processed

    def __getitem__(self,index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def collate_fn_(batch):
    graph, label, transit = batch[0]['graph'],batch[0]['labels'],batch[0]['transit']
    labTensor = torch.LongTensor(label).cuda()
    transit = torch.LongTensor(transit)
    clsMask = torch.LongTensor(batch[0]['cls_mask']).cuda()
    lengths = torch.LongTensor([batch[0]['lengths']]).cuda()

    batch_graphs = dgl.batch([graph])
    batch_graphs.ndata['encoding'] = batch_graphs.ndata['encoding'].cuda()
    batch_graphs.ndata['encoding_mask'] = batch_graphs.ndata['encoding_mask'].cuda()
    batch_graphs.ndata['segment_id'] = batch_graphs.ndata['segment_id'].cuda()
    batch_graphs.ndata['positions'] = batch_graphs.ndata['positions'].cuda()

    return batch_graphs, clsMask, lengths, labTensor, transit


class Data():
    def __init__(self, pathfile, batch_size, shuffle, config):
        tokenizer = BertTokenizer.from_pretrained(config["bert_model_file"])
        dataset = custom_dataset(pathfile, tokenizer)
        self.data_iter = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_)
