import argparse
import torch
import time
import json
import numpy as np
import math
import random
import xml.etree.ElementTree as ET
from subprocess import check_output
import torch.nn as nn
import torch.nn.functional as F

np.random.seed(1337)
random.seed(1337)
torch.manual_seed(1337)
torch.cuda.manual_seed(1337)

def init_weight(weight, method):
    if method == 'orthogonal':
        nn.init.orthogonal_(weight)
    elif method == 'xavier':
        nn.init.xavier_uniform_(weight)
    elif method == 'kaiming':
        nn.init.kaiming_uniform_(weight)
    elif method == 'none':
        pass
    else:
        raise Exception('Unknown init method')

def nn_init(nn_module, method='xavier'):
    for param_name, _ in nn_module.named_parameters():
        if isinstance(nn_module, nn.Sequential):
            i, name = param_name.split('.', 1)
            param = getattr(nn_module[int(i)], name)
        else:
            param = getattr(nn_module, param_name)
        if param_name.find('weight') > -1:
            init_weight(param, method)
        elif param_name.find('bias') > -1:
            nn.init.uniform_(param, -1e-4, 1e-4)

class Model(torch.nn.Module):
    def __init__(self, gen_emb, domain_emb,offset, num_classes=3, dropout=0.5, crf=False):
        super(Model, self).__init__()
        self.gen_embedding = torch.nn.Embedding(gen_emb.shape[0], gen_emb.shape[1])
        self.gen_embedding.weight = torch.nn.Parameter(torch.from_numpy(gen_emb), requires_grad=False)
        self.domain_embedding = torch.nn.Embedding(domain_emb.shape[0], domain_emb.shape[1])
        self.domain_embedding.weight = torch.nn.Parameter(torch.from_numpy(domain_emb), requires_grad=False)
        # self.start = Variable(torch.Tensor(np.zeros(1,300)))
        self.conv1 = torch.nn.Conv1d(450, 128, 3, padding=1)
        self.conv2 = torch.nn.Conv1d(450, 128, 3, padding=1)

        # self.conv1=torch.nn.Conv1d(gen_emb.shape[1], 256, 3, padding=2 )
        # self.conv2=torch.nn.Conv1d(256, 256, 5, padding=1 )

        self.conv11 = torch.nn.Conv1d(900, 128, 3, padding=1)
        self.conv22 = torch.nn.Conv1d(900, 128, 3, padding=1)

        self.dropout = torch.nn.Dropout(dropout)

        self.conv3 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self.conv4 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self.conv5 = torch.nn.Conv1d(256, 256, 3, padding=1)

        self.conv33 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self.conv44 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self.conv55 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self.conv6 = torch.nn.Conv1d(256, 256,3, padding=1)
        # self.conv7 = torch.nn.Conv1d(256, 256, 3, padding=1)
        self.linear_ae1 = torch.nn.Linear(256*2 , 256*2)
        nn_init(self.linear_ae1, 'xavier')
        self.linear_ae1 = torch.nn.Linear(600, 256)
        
        
        self.linear_ae2 = torch.nn.Linear(300 , 150)
        
        self.linear_ae = torch.nn.Linear(256 , num_classes)
        of = offset
        self.crf_flag = crf
        if self.crf_flag:
            from allennlp.modules import ConditionalRandomField
            self.crf = ConditionalRandomField(num_classes)

        self.attn_0 = nn.GRU(300+100, 2, bidirectional=True)
        nn_init(self.attn_0, 'xavier')
        self.attn_1 = nn.Linear(2 * 2, 1)
        nn_init(self.attn_1, 'xavier')

        self.attn_2 = nn.LSTM(300, 50, bidirectional=True)
        nn_init(self.attn_2, 'xavier')
        self.attn_22 = nn.LSTM(300, 50, bidirectional=True)
        nn_init(self.attn_22, 'xavier')

        self.c = None

    def forward(self, x, x_len, x_mask, x_tag=None, testing=False):
        self.a, self.b = (self.attn_2(self.domain_embedding(x)))
        self.aa, self.bb = (self.attn_22(self.gen_embedding(x)))

        # aaa = torch.cat((self.a, self.aa), dim=2)

        # domain_embedding = self.linear_ae2(self.domain_embedding(x))
        domain_embedding = torch.cat((self.domain_embedding(x), self.a), dim=2)

        gen_embedding = torch.cat((self.gen_embedding(x), self.aa), dim=2)

        projected_cat = torch.cat(
            (domain_embedding.unsqueeze(2), gen_embedding.unsqueeze(2)), dim=2)
        # projected_cat = torch.cat(( self.gen_embedding(x), domain_embedding), dim=2)

        s_len, b_size, _, emb_dim = projected_cat.size()
        attn_input = projected_cat
        attn_input = attn_input.view(s_len, b_size * 2, -1)
        self.m_attn = self.attn_1((self.attn_0(attn_input)[0]))
        self.m_attn = self.m_attn.view(s_len, b_size, 2)
        self.m_attn = torch.sigmoid(self.m_attn)
        attended = projected_cat * self.m_attn.view(s_len, b_size, 2,
                                                    1).expand_as(projected_cat)
        # x_conv = attended.view(s_len, b_size, 900)
        x_conv = attended.sum(2)

        x_conv = self.dropout((x_conv)).transpose(1, 2)


        x_conv = torch.nn.functional.relu(torch.cat((self.conv1(x_conv), self.conv2(x_conv)), dim=1))

        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv3(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv4(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv5(x_conv))
        #x_conv = self.dropout(x_conv)
        #x_conv = torch.nn.functional.relu(self.conv6(x_conv))
        x_conv = x_conv.transpose(1, 2)
        x_logit = self.linear_ae(x_conv)

        if testing:
            if self.crf_flag:
                score = self.crf.viterbi_tags(x_logit, x_mask)
            else:
                x_logit = x_logit.transpose(2, 0)
                score = torch.nn.functional.log_softmax(x_logit).transpose(2, 0)
        else:
            if self.crf_flag:
                score = -self.crf(x_logit, x_tag, x_mask)
            else:
                x_logit = torch.nn.utils.rnn.pack_padded_sequence(x_logit, x_len, batch_first=True)
                score = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x_logit.data), x_tag.data)
        return score


def label_rest_xml(fn, output_fn, corpus, label):
    dom=ET.parse(fn)
    root=dom.getroot()
    pred_y=[]
    for zx, sent in enumerate(root.iter("sentence") ) :
        tokens=corpus[zx]
        lb=label[zx]
        opins=ET.Element("Opinions")
        token_idx, pt, tag_on=0, 0, False
        start, end=-1, -1
        for ix, c in enumerate(sent.find('text').text):
            if token_idx<len(tokens) and pt>=len(tokens[token_idx] ):
                pt=0
                token_idx+=1

            if token_idx<len(tokens) and lb[token_idx]==1 and pt==0 and c!=' ':
                if tag_on:
                    end=ix
                    tag_on=False
                    opin=ET.Element("Opinion")
                    opin.attrib['target']=sent.find('text').text[start:end]
                    opin.attrib['from']=str(start)
                    opin.attrib['to']=str(end)
                    opins.append(opin)
                start=ix
                tag_on=True
            elif token_idx<len(tokens) and lb[token_idx]==2 and pt==0 and c!=' ' and not tag_on:
                start=ix
                tag_on=True
            elif token_idx<len(tokens) and (lb[token_idx]==0 or lb[token_idx]==1) and tag_on and pt==0:
                end=ix
                tag_on=False 
                opin=ET.Element("Opinion")
                opin.attrib['target']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)
            elif token_idx>=len(tokens) and tag_on:
                end=ix
                tag_on=False 
                opin=ET.Element("Opinion")
                opin.attrib['target']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)
            if c==' ':
                pass
            elif tokens[token_idx][pt:pt+2]=='``' or tokens[token_idx][pt:pt+2]=="''":
                pt+=2
            else:
                pt+=1
        if tag_on:
            tag_on=False
            end=len(sent.find('text').text)
            opin=ET.Element("Opinion")
            opin.attrib['target']=sent.find('text').text[start:end]
            opin.attrib['from']=str(start)
            opin.attrib['to']=str(end)
            opins.append(opin)
        sent.append(opins )
    dom.write(output_fn)

def label_laptop_xml(fn, output_fn, corpus, label):
    dom=ET.parse(fn)
    root=dom.getroot()
    pred_y=[]
    for zx, sent in enumerate(root.iter("sentence") ) :
        tokens=corpus[zx]
        lb=label[zx]
        opins=ET.Element("aspectTerms")
        token_idx, pt, tag_on=0, 0, False
        start, end=-1, -1
        for ix, c in enumerate(sent.find('text').text):
            if token_idx<len(tokens) and pt>=len(tokens[token_idx] ):
                pt=0
                token_idx+=1

            if token_idx<len(tokens) and lb[token_idx]==1 and pt==0 and c!=' ':
                if tag_on:
                    end=ix
                    tag_on=False
                    opin=ET.Element("aspectTerm")
                    opin.attrib['term']=sent.find('text').text[start:end]
                    opin.attrib['from']=str(start)
                    opin.attrib['to']=str(end)
                    opins.append(opin)
                start=ix
                tag_on=True
            elif token_idx<len(tokens) and lb[token_idx]==2 and pt==0 and c!=' ' and not tag_on:
                start=ix
                tag_on=True
            elif token_idx<len(tokens) and (lb[token_idx]==0 or lb[token_idx]==1) and tag_on and pt==0:
                end=ix
                tag_on=False 
                opin=ET.Element("aspectTerm")
                opin.attrib['term']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)
            elif token_idx>=len(tokens) and tag_on:
                end=ix
                tag_on=False 
                opin=ET.Element("aspectTerm")
                opin.attrib['term']=sent.find('text').text[start:end]
                opin.attrib['from']=str(start)
                opin.attrib['to']=str(end)
                opins.append(opin)
            if c==' ' or ord(c)==160:
                pass
            elif tokens[token_idx][pt:pt+2]=='``' or tokens[token_idx][pt:pt+2]=="''":
                pt+=2
            else:
                pt+=1
        if tag_on:
            tag_on=False
            end=len(sent.find('text').text)
            opin=ET.Element("aspectTerm")
            opin.attrib['term']=sent.find('text').text[start:end]
            opin.attrib['from']=str(start)
            opin.attrib['to']=str(end)
            opins.append(opin)
        sent.append(opins )
    dom.write(output_fn)  
    

def test(model, test_X, raw_X, domain, command, template, batch_size=128, crf=False):
    pred_y=np.zeros((test_X.shape[0], 166), np.int16)
    model.eval()
    for offset in range(0, test_X.shape[0], batch_size):
        batch_test_X_len=np.sum(test_X[offset:offset+batch_size]!=0, axis=1)
        batch_idx=batch_test_X_len.argsort()[::-1]
        batch_test_X_len=batch_test_X_len[batch_idx]
        batch_test_X_mask=(test_X[offset:offset+batch_size]!=0)[batch_idx].astype(np.uint8)
        batch_test_X=test_X[offset:offset+batch_size][batch_idx]
        batch_test_X_mask=torch.autograd.Variable(torch.from_numpy(batch_test_X_mask).long().cuda() )
        batch_test_X = torch.autograd.Variable(torch.from_numpy(batch_test_X).long().cuda() )
        batch_pred_y=model(batch_test_X, batch_test_X_len, batch_test_X_mask, testing=True)
        r_idx=batch_idx.argsort()
        if crf:
            batch_pred_y=[batch_pred_y[idx] for idx in r_idx]
            for ix in range(len(batch_pred_y) ):
                for jx in range(len(batch_pred_y[ix]) ):
                    pred_y[offset+ix,jx]=batch_pred_y[ix][jx]
        else:
            batch_pred_y=batch_pred_y.data.cpu().numpy().argmax(axis=2)[r_idx]
            pred_y[offset:offset+batch_size,:batch_pred_y.shape[1]]=batch_pred_y
    model.train()
    assert len(pred_y)==len(test_X)
    
    command=command.split()
    if domain=='restaurant':
        label_rest_xml(template, command[6], raw_X, pred_y)
        acc=check_output(command ).split()
        print(acc)
        return float(acc[9][10:])
    elif domain=='laptop':
        label_laptop_xml(template, command[4], raw_X, pred_y)
        acc=check_output(command ).split()
        print(acc)
        return float(acc[15])

def evaluate(runs, data_dir, model_dir, domain, command, template):
    ae_data=np.load(data_dir+domain+".npz")
    with open(data_dir+domain+"_raw_test.json") as f:
        raw_X=json.load(f)
    results=[]
    for r in range(runs):
        model=torch.load(model_dir+domain+str(r) )
        result=test(model, ae_data['test_X'], raw_X, domain, command, template, crf=False)
        results.append(result)
    print(sum(results)/len(results) )

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--data_dir', type=str, default="../data/prep_data/")
    #parser.add_argument('--model_dir', type=str, default="../model2/")
    #parser.add_argument('--domain', type=str, default="restaurant")
    parser.add_argument('--model_dir', type=str, default="../Decnn_res_model/")
    parser.add_argument('--domain', type=str, default="restaurant")
    args = parser.parse_args()

    if args.domain=='restaurant':
        command="java -cp ./A.jar absa16.Do Eval -prd ../data/official_data/Decnn_res_pred_seed1337_.xml -gld ../data/official_data/EN_REST_SB1_TEST.xml.gold -evs 2 -phs A -sbt SB1"
        template="../data/official_data/EN_REST_SB1_TEST.xml.A"
    elif args.domain=='laptop':
        command="java -cp ./eval.jar Main.Aspects ../data/official_data/pred.xml ../data/official_data/Laptops_Test_Gold.xml"
        template="../data/official_data/Laptops_Test_Data_PhaseA.xml"

    evaluate(args.runs, args.data_dir, args.model_dir, args.domain, command, template)























