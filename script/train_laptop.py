import argparse
import torch
import time
import json
import numpy as np
import math
import random
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
# np.random.seed(2019)
# random.seed(2019)
# torch.manual_seed(2019)
# torch.cuda.manual_seed(2019)

#np.random.seed(1337)
#random.seed(1337)
#torch.manual_seed(1337)
#torch.cuda.manual_seed(1337)

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

def batch_generator(X, y, batch_size=128, return_idx=False, crf=False):
    for offset in range(0, X.shape[0], batch_size):
        batch_X_len = np.sum(X[offset:offset + batch_size] != 0, axis=1)
        batch_idx = batch_X_len.argsort()[::-1]
        batch_X_len = batch_X_len[batch_idx]
        batch_X_mask = (X[offset:offset + batch_size] != 0)[batch_idx].astype(np.uint8)
        batch_X = X[offset:offset + batch_size][batch_idx]
        batch_y = y[offset:offset + batch_size][batch_idx]
        batch_X = torch.autograd.Variable(torch.from_numpy(batch_X).long().cuda())
        batch_X_mask = torch.autograd.Variable(torch.from_numpy(batch_X_mask).long().cuda())
        batch_y = torch.autograd.Variable(torch.from_numpy(batch_y).long().cuda())
        if len(batch_y.size()) == 2 and not crf:
            batch_y = torch.nn.utils.rnn.pack_padded_sequence(batch_y, batch_X_len, batch_first=True)
        if return_idx:  # in testing, need to sort back.
            yield (batch_X, batch_y, batch_X_len, batch_X_mask, batch_idx)
        else:
            yield (batch_X, batch_y, batch_X_len, batch_X_mask)


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
        self.conv7 = torch.nn.Conv1d(256, 256, 3, padding=1)
        #self.linear_ae1 = torch.nn.Linear(256*2 , 256*2)
        #nn_init(self.linear_ae1, 'xavier')
        self.linear_ae1 = torch.nn.Linear(600, 256)
        self.linear_ae2 = torch.nn.Linear(300, 150)
        self.linear_ae = torch.nn.Linear(256 , num_classes)
        of = offset
        self.crf_flag = crf
        if self.crf_flag:
            from allennlp.modules import ConditionalRandomField
            self.crf = ConditionalRandomField(num_classes)

        self.attn_0 = nn.GRU(300+150, 2, bidirectional=True)
        nn_init(self.attn_0, 'kaiming')
        self.attn_1 = nn.Linear(2 * 2, 1)
        nn_init(self.attn_1, 'kaiming')

        self.attn_2 = nn.LSTM(300, 75, bidirectional=True)
        nn_init(self.attn_2, 'xavier')
        self.attn_22 = nn.LSTM(300, 75, bidirectional=True)
        nn_init(self.attn_22, 'xavier')

        self.c = None

    def forward(self, x, x_len, x_mask, x_tag=None, testing=False):
        self.a, self.b = (self.attn_2(self.domain_embedding(x)))
        self.aa, self.bb = (self.attn_22(self.gen_embedding(x)))

        #aaa = torch.cat((self.a, self.aa), dim=2)

        #domain_embedding = self.linear_ae2(self.domain_embedding(x))
        domain_embedding = torch.cat((self.domain_embedding(x),self.a), dim=2)

        gen_embedding = torch.cat(( self.gen_embedding(x),self.aa), dim=2)

        projected_cat = torch.cat((domain_embedding.unsqueeze(2), gen_embedding.unsqueeze(2)), dim=2)
        #projected_cat = torch.cat(( self.gen_embedding(x), domain_embedding), dim=2)
        
        s_len, b_size, _, emb_dim = projected_cat.size()
        attn_input = projected_cat
        attn_input = attn_input.view(s_len, b_size * 2, -1)
        self.m_attn = self.attn_1((self.attn_0(attn_input)[0]))
        self.m_attn = self.m_attn.view(s_len, b_size, 2)
        self.m_attn = torch.sigmoid(self.m_attn)
        attended = projected_cat * self.m_attn.view(s_len, b_size, 2, 1).expand_as(projected_cat)
         # x_conv = attended.view(s_len, b_size, 900)
        x_conv = attended.sum(2)
        '''
        projected_cat = torch.cat((aaa.unsqueeze(2),self.domain_embedding(x).unsqueeze(2), self.gen_embedding(x).unsqueeze(2)), dim=2)
        s_len, b_size, _, emb_dim = projected_cat.size()
        attn_input = projected_cat
        attn_input = attn_input.view(s_len, b_size * 3, -1)
        self.m_attn = self.attn_1((self.attn_0(attn_input)[0]))
        self.m_attn = self.m_attn.view(s_len, b_size, 3)
        self.m_attn = torch.sigmoid(self.m_attn)
        attended = projected_cat * self.m_attn.view(s_len, b_size, 3, 1).expand_as(projected_cat)
        # x_conv = attended.view(s_len, b_size, 600)
        x_conv = projected_cat.sum(2)
        '''
#        x_emb = torch.cat((self.domain_embedding(x),self.gen_embedding(x)), dim=2).transpose(1, 2)
        '''
        projected_cat = torch.cat((aaa, self.domain_embedding(x), self.gen_embedding(x)), dim=2)
        #s_len, b_size, _, emb_dim = projected_cat.size()
        #attn_input = projected_cat
        #attn_input = attn_input.view(s_len, b_size * 2, -1)
        #self.m_attn = self.attn_1((self.attn_0(attn_input)[0]))
        #self.m_attn = self.m_attn.view(s_len, b_size,2)
        #self.m_attn = torch.sigmoid(self.m_attn)
        #attended = projected_cat * self.m_attn.view(s_len, b_size, 2, 1).expand_as(projected_cat)
        # x_conv = attended.view(s_len, b_size, 600)
        x_conv = projected_cat#.sum(2) 
        '''
        x_conv = self.dropout((x_conv)).transpose(1, 2)
        #x_conv1 = self.linear_ae1(x_conv)
        #x_conv1 = self.dropout(x_conv1)
        #x_logit = self.linear_ae(x_conv1)
        #x_conv = x_conv.transpose(1, 2)
        x_conv = torch.nn.functional.relu(torch.cat((self.conv1(x_conv), self.conv2(x_conv)), dim=1))

        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv3(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv4(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv5(x_conv))
        x_conv = self.dropout(x_conv)
        x_conv = torch.nn.functional.relu(self.conv6(x_conv))
        #x_conv = self.dropout(x_conv)
        #x_conv = torch.nn.functional.relu(self.conv7(x_conv))
        x_conv = x_conv.transpose(1, 2)
        x_logit = self.linear_ae(x_conv)



        if testing:
            if self.crf_flag:
                score = self.crf.viterbi_tags(x_logit, x_mask)
            else:
                x_logit = x_logit.transpose(2, 0)
                score = torch.nn.functional.log_softmax(x_logit).transpose(2, 0)
                # score1 = score.numpy()
                # score1 = pd.DataFrame(score1)
                # score1.to_csv('aa.csv')
        else:
            if self.crf_flag:
                score = -self.crf(x_logit, x_tag, x_mask)
            else:
                x_logit = torch.nn.utils.rnn.pack_padded_sequence(x_logit, x_len, batch_first=True)
                score = torch.nn.functional.nll_loss(torch.nn.functional.log_softmax(x_logit.data), x_tag.data)
        #if offset == 256:
        #    print((x_logit.data))

        #print((x_logit.data.size()))
        #score1 = x_logit.data.cpu().numpy()
        #score1 = pd.DataFrame(x_logit.data)
        #score1.to_csv('aa.csv')
        return score

def valid_loss(model, valid_X, valid_y, crf=False):
    model.eval()
    losses = []
    for batch in batch_generator(valid_X, valid_y, crf=crf):
        batch_valid_X, batch_valid_y, batch_valid_X_len, batch_valid_X_mask = batch
        loss = model(batch_valid_X, batch_valid_X_len, batch_valid_X_mask, batch_valid_y)
        losses.append(loss.data)
    model.train()
    return sum(losses) / len(losses)


def train(train_X, train_y, valid_X, valid_y, model, model_fn, optimizer, parameters, epochs=200, batch_size=128,
          crf=False):
    best_loss = float("inf")
    valid_history = []
    train_history = []
    for epoch in range(epochs):
        for batch in batch_generator(train_X, train_y, batch_size, crf=crf):
            batch_train_X, batch_train_y, batch_train_X_len, batch_train_X_mask = batch
            loss = model(batch_train_X, batch_train_X_len, batch_train_X_mask, batch_train_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(parameters, 1.)
            optimizer.step()
        loss = valid_loss(model, train_X, train_y, crf=crf)
        train_history.append(loss)
        loss = valid_loss(model, valid_X, valid_y, crf=crf)
        valid_history.append(loss)
        if loss < best_loss:
            best_loss = loss
            torch.save(model, model_fn)
        shuffle_idx = np.random.permutation(len(train_X))
        train_X = train_X[shuffle_idx]
        train_y = train_y[shuffle_idx]
    model = torch.load(model_fn)
    return train_history, valid_history


def run(domain, data_dir, model_dir, valid_split, runs, epochs, lr, dropout, batch_size=128):
    gen_emb = np.load(data_dir + "gen.vec.npy")
    domain_emb = np.load(data_dir + domain + "_emb.vec.npy")
    ae_data = np.load(data_dir + domain + ".npz")

    valid_X = ae_data['train_X'][-valid_split:]
    valid_y = ae_data['train_y'][-valid_split:]
    train_X = ae_data['train_X'][:-valid_split]
    train_y = ae_data['train_y'][:-valid_split]

    for r in range(runs):
        print(r)
        model = Model(gen_emb, domain_emb, 3, dropout=dropout, crf=False)
        model.cuda()
        parameters = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(parameters, lr=lr)
        train_history, valid_history = train(train_X, train_y, valid_X, valid_y, model, model_dir + domain + str(r),
                                             optimizer, parameters, epochs, crf=False)


#733.230454/ 1227.933388
if __name__ == "__main__":

  for jj in range(30):#[41]: #np.arange(41,100,3):

    i = random.randint(-1, 99999)
    start_time = time.time()  # 记录程序开始运行时间    
    model_name = "../Decnn_laptop_model111_" + str(i)
    
    if not os.path.exists(model_name):
        os.makedirs(model_name)


    np.random.seed(i)
    random.seed(i)
    torch.manual_seed(i)
    torch.cuda.manual_seed(i)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=model_name + '/')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=205)
    parser.add_argument('--runs', type=int, default=4)
    parser.add_argument('--domain', type=str, default="laptop")
    parser.add_argument('--data_dir', type=str, default="../data/prep_data/")
    parser.add_argument('--valid', type=int, default=150)  # number of validation data.
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--dropout', type=float, default=0.55)

    args = parser.parse_args()

    run(args.domain, args.data_dir, args.model_dir, args.valid, args.runs, args.epochs, args.lr, args.dropout,args.batch_size)

    end_time = time.time()
    print('Took %f second' % (end_time - start_time))