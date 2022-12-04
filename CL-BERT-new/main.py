import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
import helper
from shutil import copyfile
from loader import Data
from trainer import XHTrainer
import json
import dgl

parser = argparse.ArgumentParser()
#parser.add_argument('--data_dir', type=str, default='./Laptops_flat/')
parser.add_argument('--data_dir', type=str, default='./Restaurants16_flat/')
parser.add_argument('--num_epoch', type=int, default=7, help='Number of total training epochs.')
parser.add_argument('--batch_size', type=int, default=1, help='Training batch size.')
parser.add_argument('--log_step', type=int, default=3500, help='Print log every k steps.')
parser.add_argument('--save_dir', type=str, default='./saved_models_res', help='Root dir for saving models.')
#parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--accu_step', default=1, type=int)
parser.add_argument('--config_file', default='./config1.json', type=str)
args = parser.parse_args()

# set random seed
#torch.manual_seed(args.seed)
#np.random.seed(args.seed)
#random.seed(args.seed)
#torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False
os.environ["OMP_NUM_THREADS"] = '1'
#dgl.random.seed(args.seed)


seed = []
n_trials = 10
accuracy_trials = []
times = []
for t in range(n_trials):
  
    
    
  start_time = time.time() 
  the_seed = random.randint(-1,99999)  
  torch.manual_seed(the_seed)
  np.random.seed(the_seed)
  random.seed(the_seed)
  torch.cuda.manual_seed(the_seed)
  dgl.random.seed(the_seed) 
  seed.append(the_seed)

  args.save_dir = args.save_dir+args.data_dir.split('/')[-1]
  config = json.load(open(args.config_file, 'r', encoding="utf-8"))
  model_save_dir = args.save_dir
  helper.print_arguments(args)
  helper.ensure_dir(model_save_dir, verbose=True)

  print("Loading data from {} with batch size {}...".format(args.data_dir, args.batch_size))
  train_batch = Data(args.data_dir + '/all_train.pkl', args.batch_size, True, config).data_iter
  eval_batch = Data(args.data_dir + '/eval.pkl', args.batch_size, True, config).data_iter 
  test_batch = Data(args.data_dir + '/test.pkl', args.batch_size, False, config).data_iter 
  trainer = XHTrainer(args, config)


  train_loss_history, eval_loss_history = [], []
  for epoch in range(1, args.num_epoch+1):
    train_loss, train_step = 0., 0
    for i, batch in enumerate(train_batch):
        loss = trainer.update(batch)
        train_loss += loss
        train_step += 1
        if train_step % args.log_step == 0:
            print("batch: {}/{}, train_loss: {}".format(i+1,len(train_batch), train_loss/train_step))

    # eval 
    print("Evaluating on eval set in epoch {}...".format(epoch))
    eval_loss, eval_step =0., 0
    for i, batch in enumerate(eval_batch):
        loss = trainer.evaluate(batch)
        eval_loss += loss
        eval_step += 1


    train_loss_history.append(train_loss/train_step)

    # save best model
    if epoch == 1 or eval_loss/eval_step < min(eval_loss_history):
        model_file = model_save_dir + '/best_model.pt'
        trainer.save(model_file)
        print("new best model saved at epoch {}: train_loss {:.6f}\t eval_loss {:.6f}"\
            .format(epoch, train_loss/train_step, eval_loss/eval_step))
    eval_loss_history.append(eval_loss/eval_step)


  print("Training ended with {} epochs.".format(epoch))
  bt_train_loss = min(train_loss_history)
  bt_eval_loss = min(eval_loss_history)
  print("best train_loss: {}, best eval_loss: {}".format(bt_train_loss, bt_eval_loss))

  f1_score = trainer.predict(test_batch)
  print(" testing f1: {}".format(f1_score))
  accuracy_trials.append(f1_score)
  
  end_time = time.time()
  print('Took %f second' % (end_time - start_time))
  
  a_time = end_time - start_time
  times.append(a_time)
