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

#aa = sum(accuracy_trials)/10
#all_sm = []
#for i in np.arange(0,100,10):
#    a1 = accuracy_trials[i:i+10]
#    all_sm.append(a1.sum())
#for i in accuracy_trials:
#    
#
##accuracy_trials1 = accuracy_trials[-2:]
##accuracy_trials = [i.cpu().numpy()  for i in accuracy_trials]
#accuracy_trials = np.array(accuracy_trials)
#means = accuracy_trials.mean(0)
#stds = accuracy_trials.std(0)
##elta_time_mean = np.array(delta_time_trials).mean()
#print('{:.4f}    {:.4f}'.format(means, stds))







#0.83360004
#0.82236844
#0.8160919
#0.8427773
#0.8318441
#0.82009727
#0.81345075
#0.81244874
#0.84339315
#0.81481487






#0.83232623
#0.8493566
#0.83738595
#0.8392993
#0.843869
#0.84250766
#0.84410644
#0.86436784
#0.8534743
#0.8545176

#res 1606.377939







#res
#1622.2480509281158
#1635.7521178722382
#1604.7220153808594
#1601.5641129016876
#1611.984554052353
#1616.7399544715881
#1612.5075345039368
#1614.962651014328
#1632.944396495819
#1622.9215955734253
#
#0.8365696
#0.8080972
#0.812398
#0.8246753
#0.81615824
#0.82601625
#0.8154093
#0.81954294
#0.825
#0.8119935








#
#laptop
#0.8575758
#0.8411498
#0.8534743
#0.84122133
#0.8547271
#0.8512585
#0.8542945
#0.8392037
#0.8462709
#0.8470407
#
#789.0314013957977
#793.3007316589355
#795.0659377574921
#793.2349283695221
#794.4922688007355
#794.9564416408539
#791.9163646697998
#793.0626027584076
#793.9386074542999
#791.5195679664612

#0.8377761
#0.8407351
#0.8477429
#0.8482549
#0.838514
#0.8545455
#0.851096
#0.84639025
#0.8480243
#0.8556232
#0.84609556
#0.84170467
#0.86081696
#0.8560548
#0.84927315
#0.8420257
#0.85087055
#0.8391502
#0.8644966
#0.837963
#0.8303031
#0.847561
#0.84820753
#0.8503817
#0.853211
#0.85432476
#0.85216075
#0.8454545
#0.8545176
#0.8401826
#
#49940
#76682
#39641
#82893
#65138
#36817
#30421
#93630
#3267
#92483
#81055
#50547
#91893
#325
#72187
#9260
#16194
#84826
#75864
#46443
#17541
#97377
#42432
#54165
#78874
#24869
#4378
#34051
#59636
#46075
  
  
#  0.85323197
#0.8508033
#0.84417546
#0.84696966
#0.83978736
#0.86798185
#0.83667177
#0.8300455
#0.843869
#0.84379786
#0.84662575
#0.8514851
#0.85323197
#0.8519362
#0.8491281
#0.8495034
#0.83841467
#0.84650457
#0.84226644
#0.8525835
#0.84370255
#0.84306294
#0.85019004
#0.8386606
#0.84580153
#0.8424242
#0.8422642
#0.84714836
#
#
#325
#72187
#9260
#16194
#84826
#75864
#46443
#17541
#97377
#42432
#54165
#78874
#24869
#4378
#34051
#59636
#46075
#47619
#68342
#47837
#63248
#29123
#76564
#20253
#42069
#11781
#16353
#84541
#24201
  
  
  
  
  
  
  
  
  
  
#  74038
#3466
#49905
#29193
#29431
#35465
#13932
#78632
#68649
#93227
#45755
#98442
#77173
#440
#16285
#8560
#13946
#23284
#93376
#49173
#5741
#53991
#41673
#23832
#69713
#61165
#2551
#76007
#14938
#1397
#  
  
  
  
  
  
#  
#  0.8163592
#0.8136143
#0.82457554
#0.8220065
#0.8253205
#0.8094079
#0.81244874
#0.81685567
#0.8038898
#0.81723857
#0.80933225
#0.81706315
#0.8091728
#0.81736284
#0.8122482
#0.8166534
#0.8239095
#0.82247555
#0.8253452
#0.82198524
#0.81450945
#0.830033
#0.8188586
#0.8162602
#0.83293366
#0.82134193
#0.8287113
#0.8136696
#0.8113821
#0.8073249