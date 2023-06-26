"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
import torch.nn.functional as F
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from models import compile_model
#from models_2T import compile_model #swapping out to see the model with 2 transformers - just to try understand how sizes work - how many channels I will need when concatenating
from data import compile_data
from tools import SimpleLoss, get_batch_iou, get_val_info,points_to_voxel_loop

#seems like this is usually set as part of the config file - then when running the actual train file it is
#run by firing the main.py file which runs the train.py file and we specify the arguments to pass and point to the config file where the train label is set
#so it never actually gets explicitly passed in the train.py file where we took this code block from - so I think that's why we need to set it explicitly here
train_label = 'drivable_area'
#num_classes should be passed as part of cfg_pp - which is set up later after all explicit arguments are passed

#Want to set the logdir here - this is what will actually be passed when we call the function below and pass logdir-logdir
#but in the function definition set the default to just be "./runs"
logdir='./runs'

def train(
  train_label, #In TFGrid this isn't in the definition of train but really it is needed for compile_data that is called as part of it
               #so it should be passed - in reality it seems the Fire part of main.py that calls the config file gets it from there - but it should be here
  version = "mini",
  dataroot=os.environ["NUSCENES"],
  #NB JUST FOR EXPERIMENTATION SETTING THIS TO 5 - THE DEFAULT VALUE USED IN TFGRID IS 10000
  nepochs=7,
  gpuid = 0, #since only one GPU is available and we start counting from 0
  H=900,
  W=1600,
  resize_lim=(0.193, 0.225),
  final_dim=(128, 352),
  bot_pct_lim=(0.0, 0.22),
  rot_lim=(-5.4, 5.4),
  rand_flip=True,
  ncams=5,
  max_grad_norm=5.0,
  pos_weight=2.13,
  logdir='./runs', #let this be the default in the definition
  xbound=[-50.0, 50.0, 0.5],
  ybound=[-50.0, 50.0, 0.5],
  zbound=[-10.0, 10.0, 20.0],
  dbound=[4.0, 45.0,1.0],
  #NB - Need this new encoder_mode attribute to set which type of resnet blocks are used in BevEncode
  #0 -> resnet18
  #1 -> resnet34
  #2 -> resnet50
  encoder_mode = 0,

  bsz=4,
  nworkers=10,
  lr=1e-3,
  weight_decay=1e-7,
  num_classes = 1, #NEED TO PUT THIS IN HERE AS IT ISN'T COMING FROM THE CONFIG FILE
  num_points = 100000,
  #NB - AS WE ARE DOING A TRIAL BASED ON MINI - PASS 34718 WHEN CALLING TRAIN
  n_points=34718, ## trainval avg 34720  , mini avg 34718
  pc_range= [-50, -50, -4, 50, 50, 4], ##[xmin,ymin,zmin,xmax,ymax,zmax]
  #voxel_size = [1, 1, 8] #Not sure why in some instances this is commented and in others not - need to see exactly what the settings should be
  max_points_voxel = 100,
  max_voxels = 10000,
  input_features= 4,
  use_norm = False,
  vfe_filters = [64],
  with_distance = False,

  #below for tf_config - was missing in surce code to actually define tf_config to pass it to compile_model?
  #(Shouldn't need it if I'm not doing transfoermer anyway - just to get code runnig for now)
  n_head = 4,
  block_exp = 4,
  n_layer = 8,
  vert_anchors = 8,
  horz_anchors = 8,
  seq_len = 1, # input timesteps
  embd_pdrop = 0.1,
  resid_pdrop = 0.1,
  attn_pdrop = 0.1,
  n_views=1):

  grid_conf = {
    'xbound': xbound,
    'ybound': ybound,
    'zbound': zbound,
    'dbound': dbound,
    'encoder_mode':encoder_mode #new parameter for encoder_mode to be passed as part of grid_conf
  }

  data_aug_conf = {
    'resize_lim': resize_lim,
    'final_dim': final_dim,
    'rot_lim': rot_lim,
    'H': H, 'W': W,
    'rand_flip': rand_flip,
    'bot_pct_lim': bot_pct_lim,
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT','CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': ncams,
  }

  cfg_pp = {
    'num_classes':num_classes,
    'num_points': num_points,
    'pc_range': pc_range,
    'voxel_size' : [xbound[2],ybound[2],8],
    'max_points_voxel' : max_points_voxel,
    'max_voxels': max_voxels,
    'input_features': input_features,
    'batch_size': bsz,
    'use_norm': use_norm,
    'vfe_filters': vfe_filters,
    'with_distance': with_distance,
    'n_points': n_points,
  }
  #DO NOT NEED TF_CONFIG WHEN I AM DOING JUST CONCATENATION - THIS WAS ONLY THERE WHEN TRYING TO SEE HOW TRANSFUSER PART WORKED - JUST TO GET IT RUNNING
  # tf_config = {
  #   'n_head':n_head,
  #   'block_exp':block_exp,
  #   'n_layer':n_layer,
  #   'vert_anchors':vert_anchors,
  #   'horz_anchors':horz_anchors,
  #   'seq_len':seq_len,
  #   'embd_pdrop':embd_pdrop,
  #   'resid_pdrop':resid_pdrop,
  #   'attn_pdrop':attn_pdrop,
  #   'n_views':n_views,

  # }
  # print(tf_config)
  print('Loading data ...')
  #Had to change the way this is called slightly - it is strange, in train.py in TFGrid - it passes n_points and pc_range explicitly but this isn't how it is defined in models.py
  trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
                                          grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
                                          parser_name='segmentationdata',cfg_pp=cfg_pp, train_label=train_label) #other paraemeter - cond, dist, rank, sw can be left as default

  device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')
  print('Compiling model ...')
  model = compile_model(grid_conf, data_aug_conf, outC=1, cfg_pp=cfg_pp) ### outC number of output channels
  model.to(device)
  opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
  loss_fn = SimpleLoss(pos_weight).cuda(gpuid)
  writer = SummaryWriter(logdir=logdir)
  val_step = 500 if version == 'mini' else 5000
  model.train()
  counter = 0
  print('Starting training ...')
  for epoch in range(nepochs):
      print(f"Epoch: {epoch+1}/{nepochs}") #NOTE - ACTUAL VALUE OF epoch WILL START COUNTING FROM 0 - THIS IS JUST FOR CONVENIENCE OF READING OUTPUT TO START FROM 1
      np.random.seed()
      for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs, points) in enumerate(trainloader):
          t0 = time()
          opt.zero_grad()
          voxels, coors, num_points = points_to_voxel_loop(points, cfg_pp)
          preds = model(imgs.to(device),
                  rots.to(device),
                  trans.to(device),
                  intrins.to(device),
                  post_rots.to(device),
                  post_trans.to(device),
                  voxels.to(device),
                  coors.to(device),
                  num_points.to(device),
                  )

          binimgs = binimgs.to(device)
          loss = loss_fn(preds, binimgs)
          loss.backward()
          torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
          opt.step()
          counter += 1
          t1 = time()
          if counter % 10 == 0:
              print(counter, loss.item())
              writer.add_scalar('train/loss', loss, counter)
          if counter % 50 == 0:
              intersection, union, iou = get_batch_iou(preds, binimgs)
              writer.add_scalar('train/iou', iou, counter)
              writer.add_scalar('train/epoch', epoch, counter)
              writer.add_scalar('train/step_time', t1 - t0, counter)
          if counter % val_step == 0:
              val_info = get_val_info(model, valloader, loss_fn, device, cfg=cfg_pp,use_tqdm= True)
              print('VAL', val_info)
              writer.add_scalar('val/loss', val_info['loss'], counter)
              writer.add_scalar('val/iou', val_info['iou'], counter)
          if counter % val_step == 0:
              model.eval()
              mname = os.path.join(logdir, "model{}.pt".format(counter)) #NB This is the location-same runs/ directory where summary will be saved
                                                                         #where the model will also be saved every val_step=500 number of training steps
              print('saving ... ', mname)
              torch.save({
              'epoch': epoch,
              'model_state_dict': model.state_dict(),
              'optimizer_state_dict': opt.state_dict(),
              'loss': loss,
              'counter':counter
              }, mname)
              model.train()

