import os
import torch
import datetime
import logging
from functools import partial

from darknet import Darknet19

from datasets.pascal_voc import VOCDataset
from torch.utils.data import DataLoader
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
from random import randint

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None


logging.basicConfig(level=20, format='%(levelname)s: %(message)s')


# data loader
dataset = VOCDataset(cfg.imdb_train, cfg.DATA_DIR)
collate_fn = partial(yolo_utils.collate_fn_train,
    multi_scale_inp_size=cfg.multi_scale_inp_size)
dataloader = DataLoader(dataset, batch_size=cfg.train_batch_size,
    shuffle=True, num_workers=2, collate_fn=collate_fn)
# dst_size=cfg.inp_size)
print('load data succ...')

net = Darknet19()
# net_utils.load_net(cfg.trained_model, net)
# pretrained_model = os.path.join(cfg.train_output_dir,
#     'darknet19_voc07trainval_exp1_63.h5')
# pretrained_model = cfg.trained_model
# net_utils.load_net(pretrained_model, net)
net.load_from_npz(cfg.pretrained_model, num_conv=18)
net.cuda()
net.train()
print('load net succ...')

# optimizer
start_epoch = 0
lr = cfg.init_learning_rate
optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=cfg.momentum,
                            weight_decay=cfg.weight_decay)

# tensorboad
use_tensorboard = cfg.use_tensorboard and CrayonClient is not None
# use_tensorboard = False
remove_all_log = False
if use_tensorboard:
    cc = CrayonClient(hostname='127.0.0.1')
    if remove_all_log:
        print('remove all experiments')
        cc.remove_all_experiments()
    if start_epoch == 0:
        try:
            cc.remove_experiment(cfg.exp_name)
        except ValueError:
            pass
        exp = cc.create_experiment(cfg.exp_name)
    else:
        exp = cc.open_experiment(cfg.exp_name)

batch_per_epoch = len(dataset) // cfg.train_batch_size
train_loss = 0
bbox_loss, iou_loss, cls_loss = 0., 0., 0.
cnt = 0
t = Timer()
step_cnt = 0
step = 0  # Step is NOT reset at each epoch.
for epoch in range(start_epoch, cfg.max_epoch):
  for batch in dataloader:
    step += 1

    t.tic()
    im = batch['images']
    gt_boxes = batch['gt_boxes']
    gt_classes = batch['gt_classes']
    dontcare = batch['dontcare']
    orgin_im = batch['origin_im']
    size_index = batch['size_index']

    # forward
    im_data = net_utils.np_to_variable(im,
                                       is_cuda=True,
                                       volatile=False).permute(0, 3, 1, 2)
    net(im_data, gt_boxes, gt_classes, dontcare, size_index)

    # backward
    loss = net.loss
    bbox_loss += net.bbox_loss.data.cpu().numpy()[0]
    iou_loss += net.iou_loss.data.cpu().numpy()[0]
    cls_loss += net.cls_loss.data.cpu().numpy()[0]
    train_loss += loss.data.cpu().numpy()[0]
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    cnt += 1
    step_cnt += 1
    duration = t.toc()
    if step % cfg.disp_interval == 0:
        train_loss /= cnt
        bbox_loss /= cnt
        iou_loss /= cnt
        cls_loss /= cnt
        print(('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, '
               'cls_loss: %.3f (%.2f s/batch, rest:%s)' %
               (epoch, step_cnt, batch_per_epoch, train_loss, bbox_loss,
                iou_loss, cls_loss, duration,
                str(datetime.timedelta(seconds=int((batch_per_epoch - step_cnt) * duration))))))  # noqa

        if use_tensorboard and step % cfg.log_interval == 0:
            exp.add_scalar_value('loss_train', train_loss, step=step)
            exp.add_scalar_value('loss_bbox', bbox_loss, step=step)
            exp.add_scalar_value('loss_iou', iou_loss, step=step)
            exp.add_scalar_value('loss_cls', cls_loss, step=step)
            exp.add_scalar_value('learning_rate', lr, step=step)

        train_loss = 0
        bbox_loss, iou_loss, cls_loss = 0., 0., 0.
        cnt = 0
        t.clear()

    # End of epoch loop.

  if epoch in cfg.lr_decay_epochs:
      lr *= cfg.lr_decay
      optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                  momentum=cfg.momentum,
                                  weight_decay=cfg.weight_decay)

  save_name = os.path.join(cfg.train_output_dir,
                              '{}_{}.h5'.format(cfg.exp_name, epoch))
  net_utils.save_net(save_name, net)
  print(('save model: {}'.format(save_name)))
  step_cnt = 0

