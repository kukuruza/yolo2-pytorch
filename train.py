import os
import torch
from time import time
import logging
import argparse
from functools import partial
from torch.utils.data import DataLoader
from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from utils.timer import Timer
import cfgs.config as cfg
from random import randint
from torch.utils.data import DataLoader
from torch.autograd import Variable

try:
    from pycrayon import CrayonClient
except ImportError:
    CrayonClient = None

parser = argparse.ArgumentParser(description='PyTorch Yolo')
parser.add_argument('--train_output_dir')
parser.add_argument('--dataset_type', default='citycam', choices=['pascal_voc', 'citycam'])
parser.add_argument('--db_path', help='for citycam dataset only')
parser.add_argument('--logging', type=int, default=20, choices=[10,20,30,40])
parser.add_argument('--epochs_per_saved_model', type=int, default=10)
parser.add_argument('--use_maps', action='store_true')
args = parser.parse_args()

logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')


def define_dataset(dataset_type):
    if dataset_type == 'pascal_voc':
        from datasets.pascal_voc import VOCDataset
        logging.info ('Loading pascal_voc dataset')
        return VOCDataset(cfg.imdb_train, cfg.DATA_DIR)
    elif dataset_type == 'citycam':
        from datasets.citycams import CityDataset
        logging.info ('Loading citycams dataset')
        return CityDataset(args.db_path)
    else:
        raise Exception('Wrong dataset_type.')

# data loader
dataset = define_dataset(args.dataset_type, (5 if args.use_maps else 3))
collate_fn = partial(yolo_utils.collate_fn_train,
    multi_scale_inp_size=cfg.multi_scale_inp_size)
dataloader = DataLoader(dataset, batch_size=cfg.train_batch_size,
    shuffle=True, num_workers=2, collate_fn=collate_fn)
print('load data succ...')

net = Darknet19(num_classes=dataset.num_classes, input_nc=(5 if args.use_maps else 3))
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
train_loss, bbox_loss, iou_loss, cls_loss = 0., 0., 0., 0.
for epoch in range(start_epoch, cfg.max_epoch):
  for step, batch in enumerate(dataloader):
    start = time()

    im = batch['images']
    gt_boxes = batch['gt_boxes']
    gt_classes = batch['gt_classes']
    dontcare = batch['dontcare']
    orgin_im = batch['origin_im']
    size_index = batch['size_index']

    im_data = net_utils.np_to_variable(
        im, is_cuda=True, volatile=False).permute(0, 3, 1, 2)
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

    duration = time() - start
    if step % cfg.disp_interval == 0:
        train_loss /= cfg.disp_interval
        bbox_loss /= cfg.disp_interval
        iou_loss /= cfg.disp_interval
        cls_loss /= cfg.disp_interval
        print('epoch %d[%d/%d], loss: %.3f, bbox_loss: %.3f, iou_loss: %.3f, '
              'cls_loss: %.3f (%.2f s/batch)' %
               (epoch, step, batch_per_epoch, train_loss, bbox_loss,
                iou_loss, cls_loss, duration))

        if use_tensorboard and (step % cfg.log_interval == 0):
            absstep = step + epoch * len(dataset)
            exp.add_scalar_value('loss_train', train_loss, step=absstep)
            exp.add_scalar_value('loss_bbox', bbox_loss, step=absstep)
            exp.add_scalar_value('loss_iou', iou_loss, step=absstep)
            exp.add_scalar_value('loss_cls', cls_loss, step=absstep)
            exp.add_scalar_value('learning_rate', lr, step=absstep)

        train_loss, bbox_loss, iou_loss, cls_loss = 0., 0., 0., 0.

    # End of epoch loop.

  if epoch in cfg.lr_decay_epochs:
      lr *= cfg.lr_decay
      optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                  momentum=cfg.momentum,
                                  weight_decay=cfg.weight_decay)

  if epoch % args.epochs_per_saved_model == 0:
    train_output_dir = args.train_output_dir if args.train_output_dir is not None else cfg.train_output_dir
    train_output_path = os.path.join(train_output_dir, 'epoch_%d.h5' % epoch)
    logging.info('Saving model to %s' % train_output_path)
    net_utils.save_net(train_output_path, net)

