import sys, os, os.path as op
import cv2
import numpy as np
import pickle
import argparse
import logging
import sqlite3
from progressbar import ProgressBar
from functools import partial

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from torch.utils.data import DataLoader
import cfgs.config as cfg

sys.path.insert(0, os.path.join(os.getenv('CITY_PATH'), 'src'))
from db.lib.helperImg import ReaderVideo
from db.lib.helperDb import createDb


''' Detect from a db and write to another db. '''


parser = argparse.ArgumentParser(description='PyTorch Yolo')
parser.add_argument('--in_db_path', required=True)
parser.add_argument('--out_db_path', required=True)
parser.add_argument('--image_size_index', type=int, default=0,
                    help='setting images size index 0:320, 1:352, 2:384, 3:416, 4:448, 5:480, 6:512, 7:544, 8:576')
parser.add_argument('--trained_model', required=True, help='full path *.h5 to trained weights.')
parser.add_argument('--logging', type=int, choices=[10,20,30,40])
parser.add_argument('--use_maps', action='store_true')
args = parser.parse_args()

logging.basicConfig(level=args.logging, format='%(levelname)s: %(message)s')


# hyper-parameters
# ------------
max_per_image = 300
thresh = 0.01
vis = False
# ------------


def db_detect(net, dataset, dataloader, writer, max_per_image=300, thresh=0.5, vis=False):
    num_images = len(dataset)
    num_classes = dataset.num_classes

    for batch in ProgressBar()(dataloader):
        assert len(batch['origin_im']) == 1, 'Batch size must be 1.'

        ori_im = batch['origin_im'][0]
        im_data = net_utils.np_to_variable(batch['images'], is_cuda=True,
                                           volatile=True).permute(0, 3, 1, 2)

        bbox_pred, iou_pred, prob_pred = net(im_data)

        # to numpy
        bbox_pred = bbox_pred.data.cpu().numpy()
        iou_pred = iou_pred.data.cpu().numpy()
        prob_pred = prob_pred.data.cpu().numpy()

        bboxes, scores, cls_inds = yolo_utils.postprocess(
            bbox_pred, iou_pred, prob_pred, ori_im.shape, num_classes, 
            cfg, thresh, args.image_size_index)

        # Limit to max_per_image detections *over all classes*
        if len(scores) > max_per_image:
            thresh = np.sort(scores)[-max_per_image]
            keep = np.where(scores >= thresh)[0]
            bboxes = bboxes[keep]
            scores = scores[keep]
            cls_inds = cls_inds[keep]

        #im2show = yolo_utils.draw_detection(ori_im, bboxes, scores, cls_inds, cfg, thr=0.0)

        # Add imagefile and the cars to the database.
        c_out.execute('INSERT INTO images(imagefile) VALUES (?)', (batch['imagefile'][0],))
        for bbox, score, cls_ind in zip(bboxes, scores, cls_inds):
            s = 'imagefile,name,score,x1,y1,width,height'
            v = (batch['imagefile'][0], dataset.classes[cls_ind], float(score), 
                bbox[0], bbox[1], bbox[2]-bbox[0], bbox[3]-bbox[1])
            logging.debug('db_detect: %s = %s' % (s, str(v)))
            c_out.execute('INSERT INTO cars(%s) VALUES (?,?,?,?,?,?,?)' % s, v)


def define_dataset(dataset_type):
    if dataset_type == 'pascal_voc':
        from datasets.pascal_voc import VOCDataset
        logging.info ('Loading pascal_voc dataset')
        return VOCDataset(cfg.imdb_test, cfg.DATA_DIR)
    elif dataset_type == 'citycam':
        from datasets.citycams import CityDataset
        logging.info ('Loading citycams dataset')
        return CityDataset(args.in_db_path, use_maps=args.use_maps)
    else:
        raise Exception('Wrong dataset_type.')


if __name__ == '__main__':
    # data loader
    dataset = define_dataset('citycam')
    collate_fn = partial(yolo_utils.collate_fn_test,
        inp_size=cfg.multi_scale_inp_size[args.image_size_index])
    dataloader = DataLoader(dataset, batch_size=1,
        shuffle=False, num_workers=2, drop_last=False, collate_fn=collate_fn)

    net = Darknet19(num_classes=dataset.num_classes, input_nc=(5 if args.use_maps else 3))
    logging.info('Loading model from %s' % args.trained_model)
    net_utils.load_net(args.trained_model, net)

    net.cuda()
    net.eval()

    # Create and open databas for output.
    if os.path.exists(args.out_db_path):
      os.remove(args.out_db_path)
    conn_out = sqlite3.connect(args.out_db_path)
    createDb(conn_out)
    c_out = conn_out.cursor()

    db_detect(net, dataset, dataloader, c_out, max_per_image, thresh, vis)

    conn_out.commit()
    conn_out.close()
    dataset.close()

