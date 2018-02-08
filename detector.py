import os
import numpy as np
import argparse
import logging
from time import time
from functools import partial
from pprint import pprint
from scipy.misc import imsave

from darknet import Darknet19
import utils.yolo as yolo_utils
import utils.network as net_utils
from torch.utils.data import DataLoader
import cfgs.config as cfg



class Detector:
  def __init__(self, trained_model, max_per_image, thresh, image_size_index, num_classes):
    self.max_per_image = max_per_image
    self.thresh = thresh
    self.image_size_index = image_size_index
    self.num_classes = num_classes

    self.net = Darknet19(num_classes=num_classes)
    logging.info('Loading model from %s' % trained_model)
    net_utils.load_net(trained_model, self.net)
    self.net.cuda()
    self.net.eval()
    
    
  def __call__(self, batch):
    ''' Detect in a batch.
    '''
    
    t_all = time()

    original_shape = batch['origin_im'][0].shape
    assert len(batch['origin_im']) == 1, 'Supports only one batch_size = 1'
    # for x in batch['origin_im']: assert x.shape == original_shape  # For now.

    im_data = net_utils.np_to_variable(
        batch['images'], is_cuda=True, volatile=True).permute(0, 3, 1, 2)

    t_imdetect = time()
    bbox_pred, iou_pred, prob_pred = self.net(im_data)
    t_imdetect = time() - t_imdetect

    bbox_pred = bbox_pred.data.cpu().numpy()
    iou_pred = iou_pred.data.cpu().numpy()
    prob_pred = prob_pred.data.cpu().numpy()

    bboxes, scores, cls_inds = yolo_utils.postprocess(
        bbox_pred, iou_pred, prob_pred, original_shape, self.num_classes,
        cfg, self.thresh, self.image_size_index)

    # Keep only first self.max_per_image detections.
    sortind = np.argsort(scores)[::-1]
    if len(sortind) >= self.max_per_image:
      sortind = sortind[:self.max_per_image]
    bboxes = bboxes[sortind]
    scores = scores[sortind]
    cls_inds = cls_inds[sortind]

    # Store the results as a list of ([x1, y1, x2, y2], score, class_id)
    detections = [(bbox.tolist(), score, cls_ind) for bbox, score, cls_ind in
            zip(bboxes, scores, cls_inds)]
    
    t_all = time() - t_all
    logging.info('Detector: done in %.1f sec, including imdedect: %.1f' % (t_all, t_imdetect))
    return detections


if __name__ == '__main__':

    def define_dataset(dataset_type):
        if dataset_type == 'pascal_voc':
            from datasets.pascal_voc import VOCDataset
            logging.info ('Loading pascal_voc dataset')
            return VOCDataset(cfg.imdb_test, cfg.DATA_DIR)
        elif dataset_type == 'citycam':
            from datasets.citycams import CityDataset
            logging.info ('Loading citycams dataset')
            return CityDataset(args.db_path)
        else:
            raise Exception('Wrong dataset_type.')


    logging.basicConfig(level=20, format='%(levelname)s: %(message)s')


    parser = argparse.ArgumentParser(description='PyTorch Yolo')
    parser.add_argument('--dataset_type', default='citycam', choices=['pascal_voc', 'citycam'])
    parser.add_argument('--db_path', help='for citycam dataset only')
    parser.add_argument('--vis_path', help='if specified will write visualizations there.')
    parser.add_argument('--trained_model', required=True,
                        help='overload path to the trained model.')
    parser.add_argument('--image_size_index', type=int, default=0,
                        help='setting images size index 0:320, 1:352, 2:384, 3:416, 4:448, 5:480, 6:512, 7:544, 8:576')
    parser.add_argument('--max_per_image', type=int, default=300)
    parser.add_argument('--thresh', type=float, default=0.01)
    parser.add_argument('--vis', action='store_true')
    args = parser.parse_args()

    # data loader
    dataset = define_dataset(args.dataset_type)
    collate_fn = partial(yolo_utils.collate_fn_test,
        inp_size=cfg.multi_scale_inp_size[args.image_size_index])
    dataloader = DataLoader(dataset, batch_size=1,
        shuffle=True, num_workers=2, drop_last=False, collate_fn=collate_fn)

    detector = Detector(args.trained_model, max_per_image=args.max_per_image,
                        thresh=args.thresh, image_size_index=args.image_size_index,
                        num_classes=dataset.num_classes)

    for batch in dataloader:
      results = detector(batch)
      pprint(results)

      if args.vis_path is not None:
        assert len(batch['origin_im']) == 1, 'Supports only batch_size == 1'
        bboxes, scores, cls_inds = zip(*results)
        im2show = yolo_utils.draw_detection(
            batch['origin_im'][0], bboxes, scores, cls_inds,
            cfg.colors, dataset.classes, thr=0.5)
        print im2show.shape
        logging.info('Saving the visualzation to %s' % args.vis_path)
        imsave(args.vis_path, im2show)

      break

