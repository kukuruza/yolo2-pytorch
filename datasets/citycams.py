import os, sys
sys.path.insert(0, os.path.join(os.getenv('CITY_PATH'), 'src'))
from db.lib.dbDataset import CityimagesDataset
import logging
import numpy as np


class CityDataset(CityimagesDataset):
  ''' Dataset for YOLO2.
  Note that images are not resized, that is done in collate_fn.
  Images are normalized in collate_fn.
  '''

  def __init__(self, db_path, use_maps):
    super(CityDataset, self).__init__(db_path, use_maps=use_maps, image_constraint=
        'imagefile IN (SELECT DISTINCT imagefile FROM cars)')
    self.num_classes = 1
    self.classes = ('car',)
  
  def __getitem__(self, index):
    sample = super(CityDataset, self).__getitem__(index)

    assert 'image' in sample

    # List of lists into an array, change the name.
    sample['gt_boxes'] = np.asarray(sample['bboxes'])

    # Bbox to roi.
    sample['gt_boxes'][:,2] = sample['gt_boxes'][:,0] + sample['gt_boxes'][:,2]
    sample['gt_boxes'][:,3] = sample['gt_boxes'][:,1] + sample['gt_boxes'][:,3]

    N = len(sample['bboxes'])
    logging.debug('CityDataset sample has %d boxes, imagefile %s' %
        (N, sample['imagefile']))
        
    sample['gt_classes'] = np.zeros(shape=(N,), dtype=int)
    sample['dontcare'] = []
    return sample

