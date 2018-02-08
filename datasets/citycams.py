import os, sys
sys.path.insert(0, os.path.join(os.getenv('CITY_PATH'), 'src'))
from db.lib.dbDataset import CityimagesDataset
import logging
import numpy as np


class CityDataset(CityimagesDataset):

  def __init__(self, db_path):
    CityimagesDataset.__init__(self, db_path, image_constraint=
        'imagefile IN (SELECT DISTINCT imagefile FROM cars)')
  
  def __getitem__(self, index):
    sample = super(CityDataset, self).__getitem__(index)

    # List of lists into an array, change the name.
    sample['gt_boxes'] = np.asarray(sample['bboxes'])

    N = len(sample['bboxes'])
    logging.debug('CityDataset sample has %d boxes' % N)
        
    sample['gt_classes'] = np.ones(shape=(N,), dtype=int)
    sample['dontcare'] = []
    return sample

