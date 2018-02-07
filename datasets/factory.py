from .pascal_voc import VOCDataset

def define_dataset(dataset_type, *args):
    '''
    The factory allows to easily switch between different datasets.
    '''
    if dataset_type == 'pascal_voc':
        return VOCDataset(*args)
    else:
        raise Exception('Wrong dataset_type.')

