import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import numpy as np
import os
import yaml
from tqdm import tqdm
from typing import Optional
# from model.sra import SRACls, ResNetCls
from model.utils import get_logger, plot_classification, build_disrete_cmap, save_annotation_qupath
from torch.utils.data import DataLoader
from glob import glob
from scipy.special import softmax
from matplotlib import cm
from model.utils import get_logger
# from model.transform import get_supervised_train_augmentation, get_supervised_val_augmentation, TwoCropsTransform
# from model.sra import SRA
# from model.sra_trainer import SRATrainer
# from dataset.builder import build_dataset
from dataset.base import BaseRemapDataset, HistoDataset
from torch.utils.data import Dataset
from torch.utils.data import ConcatDataset
from typing import Tuple, Optional, Iterable
from sklearn.metrics import f1_score
from dataset.constants import (const_kather19)
from model.utils import accuracy_topk
from numpy import ndarray
# from torch.utils.tensorboard import SummaryWriter
import re
import torchvision.models as models
import torch.nn as nn

def metrics(cgt: ndarray, pred: ndarray, name_classes: Optional[Iterable[str]] = None):
        """
        Compute metrics (accuracy) for all classes

        Parameters
        ----------
        cgt: ndarray (, N)
            Ground truth of classes
        pred: ndarray (,N)
            Predicted classes
        name_classes: Iterable of string (, C)
            Classes names

        Returns
        -------
        metrics: dict
            Dictionary with name od the classes as entries and metric as values. "ALL" is used for the overall
            performance of the prediction.
        """
        # Compute accuracy over all classes
        results_F1 = {'ALL': f1_score(y_true=cgt, y_pred=pred, average='weighted')}
        # results_ACC = {'ALL': accuracy_topk(l, t, topk=(1,))[0].item() for (l, t) in zip(pred, cgt)}
        # Check if name of classes fed otherwise returns
        if name_classes is None:
            return results_F1
        # Accuracy over classes
        for i, name in enumerate(name_classes):
            results_F1[name] = f1_score(y_true=np.array(cgt) == i, y_pred=np.array(pred) == i, average='binary')
            # results_ACC[name] = accuracy_topk(l, t, topk=(1,))[0].item() for (l, t) in zip(pred, cgt)
        return results_F1


def build_dataset_cls(
        path: str, transform: object, remap: Optional[dict] =None, **kwargs
) -> Tuple[BaseRemapDataset, BaseRemapDataset, Iterable[str]]:
    Cls = HistoDataset

    # dataset_train = Cls(root=path, transform=transform_train, set='train', map=remap, **kwargs)
    dataset_cls = Cls(root=path, transform=transform, set='full', map=remap, **kwargs)

    return  dataset_cls, dataset_cls.classes


def dataset_selection(
        path: str, transform: object, **kwargs,
) -> Tuple[Dataset, Dataset, Iterable[str]]:

    # Build dataset on Kather19
    
    return build_dataset_cls(path=path, transform=transform, 
                             remap=const_kather19, **kwargs)

def main(
        patch_path: str,
        model_path: str,
        exp_name: str,
        use_cuda: Optional[bool] = True,
) -> None:
    
    transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    logger = get_logger('logs/'+args.exp_name+'.log')
    # writer = SummaryWriter(comment=exp_name)
    device = 'cuda' if use_cuda else 'cpu'
    logger.debug('Build and load model from: {}'.format(model_path))
    model = models.__dict__['resnet18']()
    model.fc = nn.Linear(512,9)
    checkpoint = torch.load(model_path)

    # rename moco pre-trained keys
    # state_dict = checkpoint["state_dict"]
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()       

    if os.path.exists(patch_path):
        patchs_path = [patch_path]
    else:
        patchs_path = glob(patch_path) 

    dataset_cls = []
    n_cgt = []
    n_pred = []
    F1_all = []
    n=0
    print(patch_path)
    for i, p in enumerate(patchs_path):
        logger.debug('[{}/{}] Run classifcation {}'.format(i+1, len(patchs_path), p))

        dataset_cls_, cls_names = dataset_selection(
            path=p,
            transform = transform
        )
        dataset_cls.append(dataset_cls_)
        
    # Create dataset
    logger.debug("Load dataset {} from: {}".format(i, p))

    loader = DataLoader(
        dataset=ConcatDataset(dataset_cls), batch_size=args.bs, num_workers=args.j, shuffle=False, drop_last=False)
    for x_img, y_label in tqdm(loader, desc="Classification"):
        if use_cuda:
            x_img = x_img.to(device)
            y_label = y_label.to(device)
        with torch.no_grad():
            y_pred = model(x_img)
        n_pred.extend(y_pred.argmax(dim=1).detach().cpu().numpy())
        n_cgt.extend(y_label.detach().cpu().numpy())

    metric = metrics(n_cgt, n_pred, cls_names)

    # logger.debug('Epoch {}'.format(epoch))
    logger.debug('F1 score:\n\t{}'.format("\t".join(["{}: {:.4f}".format(a, b) for a, b in metric.items()])))
    # F1_score = [value for value in metric.values() if isinstance(value, (int, float))]
    # F1_average = ["{:.4f}".format(y) for y in [x for x in F1_score]]
    # # pattern = r"'{[^']*}'"
    # # strings = [re.search(pattern, str(value)).group(1) for value in metric.values()]
    # results = [':'.join(pair) for pair in zip(['ALL']+cls_names,F1_average)]

    # logger.debug('F1 score_average:\n\t %s', results)


if __name__ == '__main__':
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    parser = argparse.ArgumentParser()
    parser.add_argument("--source_domain", default='K19', type=str)
    parser.add_argument("--target_domain", default='S06', type=str)
    parser.add_argument('-plot',
                        action='store_true',
                        default = 'True',
                        help='Add argument to plot results as png')
    parser.add_argument('-force',
                        action='store_true',
                        help='If added, force the computation of patches even if output already exists')
    parser.add_argument('--j', default=4, type=int,
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--bs', default=256, type=int,
                        help='mini-batch size (default: 256)')
    args = parser.parse_args()
    
    if args.target_domain == 'K19':
        args.patch_path = 'DATA/CRC-VAL-HE-7K/'
    else:
        args.patch_path = 'DATA/GDPH-CRC-HE-MS/test/'+args.target_domain
    args.exp_name = args.source_domain+'To'+args.target_domain
    args.model_path = './checkpoints/Phase2_'+args.source_domain+'.pth.tar'

    main(
        patch_path = args.patch_path,
        model_path = args.model_path,
        exp_name = args.exp_name,
        use_cuda = torch.cuda.is_available()
    )