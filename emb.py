import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import pickle
from collections import defaultdict
from datetime import datetime
import numpy as np
import argparse
import json
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

from configs import cfg, update_config
#from dataset.multi_label.coco import COCO14
from dataset.augmentation import get_transform
from metrics.ml_metrics import get_multilabel_metrics
from metrics.face_metrics import get_face_metrics
from models.model_ema import ModelEmaV2
from optim.adamw import AdamW
#from scheduler.cosine_lr import CosineLRScheduler
from tools.distributed import distribute_bn
from tools.vis import tb_visualizer_face
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from dataset.face_attr.face import FaceAttr, FaceAttrHackathon, FaceAttrHackathonText
from models.base_block import FeatClassifier
from models.model_factory import build_loss, build_classifier, build_backbone

from tools.function import get_model_log_path, get_reload_weight, seperate_weight_decay
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, str2bool, gen_code_archive
from models.backbone import swin_transformer2
# from models.backbone.models import DefineModel
from losses import bceloss, scaledbceloss
from models import base_block

from tqdm import tqdm

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
# torch.autograd.set_detect_anomaly(True)
torch.autograd.set_detect_anomaly(True)

def emb_fea(model, dataloader):
    model.eval()

    EMB = {}

    with torch.no_grad():
        for images, labels, _ in dataloader:
            # images, labels = images.cuda(), labels.cuda()
            if torch.is_tensor(images):
                images = images.cuda()
            if torch.is_tensor(labels):
                labels = labels.cuda()

            emb_fea, logits = model(images)   # , embed=True, for resnet, add 'embed=True'

            label_list = labels.tolist()
            for emb, label in zip(emb_fea, label_list):
                assert len(emb) == 64
                # if str(label) in EMB:
                #     for j in range(len(emb)):
                #         EMB[str(label)][j].append(round(emb[j].item(), 4))
                # else:
                #     EMB[str(label)] = [[] for _ in range(len(emb))]
                #     for j in range(len(emb)):
                #         EMB[str(label)][j].append(round(emb[j].item(), 4))

                label_str = str(label)
                print(label_str)
                emb_list = emb.cpu().numpy().tolist()

                if label_str not in EMB:
                    EMB[label_str] = [[] for _ in range(len(emb))]
                
                for j, value in enumerate(emb_list):
                    if isinstance(value, list):
                        EMB[label_str][j].extend([round(val, 4) for val in value])
                    else:
                        EMB[label_str][j].append(round(value, 4))

                    

    for key, value in EMB.items():
        for i in range(64):
            EMB[key][i] = round(np.array(EMB[key][i]).mean(), 4)
    
    return EMB

def main(cfg, args):
    # seed = int(np.random.choice(1000, 1))
    seed = int(605)
    
    set_seed(seed)


    train_tsfm, valid_tsfm = get_transform(cfg)
    if args.local_rank == 0:
        print(train_tsfm)

    train_set = FaceAttrHackathon(cfg=cfg, csv_file=cfg.DATASET.PHASE1_ROOT_PATH + '/train.csv', transform=train_tsfm,
                            root_path=cfg.DATASET.ROOT_PATH, target_transform=cfg.DATASET.TARGETTRANSFORM, train=True)


    # if args.distributed:
    # train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
    # else:
    train_sampler = None

    train_loader = DataLoader(
        dataset=train_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        sampler=train_sampler,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )


    labels = train_set.label
    label_ratio = labels.mean(0) if cfg.LOSS.SAMPLE_WEIGHT else None

    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)

    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=train_set.attr_num,
        c_in=1024 * 2,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale =cfg.CLASSIFIER.SCALE
    )

    model = FeatClassifier(backbone, classifier, bn_wd=cfg.TRAIN.BN_WD)
    # model = model.cuda()

    model = torch.nn.DataParallel(model)

    # model = get_reload_weight("/data/AI/backup/phuongdd/Computer-Vison/attributes", model, pth='swin_base_patch4_window7_224_22k.pth')
    model = model.cuda()
    
    # print("-------------------Batch")
    # for addition, labels, images in train_loader:
    #     print('-------------Labels',labels)
    #     print('-------------Images',images)
        
    
    emb = emb_fea(model, train_loader)
    emb_json = json.dumps(emb, indent=4)
    with open("/data/AI/backup/phuongdd/Computer-Vison/attributes/ELPT-AI-Hackathon-Face-Analysis/center_emb_train_nopretrained.json", 'w', encoding='utf-8') as f:
        f.write(emb_json)
    f.close()
    
def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
        default="/data/AI/backup/phuongdd/Computer-Vison/attributes/ELPT-AI-Hackathon-Face-Analysis/configs/face.yaml",

    )
    parser.add_argument("--emb_size", type=int, default=2048, help="emb fea size")
    parser.add_argument('--local_rank', help='node rank for distributed training', default=0,
                        type=int)
    parser.add_argument('--distributed', help='distributed', default=True,
                        type=str)
    parser.add_argument('--dist_bn', type=str, default='',
                        help='Distribute BatchNorm stats between nodes after each epoch ("broadcast", "reduce", or "")')


    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()

    update_config(cfg, args)
    main(cfg, args)
