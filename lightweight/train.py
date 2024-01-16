import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import pprint
from collections import OrderedDict, defaultdict

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from batch_engine import valid_trainer, batch_trainer
from config import argument_parser
from dataset.Dataset import Data166,get_transform
from loss.CE_loss import CEL_Sigmoid
from loss.AsymmetricLoss import AsymmetricLossOptimized
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50, resnet101, resnext50_32x4d,resnet152,resnet18,resnet34
from models.senet import se_resnet50, se_resnet101,se_resnext50_32x4d,se_resnet152
from models.efficientnet import efficientnet_b0, efficientnet_b6,efficientnet_b2,efficientnet_b4
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed
from torchsummary import summary

set_seed(605)

def main(args):
    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result', args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')
    save_model_path = os.path.join(model_dir, 'ckpt_max.pth')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    print(f'use GPU{args.device} for training')
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    print(train_tsfm)

    train_set = Data166(split='train',root=args.root_path,data_path=args.train_path,transform=train_tsfm)
    valid_set = Data166(split='val',root=args.root_path,data_path=args.valid_path,transform=valid_tsfm)
    
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=args.batchsize,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )


    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=args.batchsize,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f'{args.train_split} set: {len(train_loader.dataset)}, '
          f'{args.valid_split} set: {len(valid_loader.dataset)}, '
          f'attr_num : {train_set.attr_num}')

    labels = train_set.label
    sample_weight = np.array(labels, dtype=np.int32).mean(0)
  
    backbone = efficientnet_b0()
    # backbone = resnet18()
    # backbone = efficientnet_b2()
    classifier = BaseClassifier(nattr=train_set.attr_num)
    model = FeatClassifier(backbone, classifier)
    #summary(model.to('cuda'), (3,288,288))

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
        
    if args.use_pretrain==True:
        exp_dir = os.path.join('exp_result_vehicle1', args.dataset)
        model_path = os.path.join(exp_dir, args.dataset, 'img_model', 'ckpt_max.pth')
        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['state_dicts'])

    
    criterion = CEL_Sigmoid(sample_weight)
    # criterion = AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True)
    

    param_groups = [{'params': model.module.finetune_params(), 'lr': args.lr_ft},
                    {'params': model.module.fresh_params(), 'lr': args.lr_new}]
    optimizer = torch.optim.SGD(param_groups, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=False)
    lr_scheduler = ReduceLROnPlateau(optimizer, factor=0.1, patience=4)

    best_metric, epoch = trainer(epoch=args.train_epoch,
                                 model=model,
                                 train_loader=train_loader,
                                 valid_loader=valid_loader,
                                 criterion=criterion,
                                 optimizer=optimizer,
                                 lr_scheduler=lr_scheduler,
                                 path=save_model_path)

    print(f'{visenv_name},  best_metrc : {best_metric} in epoch{epoch}')


def trainer(epoch, model, train_loader, valid_loader, criterion, optimizer, lr_scheduler,
            path):
    maximum = float(-np.inf)
    best_epoch = 0

    result_list = defaultdict()

    for i in range(epoch):

        train_loss, train_gt, train_probs = batch_trainer(
            epoch=i,
            model=model,
            train_loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
        )

        valid_loss, valid_gt, valid_probs = valid_trainer(
            model=model,
            valid_loader=valid_loader,
            criterion=criterion,
        )

        # lr_scheduler.step(metrics=valid_loss, epoch=i)
        lr_scheduler.step(metrics=valid_loss)
        train_result = get_pedestrian_metrics(train_gt, train_probs)
        valid_result = get_pedestrian_metrics(valid_gt, valid_probs)

        print(f'Evaluation on test set, \n',
              'ma: {:.4f},  pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  valid_result.ma, np.mean(valid_result.label_pos_recall), np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1))

        print(f'{time_str()}')
        print('-' * 60)

        cur_metric = valid_result.ma

        if cur_metric > maximum:
            maximum = cur_metric
            best_epoch = i
            save_ckpt(model, path, i, maximum)

        result_list[i] = [train_result, valid_result]

    torch.save(result_list, os.path.join(os.path.dirname(path), 'metric_log.pkl'))

    return maximum, best_epoch


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    main(args)



