import os
from collections import OrderedDict

import numpy as np
import torch
from easydict import EasyDict

from tools.utils import may_mkdirs


def ratio2weight(targets, ratio):
    ratio = torch.from_numpy(ratio).type_as(targets)
    pos_weights = targets * (1 - ratio)
    neg_weights = (1 - targets) * ratio
    weights = torch.exp(neg_weights + pos_weights)

    # for RAP dataloader, targets element may be 2, with or without smooth, some element must great than 1
    weights[targets > 1] = 0.0

    return weights


def get_model_log_path(root_path, visenv):
    multi_attr_model_dir = os.path.join(root_path, f'{visenv}', 'img_model')
    may_mkdirs(multi_attr_model_dir)

    multi_attr_log_dir = os.path.join(root_path, f'{visenv}', 'log')
    may_mkdirs(multi_attr_log_dir)

    return multi_attr_model_dir, multi_attr_log_dir


class LogVisual:

    def __init__(self, args):
        self.args = vars(args)
        self.train_loss = []
        self.val_loss = []

        self.ap = []
        self.map = []
        self.acc = []
        self.prec = []
        self.recall = []
        self.f1 = []

        self.error_num = []
        self.fn_num = []
        self.fp_num = []

        self.save = False

    def append(self, **kwargs):
        self.save = False

        if 'result' in kwargs:
            self.ap.append(kwargs['result']['label_acc'])
            self.map.append(np.mean(kwargs['result']['label_acc']))
            self.acc.append(np.mean(kwargs['result']['instance_acc']))
            self.prec.append(np.mean(kwargs['result']['instance_precision']))
            self.recall.append(np.mean(kwargs['result']['instance_recall']))
            self.f1.append(np.mean(kwargs['result']['floatance_F1']))

            self.error_num.append(kwargs['result']['error_num'])
            self.fn_num.append(kwargs['result']['fn_num'])
            self.fp_num.append(kwargs['result']['fp_num'])

        if 'train_loss' in kwargs:
            self.train_loss.append(kwargs['train_loss'])
        if 'val_loss' in kwargs:
            self.val_loss.append(kwargs['val_loss'])


def get_pkl_rootpath(dataset):
    root = os.path.join("./data", f"{dataset}")
    data_path = os.path.join(root, 'dataset.pkl')

    return data_path

def load_checkpoint(fpath):
    r"""Loads checkpoint.

    ``UnicodeDecodeError`` can be well handled, which means
    python2-saved files can be read from python3.

    Args:
        fpath (str): path to checkpoint.

    Returns:
        dict

    Examples::
        >>> from torchreid.utils import load_checkpoint
        >>> fpath = 'log/my_model/model.pth.tar-10'
        >>> checkpoint = load_checkpoint(fpath)
    """
    if fpath is None:
        raise ValueError('File path is None')
    if not os.path.exists(fpath):
        raise FileNotFoundError('File is not found at "{}"'.format(fpath))
    map_location = None if torch.cuda.is_available() else 'cpu'
    try:
        checkpoint = torch.load(fpath, map_location=map_location)
    except UnicodeDecodeError:
        pickle.load = partial(pickle.load, encoding="latin1")
        pickle.Unpickler = partial(pickle.Unpickler, encoding="latin1")
        checkpoint = torch.load(
            fpath, pickle_module=pickle, map_location=map_location
        )
    except Exception:
        print('Unable to load checkpoint from "{}"'.format(fpath))
        raise
    return checkpoint

def get_pedestrian_metrics(gt_label, preds_probs, threshold=0.5):
    pred_label = preds_probs > threshold

    eps = 1e-20
    result = EasyDict()

    ###############################
    # label metrics
    # TP + FN
    gt_pos = np.sum((gt_label == 1), axis=0).astype(float)
    # TN + FP
    gt_neg = np.sum((gt_label == 0), axis=0).astype(float)
    # TP
    true_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=0).astype(float)
    # TN
    true_neg = np.sum((gt_label == 0) * (pred_label == 0), axis=0).astype(float)
    # FP
    false_pos = np.sum(((gt_label == 0) * (pred_label == 1)), axis=0).astype(float)
    # FN
    false_neg = np.sum(((gt_label == 1) * (pred_label == 0)), axis=0).astype(float)

    label_pos_recall = 1.0 * true_pos / (gt_pos + eps)  # true positive
    label_neg_recall = 1.0 * true_neg / (gt_neg + eps)  # true negative
    # mean accuracy
    label_ma = (label_pos_recall + label_neg_recall) / 2

    result.label_pos_recall = label_pos_recall
    result.label_neg_recall = label_neg_recall
    result.label_prec = true_pos / (true_pos + false_pos + eps)
    result.label_acc = true_pos / (true_pos + false_pos + false_neg + eps)
    result.label_f1 = 2 * result.label_prec * result.label_pos_recall / (
            result.label_prec + result.label_pos_recall + eps)

    result.label_ma = label_ma
    result.ma = np.mean(label_ma)

    ################
    # instance metrics
    gt_pos = np.sum((gt_label == 1), axis=1).astype(float)
    true_pos = np.sum((pred_label == 1), axis=1).astype(float)
    # true positive
    intersect_pos = np.sum((gt_label == 1) * (pred_label == 1), axis=1).astype(float)
    # IOU
    union_pos = np.sum(((gt_label == 1) + (pred_label == 1)), axis=1).astype(float)

    instance_acc = intersect_pos / (union_pos + eps)
    instance_prec = intersect_pos / (true_pos + eps)
    instance_recall = intersect_pos / (gt_pos + eps)
    instance_f1 = 2 * instance_prec * instance_recall / (instance_prec + instance_recall + eps)

    instance_acc = np.mean(instance_acc)
    instance_prec = np.mean(instance_prec)
    instance_recall = np.mean(instance_recall)
    instance_f1 = np.mean(instance_f1)

    result.instance_acc = instance_acc
    result.instance_prec = instance_prec
    result.instance_recall = instance_recall
    result.instance_f1 = instance_f1

    result.error_num, result.fn_num, result.fp_num = false_pos + false_neg, false_neg, false_pos

    return result

def load_pretrained_weights(model, weight_path):
    r"""Loads pretrianed weights to model.

    Features::
        - Incompatible layers (unmatched in name or size) will be ignored.
        - Can automatically deal with keys containing "module.".

    Args:
        model (nn.Module): network model.
        weight_path (str): path to pretrained weights.

    Examples::
        >>> from torchreid.utils import load_pretrained_weights
        >>> weight_path = 'log/my_model/model-best.pth.tar'
        >>> load_pretrained_weights(model, weight_path)
    """
    checkpoint = load_checkpoint(weight_path)
    if 'state_dicts' in checkpoint:
        state_dict = checkpoint['state_dicts']
    else:
        state_dict = checkpoint

    model_dict = model.state_dict()
    new_state_dict = OrderedDict()
    matched_layers, discarded_layers = [], []

    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:] # discard module.

        if k in model_dict and model_dict[k].size() == v.size():
            new_state_dict[k] = v
            matched_layers.append(k)
        else:
            discarded_layers.append(k)

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict)

    if len(matched_layers) == 0:
        warnings.warn(
            'The pretrained weights "{}" cannot be loaded, '
            'please check the key names manually '
            '(** ignored and continue **)'.format(weight_path)
        )
    else:
        print(
            'Successfully loaded pretrained weights from "{}"'.
            format(weight_path)
        )
        if len(discarded_layers) > 0:
            print(
                '** The following layers are discarded '
                'due to unmatched keys or layer size: {}'.
                format(discarded_layers)
            )