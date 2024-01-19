import os
import pickle
import pprint
from collections import OrderedDict, defaultdict
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from batch_engine import valid_trainer, batch_trainer
from dataset.Dataset import Data166, get_info, get_transform
from config import argument_parser
from loss.CE_loss import CEL_Sigmoid
from models.base_block import FeatClassifier, BaseClassifier
from models.resnet import resnet50, resnet101,resnet18,resnet34
from models.senet import se_resnet50, se_resnet101
from models.efficientnet import efficientnet_b0, efficientnet_b1,efficientnet_b2,efficientnet_b3
from tools.function import get_model_log_path, get_pedestrian_metrics
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, AverageMeter
from tools.function import get_model_log_path, get_pedestrian_metrics, load_pretrained_weights
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.onnx as torch_onnx
import onnx
import onnxruntime as rt
set_seed(605)

def main(args):
    visenv_name = args.dataset
    exp_dir = os.path.join('exp_result_vehicle1', args.dataset)
    model_dir, log_dir = get_model_log_path(exp_dir, visenv_name)
    stdout_file = os.path.join(log_dir, f'stdout_{time_str()}.txt')

    if args.redirector:
        print('redirector stdout')
        ReDirectSTD(stdout_file, 'stdout', False)

    pprint.pprint(OrderedDict(args.__dict__))

    print('-' * 60)
    print(f'use GPU{args.device} for training')
    print(f'train set: {args.dataset} {args.train_split}, test set: {args.valid_split}')

    train_tsfm, valid_tsfm = get_transform(args)
    print(valid_tsfm)

  
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

    # backbone = resnet50()
    # backbone = resnet18()
    # backbone = resnet34()
    backbone = efficientnet_b0()
    #backbone = se_resnet50()
    classifier = BaseClassifier(nattr=train_set.attr_num)
    model = FeatClassifier(backbone, classifier)


    print("reloading pretrained models")

    exp_dir = os.path.join('exp_result', args.dataset)
    model_path = os.path.join(exp_dir, args.dataset, 'img_model', 'ckpt_max.pth')

    load_pretrained_weights(model, model_path)
    model.eval()
    model_onnx_path = args.onnx_model_path

    dummy_input = torch.randn(1, 3, 224, 224) 
    torch_onnx.export(model, dummy_input, model_onnx_path,
                        verbose=True, input_names=["image"],
                        output_names=["output"], opset_version=11,
                        dynamic_axes={'image' : {0 : 'batch_size'},    
                                'output' : {0 : 'batch_size'}})
    print("Export of {} complete!".format(model_onnx_path))

    print("Testing...")
    model_onnx = onnx.load(model_onnx_path)
    sess = rt.InferenceSession(model_onnx_path)

    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    input_type = sess.get_inputs()[0].type
    output_name = sess.get_outputs()[0].name
    output_shape = sess.get_outputs()[0].shape
    output_type = sess.get_outputs()[0].type
    print("input name: {}, input shape: {}, input type: {}".format(input_name, input_shape, input_type))
    print("output name: {}, output shape: {}, output type: {}".format(output_name, output_shape, output_type))

if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()

    main(args)

