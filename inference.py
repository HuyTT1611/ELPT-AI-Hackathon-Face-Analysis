import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import pickle
from collections import defaultdict
from datetime import datetime

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom

from configs import cfg, update_config
#from dataset.multi_label.coco import COCO14
from dataset.augmentation import get_transform
from metrics.ml_metrics import get_multilabel_metrics
from models.model_ema import ModelEmaV2
from optim.adamw import AdamW
# from scheduler.cosine_lr import CosineLRScheduler
from tools.distributed import distribute_bn
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau, MultiStepLR
from torch.utils.data import DataLoader
import torchvision.transforms as T


from batch_engine import valid_trainer, batch_trainer
from models.base_block import FeatClassifier
from models.model_factory import build_loss, build_classifier, build_backbone

from tools.function import get_model_log_path, get_reload_weight, seperate_weight_decay
from tools.utils import time_str, save_ckpt, ReDirectSTD, set_seed, str2bool, gen_code_archive
from models.backbone import swin_transformer2
from losses import bceloss, scaledbceloss
from models import base_block
from tqdm import tqdm
import csv
from PIL import Image
from ultralytics import YOLO
import json
torch.set_printoptions(precision=10)

set_seed(605)


def main(cfg, args):
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, _ = get_model_log_path(exp_dir, cfg.NAME)

    _, valid_tsfm = get_transform(cfg)

    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)
    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=24,
        c_in=2048,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale =cfg.CLASSIFIER.SCALE
    )
    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    model = get_reload_weight(model_dir, model, pth='/data/AI/backup/phuongdd/Computer-Vison/face_attributes/best_model.pth')
    # Write results
    model.eval()
    data = []
    first_line = ['file_name', 'bbox', 'image_id', 'race', 'age', 'emotion', 'gender', 'skintone', 'masked']
    attribute = ['Baby', 'Kid', 'Teenager', '20-30s', '40-50s', 'Senior', 'Negroid', 'Caucasian', 'Mongoloid', 'unmasked', 'masked', 'mid-light', 'light', 'mid-dark', 'dark',
              'Fear', 'Disgust', 'Surprise', 'Anger', 'Sadness', 'Neutral', 'Happiness', 'Female', 'Male']
    model_yolo = YOLO(args.yolo_dir)
    json_file_path = args.json_dir

    # Read the JSON file
    with open(json_file_path, 'r') as file:
        json_data = json.load(file)
        
    if args.phase == "public" :
        image_list = list(json_data.keys())
        # Now, json_data is a Python dictionary containing the data from the JSON file
        with torch.no_grad():
            for filename in image_list:
                image = Image.open(os.path.join(args.images_dir, filename)).convert('RGB')
                copy_image = image.copy()
                results = model_yolo(image)
                image_width, image_height = image.size
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x, y, width, height = box.xywh[0]
                        x = x-width/2
                        y = y-height/2
                        # Crop the image based on the bounding box
                        cropped_image = copy_image.crop((int(x), int(y), int(x + width), int(y + height)))
                        cropped_image = valid_tsfm(cropped_image)
                        cropped_image = cropped_image.unsqueeze(0).cuda()
                        valid_logits, attns = model(cropped_image)
                        valid_probs = torch.sigmoid(valid_logits[0])
                        for valid_prob in valid_probs:
                            new_row = []
                            new_row.append(filename)
                            new_row.append([float(x.cpu()),float(y.cpu()), float(width.cpu()), float(height.cpu())])
                            new_row.append(json_data[filename])
                            _, max_index_race = torch.max(valid_prob[6:9], dim=0)
                            _, max_index_age = torch.max(valid_prob[0:6], dim=0)
                            _, max_index_emotion = torch.max(valid_prob[15:22], dim=0)
                            _, max_index_gender = torch.max(valid_prob[22:24], dim=0)
                            _, max_index_skintone = torch.max(valid_prob[11:15], dim=0)
                            _, max_index_masked = torch.max(valid_prob[9:11], dim=0)
                            new_row.append(attribute[max_index_race+6])
                            new_row.append(attribute[max_index_age])
                            new_row.append(attribute[max_index_emotion+15])
                            new_row.append(attribute[max_index_gender+22])
                            new_row.append(attribute[max_index_skintone+11])
                            new_row.append(attribute[max_index_masked+9])
                            print(new_row)
                            data.append(new_row)
                            
            data = [first_line] + data
            answer_csv_file = "./answer.csv"
            with open(answer_csv_file, mode='w', newline='') as answer_file:
                answer_writer = csv.writer(answer_file)
                for row in data:
                    answer_writer.writerow(row)
    else :
        first_line = ['','file_name', 'height', 'width','image_id','bbox', 'race', 'age','skintone', 'emotion', 'masked', 'gender']
        # first_line = ['file_name', 'bbox', 'image_id', 'race', 'age', 'emotion', 'gender', 'skintone', 'masked']
        
        json_data
        
        # Now, json_data is a Python dictionary containing the data from the JSON file
        with torch.no_grad():
            
            # for i,img in enumerate(image_list):
            count=0
            for key, value in json_data.items():
                filename = key
                # filename=file
                image = Image.open(os.path.join(args.images_dir, filename)).convert('RGB')
                height_o, width_o=image.height, image.width
                copy_image = image.copy()
                results = model_yolo(image)
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x, y, width, height = box.xywh[0]
                        x = x-width/2
                        y = y-height/2
                        # Crop the image based on the bounding box
                        cropped_image = copy_image.crop((int(x), int(y), int(x + width), int(y + height)))
                        cropped_image = valid_tsfm(cropped_image)
                        cropped_image = cropped_image.unsqueeze(0).cuda()
                        valid_logits, attns = model(cropped_image)
                        valid_probs = torch.sigmoid(valid_logits[0])
                        for valid_prob in valid_probs:
                            new_row = []
                            new_row.append(int(count))
                            new_row.append(filename)
                            new_row.append(int(height_o))
                            new_row.append(int(width_o))
                            new_row.append(value)
                            new_row.append([float(x.cpu()),float(y.cpu()), float(width.cpu()), float(height.cpu())])
                            _, max_index_race = torch.max(valid_prob[6:9], dim=0)
                            _, max_index_age = torch.max(valid_prob[0:6], dim=0)
                            _, max_index_emotion = torch.max(valid_prob[15:22], dim=0)
                            _, max_index_gender = torch.max(valid_prob[22:24], dim=0)
                            _, max_index_skintone = torch.max(valid_prob[11:15], dim=0)
                            _, max_index_masked = torch.max(valid_prob[9:11], dim=0)
                            new_row.append(attribute[max_index_race+6])
                            new_row.append(attribute[max_index_age])
                            new_row.append(attribute[max_index_skintone+11])
                            new_row.append(attribute[max_index_emotion+15])
                            new_row.append(attribute[max_index_masked+9])
                            new_row.append(attribute[max_index_gender+22])
                            print(new_row)
                            data.append(new_row)
                
                            count+=1
                           
            data = [first_line] + data
            answer_csv_file = "./answer.csv"
            with open(answer_csv_file, mode='w', newline='') as answer_file:
                answer_writer = csv.writer(answer_file)
                for row in data:
                    answer_writer.writerow(row)


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", default='./configs/face.yaml', help="decide which cfg to use", type=str,
    )
    parser.add_argument("--debug", type=str2bool, default="true")

    parser.add_argument("--images_dir", type=str, default='/data/AI/backup/phuongdd/Computer-Vison/face_attributes/private_test_data')

    parser.add_argument("--json_dir", type=str, default='/data/AI/backup/phuongdd/Computer-Vison/face_attributes/private_test/private_test/file_name_to_image_id.json')

    parser.add_argument("--yolo_dir", type=str, default='/data/AI/backup/phuongdd/Computer-Vison/face_attributes/best.pt')

    parser.add_argument("--phase", type=str, default='private')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)

    main(cfg, args)
