import argparse


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, default="Data166_b4_15124")
    parser.add_argument("--debug", action='store_false')

    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--train_epoch", type=int, default=100)
    parser.add_argument("--height", type=int, default=224)
    # parser.add_argument("--width", type=int, default=256) ##vehicle
    parser.add_argument("--width", type=int, default=224) ##motobike and pedestrain
    parser.add_argument("--lr_ft", type=float, default=0.01, help='learning rate of feature extractor')
    parser.add_argument("--lr_new", type=float, default=0.1, help='learning rate of classifier_base')
    parser.add_argument('--classifier', type=str, default='base', help='classifier name')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--root_path', type=str, default='./images')
    parser.add_argument('--train_path', type=str, default='./train.csv')
    parser.add_argument('--valid_path', type=str, default='./valid.csv')
    parser.add_argument('--pytorch_model_path', type=str, default='./face.pt')
    parser.add_argument('--onnx_model_path', type=str, default='./b0_face.onnx')
    parser.add_argument('--yolo_dir', type=str, default='../checkpoints/best.pt')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument("--train_split", type=str, default="train", choices=['train', 'trainval'])
    parser.add_argument("--valid_split", type=str, default="test", choices=['test', 'valid'])
    parser.add_argument('--device', default="1", type=str, help='gpu device ids for CUDA_VISIBLE_DEVICES')
    parser.add_argument("--redirector", action='store_false')
    parser.add_argument('--use_bn', action='store_false')
    parser.add_argument('--use_pretrain', default=True, type=bool)

    return parser
