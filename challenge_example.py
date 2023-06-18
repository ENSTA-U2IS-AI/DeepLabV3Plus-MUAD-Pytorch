from tqdm import tqdm
import network
import utils
import os
import argparse

import torch
import torch.nn as nn
import torchvision.transforms as T

import cv2


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--data_root", type=str, default='./datasets/data', help="path to Dataset")
    parser.add_argument("--save_path", type=str, default='./submission/', help="save path for the outputs")
    parser.add_argument("--dataset", type=str, default='muad',
                        choices=['voc', 'cityscapes','muad'], help='Name of dataset')
    parser.add_argument("--num_classes", type=int, default=None,
                        help="num classes (default: None), defined in the code according to the dataset")

    # Deeplab Options
    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=['deeplabv3_resnet50',  'deeplabv3plus_resnet50',
                                 'deeplabv3_resnet101', 'deeplabv3plus_resnet101',
                                 'deeplabv3_mobilenet', 'deeplabv3plus_mobilenet'], help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")

    # Visdom options
    parser.add_argument("--ckptpath", type=str, default='checkpoints',
                        help="folder where to save the ckt (default: checkpoints)")
    return parser


def get_dataset(opts, file):
    """ Dataset And Augmentation
    """
    val_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
    ])
    image_path = os.path.join(opts.data_root, file)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = val_transform(image).unsqueeze(0)

    return image


def inference(opts, model, device):
    """Do validation and return specified samples"""

    file_list = os.listdir(opts.data_root)
    if not os.path.exists(opts.save_path):
        os.makedirs(opts.save_path)

    with torch.no_grad():
        for i, file in tqdm(enumerate(file_list)):

            prediction = {}

            image = get_dataset(opts, file)

            image = image.to(device, dtype=torch.float32)

            outputs = model(image)
            outputs = outputs.detach().squeeze().cpu()
            outputs = torch.softmax(outputs, dim=0)
            conf = outputs.max(dim=0)[0].to(torch.float16)
            preds = outputs.max(dim=0)[1]
            prediction['conf'] = conf
            prediction['pred'] = preds

            save_path = os.path.join(opts.save_path, file.split('.')[0] + '.pth')
            torch.save(prediction, save_path)

def main():

    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
    elif opts.dataset.lower() == 'muad':
        opts.num_classes = 19

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Set up model
    if 'deeplabv3' in opts.model:
        model_map = {
            'deeplabv3_resnet50': network.deeplabv3_resnet50,
            'deeplabv3plus_resnet50': network.deeplabv3plus_resnet50,
            'deeplabv3_resnet101': network.deeplabv3_resnet101,
            'deeplabv3plus_resnet101': network.deeplabv3plus_resnet101,
            'deeplabv3_mobilenet': network.deeplabv3_mobilenet,
            'deeplabv3plus_mobilenet': network.deeplabv3plus_mobilenet
        }
        model = model_map[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
        if opts.separable_conv and 'plus' in opts.model:
            network.convert_to_separable_conv(model.classifier)
        utils.set_bn_momentum(model.backbone, momentum=0.01)
    else:
        print('Unknown model type. Existing.')
        exit()

    if opts.ckptpath is not None and os.path.isfile(opts.ckptpath):
        checkpoint = torch.load(opts.ckptpath, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        model.to(device)
        print("Model restored from %s" % opts.ckptpath)
        del checkpoint  # free memory
    else:
        print('No checkpoint is found. Maybe wrong path.')
        exit()

    # Set up metrics

    model.eval()
    inference(opts=opts, model=model, device=device)


if __name__ == '__main__':
    main()