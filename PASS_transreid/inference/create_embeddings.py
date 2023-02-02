import os

import PIL.Image
import torch
from tqdm import tqdm
import sys
sys.path.append("..")
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_inference
from utils.logger import setup_logger
import torchvision.transforms as T
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("transreid", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID


    val_transforms = T.Compose([
        T.Resize(cfg.INPUT.SIZE_TEST),
        T.ToTensor(),
        T.Normalize(mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD)
    ])

    num_classes = 457
    camera_num = 9
    view_num = 1
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)
    model.eval()
    model = model.cuda()

    gallery = "/home/mscherbina/Documents/github_repos/centroids-reid/data/gallery-subfolders"
    pids = os.listdir(gallery)
    pids = [p for p in pids if os.path.isdir(os.path.join(gallery, p))]
    pids.sort()
    path_embeddings = {}

    for pid in tqdm(pids):
        pid_path = os.path.join(gallery, pid)
        images = os.listdir(pid_path)
        images = [os.path.join(pid_path, i) for i in images]
        images.sort()
        for path in images:
            image = PIL.Image.open(path)
            image = val_transforms(image)
            image = image.unsqueeze(0).cuda()
            with torch.autocast(enabled=True, device_type="cuda"):
                embedding = model(image)
            embedding = embedding.detach().cpu().numpy()[0]
            path_embeddings[path] = embedding

    np.save("/home/mscherbina/Documents/github_repos/PASS-reID/PASS_transreid/inference/output-dir/path_embeddings.npy", path_embeddings)


