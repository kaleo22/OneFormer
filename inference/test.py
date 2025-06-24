from defaults import DefaultPredictor
import argparse
import multiprocessing as mp
import random
import sys
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from detectron2.data import MetadataCatalog
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger
import numpy as np
import torch
import tqdm
import time
import cv2

def get_parser():
    parser = argparse.ArgumentParser(description="oneformer demo for builtin configs")
    parser.add_argument(
        "--config-file",
        default="../configs/ade20k/swin/oneformer_swin_large_IN21k_384_bs16_160k.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--task", help="Task type")
    parser.add_argument(
        "--input",
        nargs="+",
        help="A list of space separated input images; "
        "or a single glob pattern such as 'directory/*.jpg'",
    )
    parser.add_argument(
        "--output",
        help="A file or directory to save output visualizations. "
        "If not given, will show output in an OpenCV window.",
    )

    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

class inference(object):
    def __init__(self, cfg):
        self.predictor = DefaultPredictor(cfg)

    def run(self, image, task='instance'):
        image = image[:, :, ::-1]
        outputs = self.predictor(image, task)
        instances = outputs["instances"].to(self.predictor.cpu_device)
        return outputs

    def get_metadata(self):
        return self.metadata
    
if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    infer = inference(cfg)

    cap = cv2.VideoCapture(args.input)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not cap.isOpened():
        raise IOError("Cannot open Input Video Stream")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        logger.info(f"Processing frame {frame_number}...")
        
        start_time = time.time()
        predictions, *_ = infer.run(frame)
        logger.info(
            "Frame {}: {} in {:.2f}s".format(
                frame_number,
                "detected {} instances".format(len(predictions["instances"]))
                if "instances" in predictions
                else "finished",
                time.time() - start_time,
            )
        )

    if args.input:
        for path in tqdm.tqdm(args.input, disable=not args.output):
            # use PIL, to be consistent with evaluation
                
            img = read_image(path, format="RGB")
            start_time = time.time()
            predictions, *_ = infer.run(img)
            logger.info(
                "{}: {} in {:.2f}s".format(
                    path,
                    "detected {} instances".format(len(predictions["instances"]))
                    if "instances" in predictions
                    else "finished",
                    time.time() - start_time,
                )
            )
            #if args.output:
                #if len(args.input) == 1:
                    #for k in visualized_output.keys():
                       # os.makedirs(k, exist_ok=True)
                        #out_filename = os.path.join(k, args.output)
                        #visualized_output[k].save(out_filename)    
                #else:
                    #for k in visualized_output.keys():
                        #opath = os.path.join(args.output, k)    
                        #os.makedirs(opath, exist_ok=True)
                        #out_filename = os.path.join(opath, os.path.basename(path))
                        #visualized_output[k].save(out_filename)    
            #else:
                #raise ValueError("Please specify an output path!")
    else:
        raise ValueError("No Input Given")

   

    print(predictions)