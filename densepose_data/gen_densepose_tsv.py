import sys
import os
pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(__file__)))
print(pythonpath)
sys.path.insert(0, pythonpath)
from utils.common import encoded_from_img
from utils.tsv_io import tsv_writer
from PIL import Image
import json, cv2, math, yaml
import numpy as np
import base64

from typing import Any, ClassVar, Dict, List
import torch

from detectron2.config import CfgNode, get_cfg
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T

from densepose import add_densepose_config
from densepose.vis.densepose_results import (
    DensePoseResultsContourVisualizer,
    DensePoseResultsFineSegmentationVisualizer,
    DensePoseResultsUVisualizer,
    DensePoseResultsVVisualizer,
)
from densepose.vis.extractor import DensePoseResultExtractor
from densepose.vis.extractor import (
    create_extractor,
)

class DefaultPredictor:
    """
    Create a simple end-to-end predictor with the given config that runs on
    single device for a single input image.

    Compared to using the model directly, this class does the following additions:

    1. Load checkpoint from `cfg.MODEL.WEIGHTS`.
    2. Always take BGR image as the input and apply conversion defined by `cfg.INPUT.FORMAT`.
    3. Apply resizing defined by `cfg.INPUT.{MIN,MAX}_SIZE_TEST`.
    4. Take one input image and produce a single output, instead of a batch.

    This is meant for simple demo purposes, so it does the above steps automatically.
    This is not meant for benchmarks or running complicated inference logic.
    If you'd like to do anything more complicated, please refer to its source code as
    examples to build and use the model manually.

    Attributes:
        metadata (Metadata): the metadata of the underlying dataset, obtained from
            cfg.DATASETS.TEST.

    Examples:
    ::
        pred = DefaultPredictor(cfg)
        inputs = cv2.imread("input.jpg")
        outputs = pred(inputs)
    """

    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        # if len(cfg.DATASETS.TEST):
        #     self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            image.to(self.cfg.MODEL.DEVICE)

            inputs = {"image": image, "height": height, "width": width}

            predictions = self.model([inputs])[0]
            return predictions

def setup_config(
        config_fpath: str, model_fpath: str, args, opts: List[str]
    ):
    cfg = get_cfg()
    add_densepose_config(cfg)
    cfg.merge_from_file(config_fpath)
    cfg.merge_from_list(args.opts)
    if opts:
        cfg.merge_from_list(opts)
    cfg.MODEL.WEIGHTS = model_fpath
    cfg.freeze()
    return cfg

class ARGS(object):
    def __init__(self) -> None:
        self.model = './densepose_data/densepose_rcnn_R_101_FPN_DL_WC1M_s1x.pkl'
        self.cfg = './densepose_data/densepose_rcnn_R_101_FPN_DL_WC1M_s1x.yaml'
        self.input = 'ref.png'
        self.opts = []
    
args = ARGS()
opts = []
cfg = setup_config(args.cfg, args.model, args, opts)

vis_I = DensePoseResultsFineSegmentationVisualizer()
ext_I = create_extractor(vis_I)

vis_U = DensePoseResultsUVisualizer()
ext_U = create_extractor(vis_U)

vis_V = DensePoseResultsVVisualizer()
ext_V = create_extractor(vis_V)

vis = [vis_I, vis_U, vis_V]
ext = [ext_I, ext_U, ext_V]

dataset_dir = '/data1/lihaochen/TikTok_finetuning/TiktokDance'
# split = 'train_images'
split = 'new10val_images'

def tsv_reader(tsv_file, sep='\t'):
    with open(tsv_file, 'r') as fp:
        for i, line in enumerate(fp):
            yield [x.strip() for x in line.split(sep)]

tsv_fname_img = dataset_dir + f'/{split}.tsv'
tsv_imgs = tsv_reader(tsv_fname_img)

predictor = DefaultPredictor(cfg)

print('generating images.tsv')
def gen_row(img_rows):
    for i, img_row in enumerate(img_rows):
        image_key = img_row[0]
        image = cv2.imdecode(np.frombuffer(base64.b64decode(img_row[1]), np.uint8),cv2.IMREAD_COLOR)
        img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        output = predictor(img)["instances"]
        datas = []
        for e in ext:
            datas.append(e(output))
        row = [image_key]
        for data, v in zip(datas, vis):
            image_vis = v.visualize(np.zeros_like(img), data)
            row.append(encoded_from_img(image_vis))
        # row = [image_key, encoded_from_img(image)]
        yield(row)
tsv_writer(gen_row(tsv_imgs), f"{dataset_dir}/{split}_densepose_new.tsv")