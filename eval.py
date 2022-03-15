import argparse
import os

import crossView

import numpy as np

import torch
from torch.utils.data import DataLoader

import cv2
import tqdm

from utils import mean_IU, mean_precision
from opt import get_eval_args as get_args

from PIL import Image
import matplotlib.pyplot as PLT
import matplotlib.cm as mpl_color_map


def readlines(filename):
    """Read all the lines in a text file and return as a list
    """
    with open(filename, 'r') as f:
        lines = f.read().splitlines()
    return lines


def load_model(models, model_path):
    """Load model(s) from disk
    """
    model_path = os.path.expanduser(model_path)

    assert os.path.isdir(model_path), \
        "Cannot find folder {}".format(model_path)
    print("loading model from folder {}".format(model_path))

    for key in models.keys():
        print("Loading {} weights...".format(key))
        path = os.path.join(model_path, "{}.pth".format(key))
        model_dict = models[key].state_dict()
        pretrained_dict = torch.load(path)
        pretrained_dict = {
            k: v for k,
                     v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        models[key].load_state_dict(model_dict)
    return models


def evaluate():
    opt = get_args()

    # Loading Pretarined Model
    models = {}
    models["encoder"] = crossView.Encoder(18, opt.height, opt.width, True)
    models['CycledViewProjection'] = crossView.CycledViewProjection(in_dim=8)
    models["CrossViewTransformer"] = crossView.CrossViewTransformer(128)

    models["decoder"] = crossView.Decoder(
        models["encoder"].resnet_encoder.num_ch_enc, opt.num_class)
    models["transform_decoder"] = crossView.Decoder(
        models["encoder"].resnet_encoder.num_ch_enc, opt.num_class, "transform_decoder")

    for key in models.keys():
        models[key].to("cuda")

    models = load_model(models, opt.pretrained_path)

    # Loading Validation/Testing Dataset

    # Data Loaders
    dataset_dict = {"3Dobject": crossView.KITTIObject,
                    "odometry": crossView.KITTIOdometry,
                    "argo": crossView.Argoverse,
                    "raw": crossView.KITTIRAW,
                    "habitat": crossView.Habitat}

    dataset = dataset_dict[opt.split]
    fpath = os.path.join(
        os.path.dirname(__file__),
        "splits",
        opt.split,
        "gibson_8fid_{}.txt")
    test_filenames = readlines(fpath.format("val"))
    test_dataset = dataset(opt, test_filenames, is_train=False)
    test_loader = DataLoader(
        test_dataset,
        1,
        False,
        num_workers=opt.num_workers,
        pin_memory=True,
        drop_last=True)

    ious = []

    iou, mAP = np.array([0., 0.]), np.array([0., 0.])
    trans_iou, trans_mAP = np.array([0., 0.]), np.array([0., 0.])
    for batch_idx, inputs in tqdm.tqdm(enumerate(test_loader)):
        with torch.no_grad():
            outputs = process_batch(opt, models, inputs)

        scene = inputs['folder'][0]
        # print(inputs["filename"][0])

        # Predicted Topview
        save_topview(
            inputs["filename"],
            outputs["topview"],
            os.path.join(
                opt.out_dir,
                'pred',
                scene,
                "{}.png".format(inputs["frame_index"][0])))

        # Get the unresized GT topview
        gt_64_tp = cv2.imread(os.path.join(
            "/scratch/jaidev/HabitatGibson/bevs",
            "partial_occupancy",
            inputs["folder"][0],
            "{}.png".format(inputs["frame_index"][0])
        ))

        # Save unresized GT topview
        save_gt_unresized_topview(
            inputs["filename"],
            gt_64_tp,
            os.path.join(
                opt.out_dir,
                "unresized_gt",
                scene,
                "{}.png".format(inputs["frame_index"][0])))

        # Groundtruth topview
        save_gt_topview(
            inputs["filename"],
            inputs[opt.type + "_gt"],
            os.path.join(
                opt.out_dir,
                opt.type + "_gt",
                scene,
                "{}.png".format(inputs["frame_index"][0])))
         
        # Save the RGB image!
        save_img(
            inputs["filename"],
            inputs["color"],
            os.path.join(
                opt.out_dir,
                'rgb',
                scene,
                 "{}.png".format(inputs["frame_index"][0])))

        pred = np.squeeze(
            torch.argmax(
                outputs["topview"].detach(),
                1).cpu().numpy())
        trans_pred = np.squeeze(
            torch.argmax(
                outputs["transform_topview"].detach(),
                1).cpu().numpy())
        true = np.squeeze(inputs[opt.type + "_gt"].detach().cpu().numpy())
                
        iou += mean_IU(pred, true)
        mAP += mean_precision(pred, true)

        trans_iou += mean_IU(trans_pred, true)
        trans_mAP += mean_precision(trans_pred, true)

        score = mean_IU(pred, true)
        score2 = mean_precision(pred, true)
        # print(score, score2)
        ious.append(score[1])

    iou /= len(test_loader)
    mAP /= len(test_loader)

    trans_iou /= len(test_loader)
    trans_mAP /= len(test_loader)

    print("Evaluation Results: mIoU (0) %.4f mAP (0): %.4f | mIOU (1): %.4f mAP (1): %.4f" % (iou[0], mAP[0], iou[1], mAP[1]))
    print("Evaluation Results (Transformed Layout): mIoU (0) %.4f mAP (0): %.4f | mIOU (1): %.4f mAP (1): %.4f" % (trans_iou[0], trans_mAP[0], trans_iou[1], trans_mAP[1]))

    # Top 10 examples
    ious = np.array(ious)
    sorted_iou_index = np.argsort(ious)[::-1]
    print(ious[sorted_iou_index])
    print("Top examples:")

    for i in range(len(sorted_iou_index[:25])):

        batch_idx = sorted_iou_index[i]

        inputs = None
        for batch_idx_data, inputs_data in enumerate(test_loader):
            if batch_idx == batch_idx_data:
                inputs = inputs_data
                break

        with torch.no_grad():
            outputs = process_batch(opt, models, inputs)

        scene = inputs['folder'][0]
        # print(inputs["filename"][0])

        # Predicted Topview
        save_topview(
            inputs["filename"],
            outputs["topview"],
            os.path.join(
                opt.out_dir,
                "top",
                'pred',
                scene,
                "{}.png".format(inputs["frame_index"][0])))

        # Get the unresized GT topview
        gt_64_tp = cv2.imread(os.path.join(
            "/scratch/jaidev/HabitatGibson/bevs",
            "partial_occupancy",
            inputs["folder"][0],
            "{}.png".format(inputs["frame_index"][0])
        ))

        # Save unresized GT topview
        save_gt_unresized_topview(
            inputs["filename"],
            gt_64_tp,
            os.path.join(
                opt.out_dir,
                "top",
                "unresized_gt",
                scene,
                "{}.png".format(inputs["frame_index"][0])))

        # Groundtruth topview
        save_gt_topview(
            inputs["filename"],
            inputs[opt.type + "_gt"],
            os.path.join(
                opt.out_dir,
                "top",
                opt.type + "_gt",
                scene,
                "{}.png".format(inputs["frame_index"][0])))
         
        # Save the RGB image!
        save_img(
            inputs["filename"],
            inputs["color"],
            os.path.join(
                opt.out_dir,
                "top",
                'rgb',
                scene,
                 "{}.png".format(inputs["frame_index"][0])))

        pred = np.squeeze(
            torch.argmax(
                outputs["topview"].detach(),
                1).cpu().numpy())
        trans_pred = np.squeeze(
            torch.argmax(
                outputs["transform_topview"].detach(),
                1).cpu().numpy())
        true = np.squeeze(inputs[opt.type + "_gt"].detach().cpu().numpy())
                
        print("IoU", mean_IU(pred, true), "mAP", mean_precision(pred, true))

    # bottom 10 examples
    sorted_iou_index = np.argsort(ious)
    print("Worst examples:")

    for i in range(len(sorted_iou_index[-25:])):
        batch_idx = sorted_iou_index[i]

        inputs = None
        for batch_idx_data, inputs_data in enumerate(test_loader):
            if batch_idx == batch_idx_data:
                inputs = inputs_data
                break

        with torch.no_grad():
            outputs = process_batch(opt, models, inputs)

        scene = inputs['folder'][0]
        # print(inputs["filename"][0])

        # Predicted Topview
        save_topview(
            inputs["filename"],
            outputs["topview"],
            os.path.join(
                opt.out_dir,
                "worst",
                'pred',
                scene,
                "{}.png".format(inputs["frame_index"][0])))

        # Get the unresized GT topview
        gt_64_tp = cv2.imread(os.path.join(
            "/scratch/jaidev/HabitatGibson/bevs",
            "partial_occupancy",
            inputs["folder"][0],
            "{}.png".format(inputs["frame_index"][0])
        ))

        # Save unresized GT topview
        save_gt_unresized_topview(
            inputs["filename"],
            gt_64_tp,
            os.path.join(
                opt.out_dir,
                "worst",
                "unresized_gt",
                scene,
                "{}.png".format(inputs["frame_index"][0])))

        # Groundtruth topview
        save_gt_topview(
            inputs["filename"],
            inputs[opt.type + "_gt"],
            os.path.join(
                opt.out_dir,
                "worst",
                opt.type + "_gt",
                scene,
                "{}.png".format(inputs["frame_index"][0])))
         
        # Save the RGB image!
        save_img(
            inputs["filename"],
            inputs["color"],
            os.path.join(
                opt.out_dir,
                "worst",
                'rgb',
                scene,
                 "{}.png".format(inputs["frame_index"][0])))

        pred = np.squeeze(
            torch.argmax(
                outputs["topview"].detach(),
                1).cpu().numpy())
        trans_pred = np.squeeze(
            torch.argmax(
                outputs["transform_topview"].detach(),
                1).cpu().numpy())
        true = np.squeeze(inputs[opt.type + "_gt"].detach().cpu().numpy())
                
        print("IoU", mean_IU(pred, true), "mAP", mean_precision(pred, true))

def process_batch(opt, models, inputs):
    outputs = {}
    # print(inputs["filename"])
    for key, input_ in inputs.items():
        if key not in ["filename", "folder", "frame_index"]:
            inputs[key] = input_.to("cuda")

    features = models["encoder"](inputs["color"])

    # Cross-view Transformation Module
    transform_feature, retransform_features = models["CycledViewProjection"](features)
    features = models["CrossViewTransformer"](features, transform_feature, retransform_features)

    outputs["topview"] = models["decoder"](features)
    outputs["transform_topview"] = models["transform_decoder"](transform_feature)

    return outputs


def save_topview(idx, tv, name_dest_im):
    tv_np = tv.squeeze().cpu().numpy()
    true_top_view = np.zeros((tv_np.shape[1], tv_np.shape[2]))
    true_top_view[tv_np[1] > tv_np[0]] = 255
    dir_name = os.path.dirname(name_dest_im)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cv2.imwrite(name_dest_im, true_top_view)
    # print("Saved prediction to {}".format(name_dest_im))

def save_gt_topview(idx, tv, name_dest_im):
    tv_np = tv.squeeze().cpu().numpy()
    dir_name = os.path.dirname(name_dest_im)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cv2.imwrite(name_dest_im, tv_np * 255)

def save_gt_unresized_topview(idx, tv, name_dest_im):
    dir_name = os.path.dirname(name_dest_im)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    cv2.imwrite(name_dest_im, tv)

def save_img(idx, tv, name_dest_img):
    tv_np = tv.squeeze().cpu().numpy()
    dir_name = os.path.dirname(name_dest_img)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    cv2.imwrite(name_dest_img, tv_np.transpose((1, 2, 0)) * 255)

if __name__ == "__main__":
    evaluate()
