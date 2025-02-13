from loss import FocalLoss
from collections import OrderedDict
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from model import Discriminator, Projection, PatchMaker

import numpy as np
import pandas as pd
import torch.nn.functional as F

import logging
import os
import math
import torch
import tqdm
import common
import metrics
import cv2
import utils
import glob
import shutil
from PIL import Image
import time

LOGGER = logging.getLogger(__name__)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class GLASS(torch.nn.Module):
    def __init__(self, device):
        super(GLASS, self).__init__()
        self.device = device

    def load(
            self,
            backbone,
            layers_to_extract_from,
            device,
            input_shape,
            pretrain_embed_dimension,
            target_embed_dimension,
            patchsize=3,
            patchstride=1,
            meta_epochs=640,
            eval_epochs=1,
            dsc_layers=2,
            dsc_hidden=1024,
            dsc_margin=0.5,
            train_backbone=False,
            pre_proj=1,
            mining=1,
            noise=0.015,
            radius=0.75,
            p=0.5,
            lr=0.0001,
            svd=0,
            step=20,
            limit=392,
            **kwargs,
    ):

        self.backbone = backbone.to(device)
        self.layers_to_extract_from = layers_to_extract_from
        self.input_shape = input_shape
        self.device = device

        self.forward_modules = torch.nn.ModuleDict({})
        feature_aggregator = common.NetworkFeatureAggregator(
            self.backbone, self.layers_to_extract_from, self.device, train_backbone
        )
        feature_dimensions = feature_aggregator.feature_dimensions(input_shape)
        self.forward_modules["feature_aggregator"] = feature_aggregator

        preprocessing = common.Preprocessing(feature_dimensions, pretrain_embed_dimension)
        self.forward_modules["preprocessing"] = preprocessing
        self.target_embed_dimension = target_embed_dimension
        preadapt_aggregator = common.Aggregator(target_dim=target_embed_dimension)
        preadapt_aggregator.to(self.device)
        self.forward_modules["preadapt_aggregator"] = preadapt_aggregator

        self.meta_epochs = meta_epochs
        self.lr = lr
        self.train_backbone = train_backbone
        if self.train_backbone:
            self.backbone_opt = torch.optim.AdamW(self.forward_modules["feature_aggregator"].backbone.parameters(), lr)

        self.pre_proj = pre_proj
        if self.pre_proj > 0:
            self.pre_projection = Projection(self.target_embed_dimension, self.target_embed_dimension, pre_proj)
            self.pre_projection.to(self.device)
            self.proj_opt = torch.optim.Adam(self.pre_projection.parameters(), lr, weight_decay=1e-5)

        self.eval_epochs = eval_epochs
        self.dsc_layers = dsc_layers
        self.dsc_hidden = dsc_hidden
        self.discriminator = Discriminator(self.target_embed_dimension, n_layers=dsc_layers, hidden=dsc_hidden)
        self.discriminator.to(self.device)
        self.dsc_opt = torch.optim.AdamW(self.discriminator.parameters(), lr=lr * 2)
        self.dsc_margin = dsc_margin

        self.c = torch.tensor(0)
        self.c_ = torch.tensor(0)
        self.p = p
        self.radius = radius
        self.mining = mining
        self.noise = noise
        self.svd = svd
        self.step = step
        self.limit = limit
        self.distribution = 0
        self.focal_loss = FocalLoss()

        self.patch_maker = PatchMaker(patchsize, stride=patchstride)
        self.anomaly_segmentor = common.RescaleSegmentor(device=self.device, target_size=input_shape[-2:])
        self.model_dir = ""
        self.dataset_name = ""
        self.logger = None

    def set_model_dir(self, model_dir, dataset_name):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.model_dir, dataset_name)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tb_dir = os.path.join(self.ckpt_dir, "tb")
        os.makedirs(self.tb_dir, exist_ok=True)
        self.logger = TBWrapper(self.tb_dir)

    def _embed(self, images, detach=True, provide_patch_shapes=False, evaluation=False):
        """Returns feature embeddings for images."""
        if not evaluation and self.train_backbone:
            self.forward_modules["feature_aggregator"].train()
            features = self.forward_modules["feature_aggregator"](images, eval=evaluation)
        else:
            self.forward_modules["feature_aggregator"].eval()
            with torch.no_grad():
                features = self.forward_modules["feature_aggregator"](images)

        features = [features[layer] for layer in self.layers_to_extract_from]

        for i, feat in enumerate(features):
            if len(feat.shape) == 3:
                B, L, C = feat.shape
                features[i] = feat.reshape(B, int(math.sqrt(L)), int(math.sqrt(L)), C).permute(0, 3, 1, 2)

        features = [self.patch_maker.patchify(x, return_spatial_info=True) for x in features]
        patch_shapes = [x[1] for x in features]
        patch_features = [x[0] for x in features]
        ref_num_patches = patch_shapes[0]

        for i in range(1, len(patch_features)):
            _features = patch_features[i]
            patch_dims = patch_shapes[i]

            _features = _features.reshape(
                _features.shape[0], patch_dims[0], patch_dims[1], *_features.shape[2:]
            )
            _features = _features.permute(0, -3, -2, -1, 1, 2)
            perm_base_shape = _features.shape
            _features = _features.reshape(-1, *_features.shape[-2:])
            _features = F.interpolate(
                _features.unsqueeze(1),
                size=(ref_num_patches[0], ref_num_patches[1]),
                mode="bilinear",
                align_corners=False,
            )
            _features = _features.squeeze(1)
            _features = _features.reshape(
                *perm_base_shape[:-2], ref_num_patches[0], ref_num_patches[1]
            )
            _features = _features.permute(0, -2, -1, 1, 2, 3)
            _features = _features.reshape(len(_features), -1, *_features.shape[-3:])
            patch_features[i] = _features

        patch_features = [x.reshape(-1, *x.shape[-3:]) for x in patch_features]
        patch_features = self.forward_modules["preprocessing"](patch_features)
        patch_features = self.forward_modules["preadapt_aggregator"](patch_features)

        return patch_features, patch_shapes

    def trainer(self, training_data, _, name):
        state_dict = {}
        ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
        if len(ckpt_path) != 0:
            LOGGER.info("Start testing, ckpt file found!")
            return 0., 0., 0., 0., 0., -1.

        def update_state_dict():
            state_dict["discriminator"] = OrderedDict({
                k: v.detach().cpu()
                for k, v in self.discriminator.state_dict().items()})
            if self.pre_proj > 0:
                state_dict["pre_projection"] = OrderedDict({
                    k: v.detach().cpu()
                    for k, v in self.pre_projection.state_dict().items()})

        # Initialize center c based on good quality images in training_data
        with torch.no_grad():
            for i, data in enumerate(training_data):
                img = data["image"].to(torch.float).to(self.device)
                if self.pre_proj > 0:
                    outputs = self.pre_projection(self._embed(img, evaluation=False)[0])
                    outputs = outputs[0] if len(outputs) == 2 else outputs
                else:
                    outputs = self._embed(img, evaluation=False)[0]
                outputs = outputs.reshape(img.shape[0], -1, outputs.shape[-1])

                batch_mean = torch.mean(outputs, dim=0)
                if i == 0:
                    self.c = batch_mean
                else:
                    self.c += batch_mean
            self.c /= len(training_data)

        pbar = tqdm.tqdm(range(self.meta_epochs), unit='epoch')
        for i_epoch in pbar:
            pbar_str, pt, pf = self._train_discriminator(training_data, i_epoch, pbar)
            update_state_dict()

            # Regularly save state dictionary during training
            ckpt_path_save = os.path.join(self.ckpt_dir, "ckpt.pth")
            torch.save(state_dict, ckpt_path_save)
        return

    def _train_discriminator(self, input_data, cur_epoch, pbar):
        self.forward_modules.eval()
        if self.pre_proj > 0:
            self.pre_projection.train()
        self.discriminator.train()

        all_loss, all_p_true, all_p_fake = [], [], []
        sample_num = 0

        for i_iter, data_item in enumerate(input_data):
            self.dsc_opt.zero_grad()
            if self.pre_proj > 0:
                self.proj_opt.zero_grad()

            img = data_item["image"].to(torch.float).to(self.device)
            if self.pre_proj > 0:
                true_feats = self.pre_projection(self._embed(img, evaluation=False)[0])
                true_feats = true_feats[0] if len(true_feats) == 2 else true_feats
            else:
                true_feats = self._embed(img, evaluation=False)[0]
                true_feats.requires_grad = True

            # Adding noise and calculating features for augmentation
            noise = torch.normal(0, self.noise, true_feats.shape).to(self.device)
            gaus_feats = true_feats + noise

            scores = self.discriminator(torch.cat([true_feats, gaus_feats]))
            true_scores = scores[:len(true_feats)]
            gaus_scores = scores[len(true_feats):]

            # Loss calculation for the discriminator
            true_loss = torch.nn.BCELoss()(true_scores, torch.zeros_like(true_scores))
            gaus_loss = torch.nn.BCELoss()(gaus_scores, torch.ones_like(gaus_scores))
            loss = true_loss + gaus_loss
            loss.backward()

            # Optimizer step
            if self.pre_proj > 0:
                self.proj_opt.step()
            if self.train_backbone:
                self.backbone_opt.step()
            self.dsc_opt.step()

            # Track loss and other metrics
            p_true = ((true_scores < self.dsc_margin).sum() / true_scores.numel()).item()
            p_fake = ((gaus_scores >= self.dsc_margin).sum() / gaus_scores.numel()).item()
            all_loss.append(loss.detach().cpu().item())
            all_p_true.append(p_true)
            all_p_fake.append(p_fake)

            # Log metrics to tensorboard
            self.logger.logger.add_scalar(f"p_true", p_true, self.logger.g_iter)
            self.logger.logger.add_scalar(f"p_fake", p_fake, self.logger.g_iter)
            self.logger.logger.add_scalar("loss", loss, self.logger.g_iter)
            self.logger.step()

            # Update progress bar
            sample_num += img.shape[0]
            pbar_str = f"epoch:{cur_epoch} loss:{np.mean(all_loss):.2e} pt:{np.mean(all_p_true) * 100:.2f} pf:{np.mean(all_p_fake) * 100:.2f} sample:{sample_num}"
            pbar.set_description_str(pbar_str)

            if sample_num > self.limit:
                break

        return pbar_str, np.mean(all_p_true), np.mean(all_p_fake)

    # def tester(self, test_data, name):
    #     # 检查是否存在模型检查点
    #     # ckpt_path = glob.glob(self.ckpt_dir + '/ckpt_best*')
    #     ckpt_path = glob.glob(self.ckpt_dir + '/ckpt')
    #     if len(ckpt_path) != 0:
    #         # 加载模型检查点
    #         state_dict = torch.load(ckpt_path[0], map_location=self.device)
    #         if 'discriminator' in state_dict:
    #             self.discriminator.load_state_dict(state_dict['discriminator'])
    #             if "pre_projection" in state_dict:
    #                 self.pre_projection.load_state_dict(state_dict["pre_projection"])
    #         else:
    #             self.load_state_dict(state_dict, strict=False)

    #         # 获取预测结果
    #         images, scores, segmentations, labels_gt, masks_gt = self.predict(test_data)

    #         # 只保存结果图像，不返回指标
    #         self._evaluate(images, scores, segmentations, labels_gt, masks_gt, name, path='eval')
    #     else:
    #         LOGGER.info("No ckpt file found!")

    #     # 不返回任何指标
    #     return
    
    def tester(self, ckpt_path, test_data, name):
        # 直接指定 ckpt.pth 文件路径
        # ckpt_path = os.path.join(self.ckpt_dir, 'ckpt.pth')
        
        # 检查是否存在模型检查点
        if os.path.exists(ckpt_path):
            # 加载模型检查点
            state_dict = torch.load(ckpt_path, map_location=self.device)

            if 'discriminator' in state_dict:
                print('1')
                self.discriminator.load_state_dict(state_dict['discriminator'])
                if "pre_projection" in state_dict:
                    print('2')
                    self.pre_projection.load_state_dict(state_dict["pre_projection"])
            else:
                self.load_state_dict(state_dict, strict=False)
                print('3')

            # 获取预测结果
            images, scores, segmentations, labels_gt, masks_gt, img_paths = self.predict(test_data)

            # 只保存结果图像，不返回指标
            self._evaluate(images, scores, segmentations, labels_gt, masks_gt, name, img_paths, path='eval')
        else:
            LOGGER.info("No ckpt file found!")

        # 不返回任何指标
        return

    def _evaluate(self, images, scores, segmentations, labels_gt, masks_gt, name, img_paths, path='eval'):
        # 规范化分数（去掉 image_auroc 等计算）
        scores = np.squeeze(np.array(scores))
        img_min_scores = min(scores)
        img_max_scores = max(scores)
        norm_scores = (scores - img_min_scores) / (img_max_scores - img_min_scores + 1e-10)

        if len(masks_gt) > 0:
            # 规范化分割分数
            segmentations = np.array(segmentations)
            min_scores = np.min(segmentations)
            max_scores = np.max(segmentations)
            norm_segmentations = (segmentations - min_scores) / (max_scores - min_scores + 1e-10)

        # 保存预测图像
        defects = np.array(images)
        targets = np.array(masks_gt)
        for i in range(len(defects)):
            defect = utils.torch_format_2_numpy_img(defects[i])
            # target = utils.torch_format_2_numpy_img(targets[i])

            mask = cv2.cvtColor(cv2.resize(norm_segmentations[i], (defect.shape[1], defect.shape[0])),
                                cv2.COLOR_GRAY2BGR)
            mask = (mask * 255).astype('uint8')
            mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)

            # img_up = np.hstack([defect, target, mask])
            img_up = np.hstack([defect, mask])

            img_up = cv2.resize(img_up, (640 * 2, 640))
            full_path = os.path.join('./results/', path, name)
            # print(full_path)
            utils.del_remake_dir(full_path, del_flag=False)

            img_name = os.path.basename(img_paths[i])

            cv2.imwrite(os.path.join(full_path, img_name), img_up)
            # cv2.imwrite(os.path.join(full_path, f'mask_{img_name}'), mask)
        return  # 不返回任何指标

    def predict(self, test_dataloader):
        """This function provides anomaly scores/maps for full dataloaders."""
        self.forward_modules.eval()

        img_paths = []
        images = []
        scores = []
        masks = []
        labels_gt = []
        masks_gt = []
        t1 = time.time()
        # with tqdm.tqdm(test_dataloader, desc="Inferring...", leave=True, unit='batch') as data_iterator:

        for data in test_dataloader:
            if isinstance(data, dict):
                labels_gt.extend(data["is_anomaly"].numpy().tolist())
                if data.get("mask_gt", None) is not None:
                    masks_gt.extend(data["mask_gt"].numpy().tolist())
                image = data["image"]
                images.extend(image.numpy().tolist())
                img_paths.extend(data["image_path"])
            _scores, _masks = self._predict(image)
            for score, mask in zip(_scores, _masks):
                scores.append(score)
                masks.append(mask)

        t2 = time.time()
        total_time = t2 - t1 
        print(f"Total inference time: {total_time:.2f} seconds")
        return images, scores, masks, labels_gt, masks_gt, img_paths

    def _predict(self, img):
        """Infer score and mask for a batch of images."""
        img = img.to(torch.float).to(self.device)
        self.forward_modules.eval()

        if self.pre_proj > 0:
            self.pre_projection.eval()
        self.discriminator.eval()

        with torch.no_grad():

            patch_features, patch_shapes = self._embed(img, provide_patch_shapes=True, evaluation=True)
            if self.pre_proj > 0:
                patch_features = self.pre_projection(patch_features)
                patch_features = patch_features[0] if len(patch_features) == 2 else patch_features

            patch_scores = image_scores = self.discriminator(patch_features)
            patch_scores = self.patch_maker.unpatch_scores(patch_scores, batchsize=img.shape[0])
            scales = patch_shapes[0]
            patch_scores = patch_scores.reshape(img.shape[0], scales[0], scales[1])
            masks = self.anomaly_segmentor.convert_to_segmentation(patch_scores)

            image_scores = self.patch_maker.unpatch_scores(image_scores, batchsize=img.shape[0])
            image_scores = self.patch_maker.score(image_scores)
            if isinstance(image_scores, torch.Tensor):
                image_scores = image_scores.cpu().numpy()

        return list(image_scores), list(masks)
