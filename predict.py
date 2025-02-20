from datetime import datetime
import pandas as pd
import os
import logging
import sys
import torch
import warnings
import backbones
from train_noneed_gt.glass import GLASS
import utils

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))

def net():
    dsc_margin = 0.5
    train_backbone = True
    backbone_names = ['wideresnet50'] # list(backbone_names)
    layers_to_extract_from = ["layer2","layer3"]
    pretrain_embed_dimension = 1536
    target_embed_dimension = 1536
    patchsize = 3
    meta_epochs = 640
    eval_epochs = 1
    dsc_layers = 2
    dsc_hidden = 1024
    dsc_margin = 1
    pre_proj = 1
    mining = 1
    noise = 0.015
    radius = 0.75
    p = 0.5
    lr = 0.0001
    svd = 0
    step = 20
    limit = 392

    if len(backbone_names) > 1:
        layers_to_extract_from_coll = []
        for idx in range(len(backbone_names)):
            layers_to_extract_from_coll.append(layers_to_extract_from)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    def get_glass(input_shape, device):
        glasses = []
        for backbone_name, layers_to_extract_from in zip(backbone_names, layers_to_extract_from_coll):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(backbone_name.split("-")[-1])
            backbone = backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            glass_inst = GLASS(device)
            glass_inst.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                meta_epochs=meta_epochs,
                eval_epochs=eval_epochs,
                dsc_layers=dsc_layers,
                dsc_hidden=dsc_hidden,
                dsc_margin=dsc_margin,
                train_backbone=train_backbone,
                pre_proj=pre_proj,
                mining=mining,
                noise=noise,
                radius=radius,
                p=p,
                lr=lr,
                svd=svd,
                step=step,
                limit=limit,
            )
            glasses.append(glass_inst.to(device))
        return glasses

    return get_glass


def dataset(
        data_path,          # 数据集根目录
        subdatasets,        # 子数据集
        ):
    name = "mvtec"
    aug_path = "/home/fei/code/glass/dtd/images"         # 增强数据集目录
    batch_size = 1        # 批大小
    resize = 640      
    imagesize = 640
    num_workers = 1     
    rotate_degrees = 0
    translate = 0
    scale = 0.0
    brightness = 0.0
    contrast = 0.0
    saturation = 0.0
    gray = 0.0
    hflip = 0.0
    vflip = 0.0
    distribution = 0
    mean = 0.5
    std = 0.1
    fg = 0
    rand_aug = 1
    augment = False

    _DATASETS = {"mvtec": ["train_noneed_gt.mvtec", "MVTecDataset"], "visa": ["train_noneed_gt.visa", "VisADataset"],
                 "mpdd": ["train_noneed_gt.mvtec", "MVTecDataset"], "wfdd": ["train_noneed_gt.mvtec", "MVTecDataset"], }
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    def get_dataloaders(seed, test, get_name=name):
        dataloaders = []
        # for subdataset in subdatasets:
        if test == 'test':
            test_dataset = dataset_library.__dict__[dataset_info[1]](
            data_path,
            aug_path,
            classname=subdatasets,
            resize=resize,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.TEST,
            seed=seed,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                prefetch_factor=None,
                pin_memory=True,
            )

            test_dataloader.name = subdatasets

        if test == 'train':
            # 仅加载训练数据集
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                aug_path,
                dataset_name=get_name,
                classname=subdatasets,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                rotate_degrees=rotate_degrees,
                translate=translate,
                brightness_factor=brightness,
                contrast_factor=contrast,
                saturation_factor=saturation,
                gray_p=gray,
                h_flip_p=hflip,
                v_flip_p=vflip,
                scale=scale,
                distribution=distribution,
                mean=mean,
                std=std,
                fg=fg,
                rand_aug=rand_aug,
                augment=augment,
                batch_size=batch_size,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                prefetch_factor=2,
                pin_memory=True,
            )

            train_dataloader.name = subdatasets
            LOGGER.info(f"Dataset {subdatasets.upper():^20}: train={len(train_dataset)}")

            dataloader_dict = {
                "training": train_dataloader,
                "testing": None  # 不加载测试数据
            }
            dataloaders.append(dataloader_dict)
        else:
            train_dataloader = test_dataloader
            LOGGER.info(f"Dataset {subdatasets.upper():^20}: train={0} test={len(test_dataset)}")

            dataloader_dict = {
                "training": train_dataloader,
                "testing": test_dataloader,
            }
            dataloaders.append(dataloader_dict)

        print("\n")
        return dataloaders

    return get_dataloaders

def run(
        ckpt_path,data_path,subdatasets

):
    results_path = "results"
    gpu = "0"
    seed = 0
    log_group = 'group'
    log_project = 'project'
    run_name = 'test'
    test = 'test'
    get_dataloaders = dataset(data_path=data_path,
                              subdatasets=subdatasets)
    get_glass = net()
    methods = {'get_dataloaders':get_dataloaders, 'get_glass':get_glass}

    
    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed, test)

    device = utils.set_torch_device(gpu)

    data = {'Class': [], 'Distribution': [], 'Foreground': []}
    df = pd.DataFrame(data)

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        utils.fix_seeds(seed, device)
        dataset_name = dataloaders["training"].name
        imagesize = dataloaders["training"].dataset.imagesize
        glass_list = methods["get_glass"](imagesize, device)

        LOGGER.info(
            "Selecting dataset [{}] ({}/{}) {}".format(
                dataset_name,
                dataloader_count + 1,
                len(list_of_dataloaders),
                datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            )
        )

        # models_dir = os.path.join(run_save_path, "models")
        # os.makedirs(models_dir, exist_ok=True)
        for i, GLASS in enumerate(glass_list):
            flag = None
              # 设置模型目录
            # GLASS.set_model_dir(os.path.join(models_dir, f"backbone_{i}"), dataset_name)
            # print(os.path.join(models_dir, f"backbone_{i}"), dataset_name)
            # 训练过程
            # if test == 'train':
            #     for i, GLASS in enumerate(glass_list):
            #         if GLASS.backbone.seed is not None:
            #             utils.fix_seeds(GLASS.backbone.seed, device)

            #         GLASS.set_model_dir(os.path.join(models_dir, f"backbone_{i}"), dataset_name)
            #         flag = GLASS.trainer(dataloaders["training"], None, dataset_name)
                    
            #         # 如果 `GLASS.trainer` 返回 int，则记录分布信息
            #         if isinstance(flag, int):
            #             row_dist = {'Class': dataloaders["training"].name, 'Distribution': flag, 'Foreground': flag}
            #             df = pd.concat([df, pd.DataFrame(row_dist, index=[0])])
            if test == 'test':
                # 测试过程（不接收指标返回值）
                if not isinstance(flag, int):
                    GLASS.tester(ckpt_path, dataloaders["testing"], dataset_name)  # 不需要返回指标


    # 保存分布信息到Excel
    # if len(df['Class']) != 0:
    #     os.makedirs('./datasets/excel', exist_ok=True)
    #     xlsx_path = './datasets/excel/' + dataset_name.split('_')[0] + '_distribution.xlsx'
    #     df.to_excel(xlsx_path, index=False)
if __name__ == "__main__":
    class_names = ['cxjzq1', 'cxjzq2','cxjzq3', 'dgnzj', 'dxllt1', 'dxllt2', 'dxllt3','dxllt4', 'fqg1', 'fqg2', 'fql', 'fzzxg', 
        'hxjzq', 'hxlg', 'jdx', 'qylgs', 'qylgx', 'rdx', 'sbqb', 'tyjsq', 'dxllt4', 'xjslh1', 'xjslh2', 'zdp']
    print(len(class_names))
    for class_name in class_names:
        run(ckpt_path=f'/home/fei/data/glass_res/results/models/backbone_0/{class_name}/ckpt.pth',data_path='/home/fei/data/wuhu/new_images2_seg2_resize640',subdatasets=class_name)

    # run(ckpt_path=f'/home/fei/data/glass_res/results/models/backbone_0/mvtec_dxllt2/ckpt.pth',data_path='/home/fei/data/wuhu/new_images2_seg_resize640',subdatasets='dxllt2')