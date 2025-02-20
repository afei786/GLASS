from datetime import datetime
import pandas as pd
import os
import logging
import sys
import click
import torch
import warnings
import backbones
from train_noneed_gt.glass import GLASS
import utils

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)
LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))

@click.group(chain=True)
@click.option("--results_path", type=str, default="results")
@click.option("--gpu", type=int, default=["0"], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--run_name", type=str, default="test")
@click.option("--test", type=str, default="test")  # 测试或训练
@click.option("--ckpt_path", type=str)  # 如果测试，则指定ckpt_path
@click.option("--save_path", type=str)  # 结果保存路径
def main(**kwargs):
    pass


@main.command("net")
@click.option("--dsc_margin", type=float, default=0.5)
@click.option("--train_backbone", is_flag=True)
@click.option("--backbone_names", "-b", type=str, multiple=True, default=["wideresnet50"])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=["layer2","layer3"])
@click.option("--pretrain_embed_dimension", type=int, default=1536)
@click.option("--target_embed_dimension", type=int, default=1536)
@click.option("--patchsize", type=int, default=3)
@click.option("--meta_epochs", type=int, default=640)  # 最大训练轮次
@click.option("--eval_epochs", type=int, default=1)
@click.option("--dsc_layers", type=int, default=2)
@click.option("--dsc_hidden", type=int, default=1024)
@click.option("--pre_proj", type=int, default=1)
@click.option("--mining", type=int, default=1)
@click.option("--noise", type=float, default=0.015)
@click.option("--radius", type=float, default=0.75)
@click.option("--p", type=float, default=0.5)
@click.option("--lr", type=float, default=0.0001)
@click.option("--svd", type=int, default=0)
@click.option("--step", type=int, default=20)
@click.option("--limit", type=int, default=392)
def net(
        backbone_names,
        layers_to_extract_from,
        pretrain_embed_dimension,
        target_embed_dimension,
        patchsize,
        meta_epochs,
        eval_epochs,
        dsc_layers,
        dsc_hidden,
        dsc_margin,
        train_backbone,
        pre_proj,
        mining,
        noise,
        radius,
        p,
        lr,
        svd,
        step,
        limit,
):
    backbone_names = list(backbone_names)
    print('backbone_names',backbone_names)
    print('layers_to_extract_from',layers_to_extract_from)
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

    return "get_glass", get_glass


@main.command("dataset")
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))  # 数据集根目录
@click.argument("name", type=str, default="mvtec")  # 数据集类型名
@click.argument("aug_path",  default="/home/fei/code/glass/dtd/images", type=click.Path(exists=True, file_okay=False))  # 增强数据集路径
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)  # 数据集子集名 / 类别名
@click.option("--batch_size", default=8, type=int, show_default=True)  # 训练/测试时 batch_size
@click.option("--num_workers", default=1, type=int, show_default=True)
@click.option("--resize", default=288, type=int, show_default=True)
@click.option("--imagesize", default=288, type=int, show_default=True)
@click.option("--rotate_degrees", default=0, type=int)
@click.option("--translate", default=0, type=float)
@click.option("--scale", default=0.0, type=float)
@click.option("--brightness", default=0.0, type=float)
@click.option("--contrast", default=0.0, type=float)
@click.option("--saturation", default=0.0, type=float)
@click.option("--gray", default=0.0, type=float)
@click.option("--hflip", default=0.0, type=float)
@click.option("--vflip", default=0.0, type=float)
@click.option("--distribution", default=0, type=int)
@click.option("--mean", default=0.5, type=float)
@click.option("--std", default=0.1, type=float)
@click.option("--fg", default=0, type=int)
@click.option("--rand_aug", default=1, type=int)
@click.option("--augment", is_flag=True)
def dataset(
        name,
        data_path,
        aug_path,
        subdatasets,
        batch_size,
        resize,
        imagesize,
        num_workers,
        rotate_degrees,
        translate,
        scale,
        brightness,
        contrast,
        saturation,
        gray,
        hflip,
        vflip,
        distribution,
        mean,
        std,
        fg,
        rand_aug,
        augment,
):
    _DATASETS = {"mvtec": ["train_noneed_gt.mvtec", "MVTecDataset"], "visa": ["train_noneed_gt.visa", "VisADataset"],
                 "mpdd": ["train_noneed_gt.mvtec", "MVTecDataset"], "wfdd": ["train_noneed_gt.mvtec", "MVTecDataset"], }
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])
    print('augment',augment)
    def get_dataloaders(seed, test, get_name=name):
        dataloaders = []
        for subdataset in subdatasets:
            if test == 'test':
                test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                aug_path,
                classname=subdataset,
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

                test_dataloader.name = get_name + "_" + subdataset

            if test == 'train':
                # 仅加载训练数据集
                train_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    aug_path,
                    dataset_name=get_name,
                    classname=subdataset,
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

                train_dataloader.name = get_name + "_" + subdataset
                LOGGER.info(f"Dataset {subdataset.upper():^20}: train={len(train_dataset)}")

                dataloader_dict = {
                    "training": train_dataloader,
                    "testing": None  # 不加载测试数据
                }
                dataloaders.append(dataloader_dict)
            else:
                train_dataloader = test_dataloader
                LOGGER.info(f"Dataset {subdataset.upper():^20}: train={0} test={len(test_dataset)}")

                dataloader_dict = {
                    "training": train_dataloader,
                    "testing": test_dataloader,
                }
                dataloaders.append(dataloader_dict)

        print("\n")
        return dataloaders

    return "get_dataloaders", get_dataloaders


@main.result_callback()
def run(
        methods,
        results_path,
        gpu,
        seed,
        log_group,
        log_project,
        run_name,
        test,
        ckpt_path,
        save_path,
):
    methods = {key: item for (key, item) in methods}
    print('results_path', results_path)
    run_save_path = utils.create_storage_folder(
        results_path, log_project, log_group, run_name, mode="overwrite"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed, test)
    print('list_of_dataloaders', list_of_dataloaders)
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

        models_dir = os.path.join(run_save_path, "models")
        os.makedirs(models_dir, exist_ok=True)
        if test == 'test':
            flag = None
              # 设置模型目录
            GLASS = glass_list[0]
            # GLASS.set_model_dir(os.path.join(models_dir, f"backbone_0"), dataset_name)
            # 测试过程（不接收指标返回值）
            if not isinstance(flag, int):
                GLASS.tester(ckpt_path, dataloaders["testing"], save_path)  # 不需要返回指标

        # for i, GLASS in enumerate(glass_list):
        #     flag = None
        #       # 设置模型目录
        #     GLASS.set_model_dir(os.path.join(models_dir, f"backbone_{i}"), dataset_name)
        #     print(os.path.join(models_dir, f"backbone_{i}"), dataset_name)
        #     # 训练过程
        #     if test == 'train':
        #         for i, GLASS in enumerate(glass_list):
        #             if GLASS.backbone.seed is not None:
        #                 utils.fix_seeds(GLASS.backbone.seed, device)

        #             GLASS.set_model_dir(os.path.join(models_dir, f"backbone_{i}"), dataset_name)
        #             flag = GLASS.trainer(dataloaders["training"], None, dataset_name)
                    
        #             # 如果 `GLASS.trainer` 返回 int，则记录分布信息
        #             if isinstance(flag, int):
        #                 row_dist = {'Class': dataloaders["training"].name, 'Distribution': flag, 'Foreground': flag}
        #                 df = pd.concat([df, pd.DataFrame(row_dist, index=[0])])
        #     elif test == 'test':
        #         # 测试过程（不接收指标返回值）
        #         if not isinstance(flag, int):
        #             GLASS.tester(ckpt_path, dataloaders["testing"], dataset_name)  # 不需要返回指标


    # 保存分布信息到Excel
    if len(df['Class']) != 0:
        os.makedirs('./datasets/excel', exist_ok=True)
        xlsx_path = './datasets/excel/' + dataset_name.split('_')[0] + '_distribution.xlsx'
        df.to_excel(xlsx_path, index=False)

def main_entry(args=None):
    """
    This function serves as an entry point for external calls.
    :param args: List of arguments as you would pass via the command line.
    """
    if args is not None:
        sys.argv = [sys.argv[0]] + args
    main()

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    logging.basicConfig(level=logging.INFO)
    LOGGER = logging.getLogger(__name__)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
