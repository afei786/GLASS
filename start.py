from main_noneed_gt_1 import main_entry

def train():
    # 模拟命令行参数
    args = [
        "--gpu", "0",
        "--seed", "0",
        "--test", "train",
        "--ckpt_path", "results/models/backbone_0/mvtec_cxjzq1_new/ckpt.pth",
        "net",
        "-b", "wideresnet50",
        "-le", "layer2",
        "-le", "layer3",
        "--pretrain_embed_dimension", "1536",
        "--target_embed_dimension", "1536",
        "--patchsize", "3",
        "--meta_epochs", "200",  # epochs
        "--eval_epochs", "1",
        "--dsc_layers", "2",
        "--dsc_hidden", "1024",
        "--pre_proj", "1",
        "--mining", "1",
        "--noise", "0.015",
        "--radius", "0.75",
        "--p", "0.5",
        "--step", "20",
        "--limit", "392",
        "dataset",
        "--distribution", "0",
        "--mean", "0.5",
        "--std", "0.1",
        "--fg", "0",
        "--rand_aug", "1",
        "--batch_size", "8",
        "--resize", "480",
        "--imagesize", "480",
        "-d", "cxjzq2",
        "mvtec", "/home/fei/data/wuhu/images_seg", "/home/fei/code/glass/dtd/images"
    ]

    # 调用 main_noneed_gt 的主函数
    main_entry(args)




def trian2(class_name):
    # 模拟命令行参数
    args = [
        '--test','train',
        # "--ckpt_path", "results/models/backbone_0/mvtec_cxjzq2/ckpt.pth",
        "net",
        "--meta_epochs", '1',
        "dataset",
        "--batch_size", "8",
        "--resize", "640",
        "--imagesize", "640",
        "-d", class_name,
        "/home/fei/data/wuhu/images_seg"
    ]

    # 调用 main_noneed_gt 的主函数
    main_entry(args)

if __name__ == "__main__":
    import os
    dataset = '/home/fei/data/wuhu/images_seg'
    class_names = os.listdir(dataset)
    for class_name in class_names:
        print(class_name)
        trian2(class_name)