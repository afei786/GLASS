from main_noneed_gt_1 import main_entry
import trace

def pred():
    # 模拟命令行参数
    args = [
        "--gpu", "0",
        "--seed", "0",
        "--test", "test",
        "--ckpt_path", "results/models/backbone_0/mvtec_dxllt1_new/ckpt.pth",
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
        "--batch_size", "1",
        "--resize", "256",
        "--imagesize", "256",
        "-d", "dxllt1_new",
        "mvtec", "/home/fei/code/jm_wh3/images_seg_dxllt_new", "/home/fei/code/glass/dtd/images"
    ]

    # 调用 main_noneed_gt 的主函数
    main_entry(args)

def pred1():
    # 模拟命令行参数
    args = [
        '--test','test',
        "--ckpt_path", "results/models/backbone_0/mvtec_dxllt1_new/ckpt.pth",
        "net",
        "dataset",
        "-d", "dxllt1_new2",
        "/home/fei/code/jm_wh3/images_seg_dxllt_new"
    ]

    # 调用 main_noneed_gt 的主函数
    main_entry(args)


if __name__ == "__main__":
    pred1()