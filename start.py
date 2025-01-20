from main_noneed_gt import main_entry

if __name__ == "__main__":
    # 模拟命令行参数
    args = [
        "--gpu", "0",
        "--seed", "0",
        "--test", "train",
        "net",
        "-b", "wideresnet50",
        "-le", "layer2",
        "-le", "layer3",
        "--pretrain_embed_dimension", "1536",
        "--target_embed_dimension", "1536",
        "--patchsize", "3",
        "--meta_epochs", "640",
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
        "--resize", "640",
        "--imagesize", "640",
        "-d", "cxjzq13",
        "mvtec", "/home/fei/data/new_images_seg", "/home/fei/code/glass/dtd/images"
    ]

    # 调用 main_noneed_gt 的主函数
    main_entry(args)
