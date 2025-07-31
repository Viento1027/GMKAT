import os
import math
import argparse
import shutil
import torch
import numpy as np
import random
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
import json
from ptflops import get_model_complexity_info
from my_dataset import GisDataSet
from utils import read_split_gisdata, train_one_epoch, evaluate, calculate_metrics
from sklearn.metrics import roc_auc_score

from GMKAT import gmkat_base as create_model_gmkat



def set_random_seeds(seed=1027):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    if opt.tensorboard:
        # tensorboard数据存放路径
        log_path = os.path.join('./tb_results/tensorboard', args.model)
        print('Start Tensorboard with "tensorboard --logdir={}"'.format(log_path))

        if os.path.exists(log_path) is False:
            os.makedirs(log_path)
            print("tensorboard log save in {}".format(log_path))
        else:
            shutil.rmtree(log_path)

        tb_writer = SummaryWriter(log_path)

    train_images_path, train_images_label, val_images_path, val_images_label = read_split_gisdata(args.data_path)
    # 实例化训练数据集
    train_dataset = GisDataSet(images_path=train_images_path,
                               images_class=train_images_label,
                               transform=None)
    # 实例化验证数据集
    val_dataset = GisDataSet(images_path=val_images_path,
                             images_class=val_images_label,
                             transform=None)

    batch_size = args.batch_size
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    print('Using {} dataloader workers every process'.format(nw))
    # 使用 DataLoader 将加载的数据集处理成批量（batch）加载模式
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=True,
                                               num_workers=nw,
                                               collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=batch_size,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=nw,
                                             collate_fn=val_dataset.collate_fn)
    model = create_model_gmkat().to(device)
    print(model)
    print(".........................")

    macs, params = get_model_complexity_info(model, (9, 16, 16), as_strings=False, print_per_layer_stat=False)
    init_img = torch.zeros((1, 9, 16, 16), device=device)
    tb_writer.add_graph(model, init_img)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=args.lr, momentum=0.9, weight_decay=5E-5)
    lf = lambda x: ((1 + math.cos(x * math.pi / args.epochs)) / 2) * (1 - args.lrf) + args.lrf  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    best_score = 0.

    save_path = os.path.join(os.getcwd(), 'results', args.model)
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)
    log_file = os.path.join(save_path, 'training_log.json')
    performance_file = os.path.join(save_path, 'performance.json')
    log_data = []
    early_stop_counter = 0
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Train
        train_loss, train_labels, train_preds, train_probs = train_one_epoch(model=model, optimizer=optimizer,
                                                                             data_loader=train_loader, device=device,
                                                                             epoch=epoch)
        # Calculate metrics
        train_acc, train_precision, train_recall, train_f1 = calculate_metrics(train_labels, train_preds)

        scheduler.step()
        torch.cuda.empty_cache()

        # Validate
        val_loss, val_labels, val_preds, val_probs = evaluate(model=model, data_loader=val_loader, device=device,
                                                              epoch=epoch)
        # Calculate metrics
        val_acc, val_precision, val_recall, val_f1 = calculate_metrics(val_labels, val_preds)
        # Calculate ROC and AUC
        val_roc_auc = roc_auc_score(val_labels, val_probs)

        # Calculate weighted score
        weighted_score = (0.4 * val_acc + 0.2 * val_precision + 0.2 * val_recall + 0.2 * val_f1)

        # Log results
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'train_precision': train_precision,
            'train_recall': train_recall,
            'train_f1': train_f1,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_precision': val_precision,
            'val_recall': val_recall,
            'val_f1': val_f1,
            'val_roc_auc': val_roc_auc
        }

        log_data.append(log_entry)
        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=4)

        if args.tensorboard:
            tags = ["train_loss", "train_acc", "train_precision", "train_recall", "train_f1", "val_loss", "val_acc",
                    "val_precision", "val_recall", "val_f1", "val_roc_auc", "learning_rate"]
            tb_writer.add_scalar(tags[0], train_loss, epoch)
            tb_writer.add_scalar(tags[1], train_acc, epoch)
            tb_writer.add_scalar(tags[2], train_precision, epoch)
            tb_writer.add_scalar(tags[3], train_recall, epoch)
            tb_writer.add_scalar(tags[4], train_f1, epoch)
            tb_writer.add_scalar(tags[5], val_loss, epoch)
            tb_writer.add_scalar(tags[6], val_acc, epoch)
            tb_writer.add_scalar(tags[7], val_precision, epoch)
            tb_writer.add_scalar(tags[8], val_recall, epoch)
            tb_writer.add_scalar(tags[9], val_f1, epoch)
            tb_writer.add_scalar(tags[10], val_roc_auc, epoch)
            tb_writer.add_scalar(tags[11], optimizer.param_groups[0]["lr"], epoch)

        # Log extra metrics for best model
        if weighted_score > best_score:
            best_score = weighted_score
            # Convert val_labels and val_preds to list
            val_labels_list = [int(label) for label in val_labels]
            val_probs_list = [float(prob) for prob in val_probs]
            performance_entry = {
                'params': params,
                'macs': macs,
                'accuracy': val_acc,
                'precision': val_precision,
                'recall': val_recall,
                'f1': val_f1,
                'roc_auc': val_roc_auc,
                'val_labels': val_labels_list,
                'val_probs': val_probs_list
            }
            with open(performance_file, 'w') as f:
                json.dump(performance_entry, f, indent=4)
            torch.save(model.state_dict(), save_path + f"/{args.model}_weights.pth")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= args.early_stop:
                print(f"Early stopping triggered after {early_stop_counter} epochs without improvement.")
                break

        print("finish epoch:" + str(epoch))

        # Clean up
        if args.tensorboard:
            tb_writer.close()

        if os.path.exists(log_path):
            shutil.rmtree(log_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=20)
    parser.add_argument('--lr', type=float, default=0.0005)
    parser.add_argument('--lrf', type=float, default=0.001)
    parser.add_argument('--early_stop', type=int, default=5)
    parser.add_argument('--data-path', type=str,
                        default="G:/FSM/data/flood_24")
    parser.add_argument('--model', type=str, default="gmkat_24x24", help=' select a model for training')
    parser.add_argument('--tensorboard', default=True, action='store_true', help=' use tensorboard for visualization')
    # 预训练权重路径，如果不想载入就设置为空字符
    parser.add_argument('--weights', type=str, default='',
                        help='initial weights path')
    parser.add_argument('--device', default='cuda:0', help='device id (i.e. 0 or 0,1 or cpu)')

    opt = parser.parse_args()

    main(opt)
