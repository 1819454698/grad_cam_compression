import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from utils.train import base_train,attri_train
from models.vgg import vgg19_bn,vgg16_bn
from base_config import get_baseconfig_by_epoch
from models.resnet import resnet18,resnet34,resnet50,resnet101
base_architecture_to_features = {
                                 'vgg16':vgg16_bn,
                                 'vgg19': vgg19_bn,
                                 'resnet18':resnet18,
                                 'resnet34': resnet34,
                                 'resnet50': resnet50,
                                 'resnet101': resnet101}
def train_pipline(init_weight,base_train_config,attri_train_config,ratio):
    model=base_architecture_to_features[base_train_config.network_type]()
    if init_weight==None:
        base_train(model,base_train_config)
    else:
        model.load_state_dict(torch.load(init_weight))
    attri_train(model,attri_train_config,ratio)


if __name__ == '__main__':
    network_type = 'vgg16'
    dataset_name = 'cifar10'
    batch_size = 32
    base_log_dir = 'gsm_exps/{}_base_train'.format(network_type)
    gsm_log_dir = 'gsm_exps/{}_gsm'.format(network_type)
    base_train_config = get_baseconfig_by_epoch(network_type=network_type, dataset_name=dataset_name, dataset_subset='train',
                                                global_batch_size=batch_size, num_node=1, weight_decay=1e-4, optimizer_type='sgd',
                                                momentum=0.9, max_epochs=500, base_lr=0.1, lr_epoch_boundaries=[100, 200, 300, 400],
                                                lr_decay_factor=0.1, linear_final_lr=None, warmup_epochs=5, warmup_method='linear',
                                                warmup_factor=0, ckpt_iter_period=40000, tb_iter_period=100,
                                                output_dir=base_log_dir, tb_dir=base_log_dir, save_weights=None,
                                                weight_decay_bias=0,
                                                val_epoch_period=2)

    attri_train_config = get_baseconfig_by_epoch(network_type=network_type, dataset_name=dataset_name, dataset_subset='train',
                                                global_batch_size=batch_size, num_node=1, weight_decay=1e-3, optimizer_type='sgd',
                                                momentum=0.98, max_epochs=600, base_lr=5e-3, lr_epoch_boundaries=[400, 500],     # Note this line
                                                lr_decay_factor=0.1, linear_final_lr=None, warmup_epochs=5, warmup_method='linear',
                                                warmup_factor=0, ckpt_iter_period=40000, tb_iter_period=100,
                                                output_dir=gsm_log_dir, tb_dir=gsm_log_dir, save_weights=None,
                                                weight_decay_bias=1e-3,
                                                val_epoch_period=2)
    train_pipline(init_weight="/home/explainable/src/Grad-cam-compress-cifar/base_model/vgg16/90.6501.pth", base_train_config=base_train_config, attri_train_config=attri_train_config, ratio=0.30)

