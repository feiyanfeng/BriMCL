# import argparse
import numpy as np

import torch.optim as optim
import math
import importlib

from tools.logger import Logger
import torch
import random

class Optimizer:

    def get_optimizer(self, config_optim, parameters):
        
        if config_optim.optimizer == 'Adam':
            return optim.Adam(parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay,
                            betas=(config_optim.beta1, 0.999), amsgrad=config_optim.amsgrad,
                            eps=config_optim.eps)
        elif config_optim.optimizer == 'RMSProp':
            return optim.RMSprop(parameters, lr=config_optim.lr, weight_decay=config_optim.weight_decay)
        elif config_optim.optimizer == 'SGD':
            return optim.SGD(parameters, lr=config_optim.lr, weight_decay=1e-4, momentum=0.9)
        else:
            raise NotImplementedError(
                'Optimizer {} not understood.'.format(config_optim.optimizer))

    def adjust_learning_rate(self, optimizer, epoch, config_opt, config_train):
        """Decay the learning rate with half-cycle cosine after warmup"""
        if epoch < config_opt.warmup_epochs:
            lr = config_opt.lr * epoch / config_train.warmup_epochs
        else:
            lr = config_opt.min_lr + (config_opt.lr - config_opt.min_lr) * 0.5 * \
                (1. + math.cos(math.pi * (epoch - config_train.warmup_epochs) / (
                        config_train.epochs - config_train.warmup_epochs)))
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr
        return lr

def instantiate_from_config(config):
    # return get_obj_from_str(config["target"])(config["params"])
    return get_obj_from_str(config.target)(config.params)


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def resetting(cfg): 
    # # print(cfg)
    # print(cfg.__contains__('device'))
    # print(torch.device("cuda"))
    # cfg.device = torch.device("cuda")
    # print(cfg.device)
    # exit()
    completion = []
    warning = ''

    if not cfg.__contains__('name'): completion.append('name')
    if not cfg.data.params.__contains__('data_name'): completion.append('data.name')
    if not cfg.data.params.__contains__('root'): completion.append('data.root')
    if not cfg.model.__contains__('name'): completion.append('model.name')

    # if not (cfg.data.__contains__('train_list') and cfg.data.__contains__('test_list')) :  # 读取数据时自有分辨
    #     cfg.data.train_list = f'fold{cfg.data.fold}_train'
    #     cfg.data.test_list = f'fold{cfg.data.fold}_test'

    assert len(completion)==0, ', '.join(completion)+' should be given'
    # if cfg.test.batch_size:print('batch_size-----', cfg.test.batch_size)
    
    if cfg.data.params.test_batch_size is None:
        cfg.data.params.test_batch_size = cfg.data.params.train_batch_size*2

    if cfg.log.name is None: cfg.log.name = cfg.name
    if cfg.device is None:
        if torch.cuda.is_available(): cfg.device = "cuda"
        else:
            cfg.device = "cpu"
            warning += 'No GPU available, CPU is used!\n'

    cfg.model.params.device = cfg.device
    if cfg.model.params.__contains__('diffusion'):
        cfg.model.params.diffusion.device = cfg.device
    cfg.data.params.train_name = cfg.name # 保存训练时数据信息时会用到

    return cfg, warning

def print_namespace(args, logger):
    for arg_name, arg_value in args.items(): logger.info(f'### {arg_name}: {arg_value}', '%(message)s')

def set_random_seed(SEED=1234, logger=None):
    if logger: logger.info(f'set random seed {SEED}')

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.enabled = True  # 使用cuDNN库来加速计算，通常默认就是True
    torch.backends.cudnn.benchmark = False # 当benchmark为True时，意味着PyTorch会在程序开始时花费一些时间来寻找最适合当前配置（如输入数据大小和网络结构）的cuDNN算法
    torch.backends.cudnn.deterministic = True # 当deterministic为True时，每次计算的结果都是确定性的。即对于相同的输入，每次运行都会得到相同的输出。但是，设置为True可能会降低计算性能。默认值通常是 False，相同的输入可能在多次运行中得到不同的输出


def task_wrapper(task_func):
    def wrap(cfg):
        cfg, warning = resetting(cfg)
        log_path= f'log/{cfg.data.params.data_name}/{cfg.log.name}.log'
        logger = Logger(log_path)
        if warning: logger.warning(warning)
        print_namespace(cfg._content, logger)
        logger.info(f'{cfg.mode} to {cfg.log.name}\n')
        logger.info(f'\n{cfg.mode} start...\n')
        
        if cfg.random >= 0:
            seed = cfg.random if cfg.random>0 else random.randint(0, 2025)
            set_random_seed(seed, logger)
        return task_func(cfg, logger) # 这个logger是在这里加上去的。所以调用原函数是传的是一个参数，但是原函数的定义里是两个参数。因为原函数外面有个装饰器。
    return wrap
