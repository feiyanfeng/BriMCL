import sys
import os
import traceback

import torch
from trainer import Trainer
from tools.utils import task_wrapper
import hydra

from tools.send_email import send_email,get_gpu_memory
from datetime import datetime


@task_wrapper
def train(args, logger):
    try:
        runner = Trainer(args, logger)
        if args.mode=='train':
            if args.model.stage==2: runner.train()
            elif args.model.stage==3: runner.train2()
            else: raise ValueError()
        elif args.mode=='test':
            runner.test()
        else:
            raise NotImplementedError(f'{args.mode} not matched')
        logger.info('ending...\n\n\n')
         
    except Exception:
        # print(traceback.format_exc())
        logger.error(traceback.format_exc())
        # logger.info(f"{args.mode=='train'} , {'runner' in locals()} , {hasattr(runner, 'epoch')}")
        if args.mode=='train' and 'runner' in locals() and hasattr(runner, 'epoch'):  # 全局：globals()
            states = [
                    runner.model.state_dict(),
                    runner.optimizer.state_dict(),
                    runner.epoch
                ]
            if args.model.ema:
                states.append(runner.ema_helper.state_dict())
            torch.save(states, os.path.join(runner.weight_path, f"exception.pth"))
        return (args.name, traceback.format_exc())
    
    return (args.name, 0)

@hydra.main(version_base="1.2", config_path="configs", config_name="main.yaml")
def main(cfg):
    gpu_before = get_gpu_memory()
    time1 = datetime.now()
    kid, feadback = train(cfg)
    time2 = datetime.now()
    if cfg.email:
        send_email(kid, str(time2-time1), feadback, gpu_before)

if __name__ == '__main__':
    sys.exit(main())
