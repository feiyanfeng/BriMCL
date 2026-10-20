import os
import numpy as np
# import numpy as np
import torch
# import torchio as tio

from tools.ema import EMA
# import torch.nn.functional as F


from tools.data.data import *
from tools.evaluation import  Evaluation
from tools.utils import Optimizer, instantiate_from_config


import SimpleITK as sitk
# class Namespace:  # 命名空间
#     def __init__(self, **kwargs):  
#         self.__dict__.update(kwargs)

class Trainer:
    def __init__(self, args, logger):
        self.logger = logger
        self.device = args.device
        
        # model
        self.model = instantiate_from_config(args.model).to(self.device)
        if args.model.view: logger.info(self.model, '%(message)s')
        
        #  save weight
        self.weight_path = f'{args.training.weigth_path}/{args.name}'
        os.makedirs(self.weight_path, exist_ok=True)

        self.args = args
        self.get_load()

    def get_load(self,):
        load = self.args.load
        if load and isinstance(load, int): 
            self.args.load = os.path.join(self.weight_path, f"epoch{load}_ckpt.pth")
            return
        if load == 'best':
            self.args.load = os.path.join(self.weight_path, "best_ckpt.pth")
            return
        if load and '/' not in load:
            self.args.load = os.path.join(self.weight_path, f"{load}.pth")
            return




    @torch.no_grad()
    def evaluate(self, loader, print_probability=False):
        model = self.model
        model.eval()
        predict = []
        groundtruth = []
        name_list = []
        e = Evaluation(self.logger, metrics=self.args.evaluation.metrics)
        # if save: os.makedirs(save, exist_ok=True)
        for batch in loader:
            if isinstance(batch, dict):
                names, image, label = batch['name'], batch['t1'][tio.DATA], batch['label']
                image = image.permute(0,1,4,2,3)
            else:
                # names, image, label = batch
                names, image, label, c = batch
            image = image.to(self.device)
            c = c.to(self.device)
            #print('----------------------', image.shape)
            pre = model.predict(image, c)
            predict.extend(pre.cpu().tolist())
            groundtruth.extend(label.cpu().tolist())
            name_list.extend(list(names))
            
        return e.view(name_list, predict, groundtruth, print_probability)
    
    @torch.no_grad()
    def evaluate_test(self, loader, print_probability=False): # 这个会额外打印概率
        model = self.model
        model.eval()
        predict = []
        groundtruth = []
        name_list = []
        e = Evaluation(self.logger, metrics=self.args.evaluation.metrics)
        # if save: os.makedirs(save, exist_ok=True)
        for batch in loader:
            if isinstance(batch, dict):
                names, image, label = batch['name'], batch['t1'][tio.DATA], batch['label']
                image = image.permute(0,1,4,2,3)
            else:
                # names, image, label = batch
                names, image, label, c = batch
            image = image.to(self.device)
            c = c.to(self.device)
            #print('----------------------', image.shape)
            pre = model.predict(image, c)
            predict.extend(pre.cpu().tolist())
            groundtruth.extend(label.cpu().tolist())
            name_list.extend(list(names))
            
        return e.view(name_list, predict, groundtruth, print_probability)
      
    def train(self):
        args = self.args
        train_loader, val_loader = instantiate_from_config(args.data) #dataset(args, 'train')
        optim = Optimizer()
        self.optimizer = optim.get_optimizer(args.optim, self.model.parameters())

        # criterion = F.mse_loss
        if args.model.ema:
            self.ema_helper = EMA(mu=args.model.ema_rate)
            self.ema_helper.register(self.model)
        else:
            self.ema_helper = None
        # load
        start_epoch = 1
        if args.load:
            states = torch.load(args.load, map_location=self.device)
            self.model.load_state_dict(states[0])
            states[1]["param_groups"][0]["eps"] = args.optim.eps
            self.optimizer.load_state_dict(states[1])
            start_epoch = states[2]+1
            if args.model.ema:
                self.ema_helper.load_state_dict(states[3])
        
        # train
        self.logger.info('training...')
        for epoch in range(start_epoch, args.training.epochs+1):
            self.model.train()
            total_loss = 0

            for i, batch in enumerate(train_loader):
                names, x_pCR0, y_pCR0, x_pCR1, y_pCR1, c1, c2 = batch
                x_pCR0 = x_pCR0.to(self.device)
                y_pCR0 = y_pCR0.to(self.device)
                x_pCR1 = x_pCR1.to(self.device)
                y_pCR1 = y_pCR1.to(self.device)
                c1 = c1.to(self.device)
                c2 = c2.to(self.device)

                
                if args.optim.lr_schedule:
                    optim.adjust_learning_rate(self.optimizer, i / len(train_loader) + epoch, args)
                # loss = self.model(x_pCR0, y_pCR0, y_pCR1)
                # loss = self.model(x_pCR0, y_pCR0, x_pCR1, y_pCR1)

                loss = self.model(x_pCR0, y_pCR0, x_pCR1, y_pCR1, c1, c2, 1)
                total_loss += loss.item()/3
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() # 清空过往梯度
                if args.model.ema: # TODO:
                    self.ema_helper.update(self.model)

                loss = self.model(x_pCR0, y_pCR0, x_pCR1, y_pCR1, c1, c2, 2)
                total_loss += loss.item()/3
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() # 清空过往梯度
                if args.model.ema: # TODO:
                    self.ema_helper.update(self.model)
                
                loss = self.model(x_pCR0, y_pCR0, x_pCR1, y_pCR1, c1, c2, 3)
                total_loss += loss.item()/3
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad() # 清空过往梯度
                if args.model.ema: # TODO:
                    self.ema_helper.update(self.model)

                # if ((i+1)%args.training.accumulate_grad_batches)==0:
                #     self.optimizer.step()
                #     self.optimizer.zero_grad() # 清空过往梯度
                #     if args.model.ema: # TODO:
                #         self.ema_helper.update(self.model)

            self.epoch = epoch
            ave_loss = total_loss/len(train_loader)
            self.logger.info(f'epoch {epoch}, ave_loss={ave_loss}')
            self.epoch_process(epoch, val_loader)
        
        self.train_end(epoch, val_loader)
        

    def epoch_process(self, epoch, loader):
        if (epoch >= self.args.training.sample_checkpoint and epoch % self.args.training.sample_rate == 0) or (epoch in self.args.training.sample_some): 
            
            states = [
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                    epoch
                ]
            if self.args.model.ema:
                states.append(self.ema_helper.state_dict())
            torch.save(states, os.path.join(self.weight_path, f"epoch{epoch}_ckpt.pth"))
        return


    def train_end(self, epoch, loader):
        # self.logger.info(f'training over, best_acc {self.best_acc} in epoch {self.best_epoch}')

        # save last
        states = [
                self.model.state_dict(),
                self.optimizer.state_dict(),
                epoch
            ]
        if self.args.model.ema:
            states.append(self.ema_helper.state_dict())
        torch.save(states, os.path.join(self.weight_path, f"epoch{epoch}_ckpt.pth"))
        
        # if epoch < self.args.training.sample_checkpoint or epoch % self.args.training.sample_rate != 0: self.evaluate_test(loader)
        return

    def train2(self):
        args = self.args
        self.best_acc = 0.0
        self.best_epoch = 0
        # data
        train_loader, val_loader = instantiate_from_config(args.data)

        # optim
        optim = Optimizer()
        self.optimizer = optim.get_optimizer(args.optim, self.model.parameters())

        # criterion = F.mse_loss
        if args.model.ema:
            self.ema_helper = EMA(mu=args.model.ema_rate)
            self.ema_helper.register(self.model)
        else:
            self.ema_helper = None
        # load
        start_epoch = 1
        if args.load:
            states = torch.load(args.load, map_location=self.device)
            self.model.load_state_dict(states[0], strict=False)
            states[1]["param_groups"][0]["eps"] = args.optim.eps

        # train
        self.logger.info('training...')
        for epoch in range(start_epoch, args.training.epochs+1):
            self.model.train()
            total_loss = 0

            for i, batch in enumerate(train_loader):
                if isinstance(batch, dict):
                    names, image, label = batch['name'], batch['t1'][tio.DATA], batch['label']
                    image = image.permute(0,1,4,2,3)
                else:
                    # names, image, label = batch
                    names, image, label, c = batch
                image = image.to(self.device)
                label = label.to(self.device)
                c = c.to(self.device)

                
                if args.optim.lr_schedule:
                    optim.adjust_learning_rate(self.optimizer, i / len(train_loader) + epoch, args)
                # loss = self.model(image, label)
                loss = self.model(image, c, label)
                total_loss += loss.item()

                loss.backward()
                if ((i+1)%args.training.accumulate_grad_batches)==0:
                    self.optimizer.step()
                    self.optimizer.zero_grad() # 清空过往梯度
                    if args.model.ema: # TODO:
                        self.ema_helper.update(self.model)

            self.epoch = epoch
            ave_loss = total_loss/len(train_loader)
            self.logger.info(f'epoch {epoch}, ave_loss={ave_loss}')
            self.epoch_process2(epoch, val_loader)
        
        self.train_end2(epoch, val_loader)

    def epoch_process2(self, epoch, loader):
        if epoch > n
            acc = self.evaluate(loader)
            if acc > self.best_acc:
                self.logger.info(f'save best: {acc}')
                self.best_acc = acc
                self.best_epoch = epoch
                states = [
                        self.model.state_dict(),
                        self.optimizer.state_dict(),
                        epoch
                    ]
                if self.args.model.ema:
                    states.append(self.ema_helper.state_dict())
                torch.save(states, os.path.join(self.weight_path, "best_ckpt.pth"))
        return


    def train_end2(self, epoch, loader):
        self.logger.info(f'training over, best_acc {self.best_acc} in epoch {self.best_epoch}')

        # save last
        if self.args.model.save_last:
            states = [
                    self.model.state_dict(),
                    self.optimizer.state_dict(),
                    epoch
                ]
            if self.args.model.ema:
                states.append(self.ema_helper.state_dict())
            torch.save(states, os.path.join(self.weight_path, f"epoch{epoch}s2_ckpt.pth"))
        
        if epoch < self.args.training.sample_checkpoint or epoch % self.args.training.sample_rate != 0: self.evaluate_test(loader)
        return


    def load(self): # load model, only for test
        states = torch.load(self.args.load, map_location=self.device)
        self.model.load_state_dict(states[0])
        self.logger.info(f'---in epoch {states[2]}---', '%(message)s')

    def test(self, load=False): 
        if not load: self.load()
        self.args.data.params.tag='test'
        test_loader = instantiate_from_config(self.args.data)
        self.evaluate_save(test_loader, True)
        self.logger.info('-'*10, '%(message)s')

    @torch.no_grad()
    def evaluate_save(self, loader, print_probability=False): # 这个会把预测结果（概率）保存，供以后画图用
        model = self.model
        model.eval()
        predict = []
        groundtruth = []
        name_list = []
        e = Evaluation(self.logger, metrics=self.args.evaluation.metrics)
        save = f'predict/{self.args.name}'
        os.makedirs(save, exist_ok=True)
        for batch in loader:
            if isinstance(batch, dict):
                names, image, label = batch['name'], batch['t1'][tio.DATA], batch['label']
                image = image.permute(0,1,4,2,3)
            else:
                # names, image, label = batch
                names, image, label, c = batch
                image = image.to(self.device)
                c = c.to(self.device)
            #print('----------------------', image.shape)
            pre = model.predict(image, c)
            predict.extend(pre.cpu().tolist())
            groundtruth.extend(label.cpu().tolist())
            name_list.extend(list(names))
        
        np.save(f"{save}/pre.npy", np.array(predict))
        np.save(f"{save}/gt.npy", np.array(groundtruth))
        np.save(f"{save}/name.npy", np.array(name_list))
        return e.view(name_list, predict, groundtruth, print_probability)
