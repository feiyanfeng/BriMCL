import os
from tools.utils import instantiate_from_config
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from .BBM_utils import extract, default
from .model import *

import cv2

import matplotlib.pyplot as plt


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        # 0: 同类拉近； 1：异类推远

    def forward(self, output1, output2, label):
        # 计算两个特征之间的欧氏距离
        euclidean_distance = torch.nn.functional.pairwise_distance(output1, output2)
        print(f"Current feature distance mean: {euclidean_distance.mean().item()}, max: {euclidean_distance.max().item()}, label mean: {label.mean().item()}")
        # print('d:', euclidean_distance)
        
        # 计算对比损失
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      label * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
        return loss_contrastive
    

class BrownianBridgeModel(nn.Module):
    def __init__(self, config, view_only=False):
        super().__init__()
        model_config, diff_config = config.model, config.diffusion
        self.model_config = model_config # TODO: not used
        # model hyperparameters
        # model_params = model_config.BB.params
        self.num_timesteps = diff_config.timesteps
        self.mt_type = diff_config.beta_schedule
        self.max_var = diff_config.max_var if diff_config.__contains__("max_var") else 1
        self.var_scale = diff_config.var_scale
        self.eta = diff_config.eta if diff_config.__contains__("eta") else 1
        self.skip_sample = diff_config.skip_sample
        self.sample_type = diff_config.beta_schedule
        self.sample_step = diff_config.sample_step
        self.steps = None
        self.register_schedule()

        # loss and objective
        # self.loss_type = diff_config.loss_type
        self.objective = diff_config.objective
        # self.BCE = torch.nn.BCELoss()
        self.contrastive_loss1 = ContrastiveLoss(margin=config.contrastive_margin)
        self.contrastive_loss2 = ContrastiveLoss(margin=config.contrastive_margin)
        self.contrastive_loss3 = ContrastiveLoss(margin=config.contrastive_margin)
        self.t_in_model = diff_config.t_in_model

        if not view_only:
            self.device = diff_config.device 
            self.denoise_fn = instantiate_from_config(self.model_config).to(self.device)


    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * self.var_scale * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)



    def q_sample(self, x0, y, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x0))
        m_t = extract(self.m_t, t, x0.shape)
        var_t = extract(self.variance_t, t, x0.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (y - x0) + sigma_t * noise  # y - x_t
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = y - x0
        elif self.objective == 'x0':
            objective = x0
        elif self.objective == 'xt-1':
            m_t_1 = extract(self.m_t, t-1, x0.shape)
            var_t_1 = extract(self.variance_t, t-1, x0.shape)
            sigma_t_1 = torch.sqrt(var_t_1)
            objective = (1. - m_t_1) * x0 + m_t_1 * y + sigma_t_1 * noise
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x0 + m_t * y + sigma_t * noise,
            objective
        )

    def view_q_sample(self, x0, y, path, name):
        noise = torch.randn_like(x0)
        for t in self.steps:
            t_batch = torch.tensor([t])
            # print(x0.shape, t.shape, y.shape)
            # exit()
            x_t, _ = self.q_sample(x0, y, t_batch, noise)
            # print(torch.max(x_t), torch.min(x_t)) # tensor(1.0358) tensor(-1.1701)
            # exit()
            x_t = (x_t.numpy().clip(-1, 1)+1)/2
            x_t = np.uint8(255 * x_t[0][0])
            # perturbed_x = np.broadcast_to(perturbed_x[:,:, np.newaxis],(128,128,3))
            cv2.imwrite(f"{path}/{name}_{t}.png", x_t)
            # exit()

    def forward(self, x_pCR0, y_pCR0, x_pCR1, y_pCR1, c1, c2, s):
        pass
    


    @torch.no_grad()
    def get_features(self, x, y, c):
        b = x.shape[0]
        t = torch.full([b], self.num_timesteps-1, device=self.device)
        noise = torch.randn_like(x)
        x_t, _ = self.q_sample(x, y, t, noise)
        features, _ = self.denoise_fn(x_t, c, t=t if self.t_in_model else None)
        return features



    def predict_x0_from_objective(self, x_t, y, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)  # TODO: variance_t
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * y - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = y - objective_recon
        elif self.objective == 'x0':
            x0_recon = objective_recon
        else: # xt-1 不需要到这
            raise NotImplementedError
        return x0_recon



    @torch.no_grad()
    def p_sample(self, x_t, y, context, i, clip_denoised=False):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, t=t, y=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            objective_recon = self.denoise_fn(x_t, t=t, y=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * y + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * y)

            return x_tminus_mean + sigma_t * noise, x0_recon

 
    @torch.no_grad()
    def p_sample_loop(self, y, context=None, clip_denoised=True, sample_mid_step=False):
        # if self.condition_key == "nocond":
        #     context = None
        # else:
        context = y if context is None else context

        if sample_mid_step:
            imgs, one_step_imgs = [y], []
            # for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
            for i in range(len(self.steps)):
                img, x0_recon = self.p_sample(x_t=imgs[-1], y=y, context=context, i=i, clip_denoised=clip_denoised)
                imgs.append(img)
                one_step_imgs.append(x0_recon)
            return imgs, one_step_imgs
        else:
            img = y
            # for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):  # 这里的step是测试的step
            for i in range(len(self.steps)):  # 这里的step是测试的step
                img, _ = self.p_sample(x_t=img, y=y, context=context, i=i, clip_denoised=clip_denoised)
            return img


    @torch.no_grad()
    def sample(self, y, context, clip_denoised=True, sample_mid_step=False, type='default'):
        if type=='default':
            return self.p_sample_loop(y, context, clip_denoised, sample_mid_step).cpu().detach()
        elif type=='x0': # 测试的x0和训练的x0不是一个概念。训练的x0应该属于default
            x_t = y
            t = torch.full((x_t.shape[0],), self.steps[0], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, t=t, y=context)
            x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon.cpu().detach()
        elif type=='x0_to_xt-1':
            x_t=y
            for i in range(len(self.steps)-1):  # 这里的step是测试的step
                t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
                objective_recon = self.denoise_fn(x_t, t=t, y=context)
                x0_recon = self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon)
                if clip_denoised:
                    x0_recon.clamp_(-1., 1.)
                t_1 = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)
                x_t, _ = self.q_sample(x0_recon, y, t_1, noise=None) # xt-1
            t = torch.full((x_t.shape[0],), self.steps[-1], device=x_t.device, dtype=torch.long)
            objective_recon = self.denoise_fn(x_t, t=t, y=context)
            return self.predict_x0_from_objective(x_t, y, t, objective_recon=objective_recon).cpu().detach()
        elif type == 'xt-1':
            x_t=y
            for i in range(len(self.steps)-1):
                t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
                x_t = self.denoise_fn(x_t, t=t, y=context)
            return x_t.cpu().detach()
        else:
            raise ValueError(type)
