import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from hiera_diffusion_policy.so3diffusion.util import *
from hiera_diffusion_policy.so3diffusion.denoising_diffusion_pytorch \
    import (extract,
            exists,
            default,
            noise_like,
            cosine_beta_schedule,
            )
from tqdm import tqdm
from hiera_diffusion_policy.so3diffusion.distributions import IsotropicGaussianSO3, IGSO3xR3


def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()


class ObjCache(object):
    def __init__(self, cls, device=torch.device('cpu')):
        self.cls = cls
        self.device = device
        self.objdict = dict()

    def __call__(self, *args):
        try:
            obj = self.objdict[args]
        except KeyError:
            obj = self.cls(*args, device=self.device)
            self.objdict[args] = obj
        return obj


# tweaked lucidrains diffusion implementation to support non 2D data.
class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            # denoise_fn,
            *,
            image_size,
            channels=3,
            timesteps=1000,
            loss_type='l2',
            betas=None
            ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        # self.denoise_fn = denoise_fn

        if exists(betas):
            betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        else:
            betas = cosine_beta_schedule(timesteps)

        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.denoise_fn(x, t))

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))
        return img

    @torch.no_grad()
    def sample(self, batch_size=16):
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size))

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc='interpolation sample time step', total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        x_recon = self.denoise_fn(x_noisy, t)

        if self.loss_type == 'l1':
            loss = (noise - x_recon).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss

    def forward(self, x, *args, **kwargs):
        b = x.shape[0]
        device = x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)


#* =======================================================================================

class SO3Diffusion(GaussianDiffusion):
    def __init__(self, timesteps=100, loss_type='skewvec', betas=None):
        super().__init__(image_size=None, timesteps=timesteps, loss_type=loss_type, betas=betas)
        self.register_buffer("identity", torch.eye(3))

    def q_mean_variance(self, x_start, t):
        #! so3_lerp 可能要换成 so3_scale，看到的时候注意
        mean = so3_lerp(self.identity, x_start, extract(self.sqrt_alphas_cumprod, t, x_start.shape))
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        """
        根据预测的v计算x_0, x_0 = a_1 * a_2^{-1}
        args:
            - x: (B, 3, 3) 含噪声的旋转矩阵
            - t: (1,) timestep
            - noise: (B, 3) v
        """
        x_t_term = so3_scale(x_t, extract(self.sqrt_recip_alphas_cumprod, t, t.shape))  # a_1
        noise_vec = noise * extract(self.sqrt_recipm1_alphas_cumprod, t, t.shape)[..., None]
        noise_term = torch.matrix_exp(vec2skew(noise_vec))  # a_2
        # Translation = subtraction,
        # Rotation = multiply by inverse op (matrices, so transpose)
        return x_t_term @ noise_term.transpose(-1, -2)

    def q_posterior(self, x_start, x_t, t):
        """
        根据x_0和x_t，计算x_{t-1}
        args:
            - x_start: (B, 3, 3) x_0
            - x_t: (B, 3, 3) x_t
        """
        c_1 = so3_scale(x_start, extract(self.posterior_mean_coef1, t, t.shape))    # x_0缩放  (B, 3, 3)
        c_2 = so3_scale(x_t, extract(self.posterior_mean_coef2, t, t.shape))    # x_t缩放   (B, 3, 3)
        posterior_mean = c_1 @ c_2  # x_{t-1} 均值 (B, 3, 3)

        posterior_variance = extract(self.posterior_variance, t, t.shape)   # x_{t-1} 方差 (1,)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, t.shape)   # log(x_{t-1} 方差) (1,)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        """
        x: (B, 3, 3) 含噪声的旋转矩阵
        t: (1,) timestep
        """
        predict = self.denoise_fn(x, t) # (B,3) 预测v
        x_recon = self.predict_start_from_noise(x, t=t, noise=predict)  # x_0, (B, 3, 3)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)

        return model_mean, posterior_variance, posterior_log_variance
    

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=False, repeat_noise=False):
        """逆扩散，计算x_{t-1}
        x: (B, 3, 3) 含噪声的旋转矩阵
        t: (B,) timestep
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised) # 噪声均值和log(方差)

        if (t == 0.0).all():
            # no noise when t == 0
            return model_mean
        else:
            model_stdev = (0.5 * model_log_variance).exp()  # 噪声 标准差 (B,)
            # sample = IsotropicGaussianSO3(model_stdev[0]).sample([b])   # 采样噪声旋转矩阵 (B, 3, 3)
            sample = IsotropicGaussianSO3(model_stdev).sample()   # 采样噪声旋转矩阵 (B, 3, 3)
            return model_mean @ sample  # 均值x噪声
        
    @torch.no_grad()
    def step(self, noise, t, rotation, re_0=False):
        """逆扩散，计算x_{t-1}

        args:
            - noise: 模型预测的噪声旋转的v, (B, 3)
            - t: timestep (B,)
            - rotation: 含噪声的旋转矩阵 (B, 3, 3)
        
        return:
            - x_{t-1}: 去除噪声的rotation, (B, 3, 3)
            - x_start: 由当前步预测的噪声计算的x_0
        """
        x_start = self.predict_start_from_noise(rotation, t=t, noise=noise)  # x_0, (B, 3, 3)
        model_mean, _, model_log_variance = self.q_posterior(x_start=x_start, x_t=rotation, t=t)   # x_{t-1}均值 和 log(方差)

        if (t == 0.0).all():
            # no noise when t == 0
            if re_0:
                return model_mean, x_start
            else:
                return model_mean
        else:
            model_stdev = (0.5 * model_log_variance).exp()  # 标准差 (B,)
            sample = IsotropicGaussianSO3(model_stdev).sample()   # 采样噪声旋转矩阵 (B, 3, 3)
            if re_0:
                return model_mean @ sample, x_start
            else:
                return model_mean @ sample

    @torch.no_grad()
    def p_sample_loop(self, shape):
        device = self.betas.device
        b = shape[0]
        # Initial Haar-Uniform random rotations from QR decomp of normal IID matrix
        x = IsotropicGaussianSO3(eps=torch.ones([], device=device)).sample(shape)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long))
        return x

    def q_sample(self, x_start, t, noise=None):
        """对旋转矩阵加噪，缩放旋转矩阵，再乘噪声旋转矩阵
        x_start: (B, 3, 3)
        """
        # if noise is None:
        #     eps = extract(self.sqrt_one_minus_alphas_cumprod, t, t.shape)
        #     noise = IsotropicGaussianSO3(eps).sample()
        scale = extract(self.sqrt_alphas_cumprod, t, t.shape)   # shape=(B,) 与timestep对应的 sqrt_alphas_cumprod
        x_blend = so3_scale(x_start, scale) # (B, 3, 3) # 缩放后的旋转矩阵
        return x_blend @ noise

    def p_losses(self, x_start, t, noise=None):
        """
        x_start: (B, 3, 3) Batch个沿z轴旋转90或-90度的旋转矩阵
        t: (B,) diffusion timestep
        """
        eps = extract(self.sqrt_one_minus_alphas_cumprod, t, t.shape)   # shape=(B,) 与timestep对应的 sqrt_one_minus_alphas_cumprod
        noisedist = IsotropicGaussianSO3(eps)
        noise = noisedist.sample()  # (B, 3, 3) 采样噪声旋转矩阵IG(I, sqrt_one_minus_alphas_cumprod)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)  # 对x_start加噪后的旋转矩阵，包括缩放和乘噪声
        x_recon = self.denoise_fn(x_noisy, t)   # 预测v

        descaled_noise = skew2vec(log_rmat(noise)) * (1 / eps)[..., None]   # real v, (B, 3) * (B, 1) = (B, 3)
        
        if self.loss_type == "skewvec": # <--
            loss = F.mse_loss(x_recon, descaled_noise)
        elif self.loss_type == "prevstep":
            # Calculate mean of previous step's distribution
            posterior_mean, _, _ = self.q_posterior(x_start, x_noisy, t)
            # Calculate rotation from current rotation (x_noisy) to previous step (step)
            # Treat p_m = x_smooth @ step
            # x_smooth^-1 @ p_m = I @ step
            step = x_noisy.transpose(-1, -2) @ posterior_mean
            loss = rmat_dist(x_recon, step).pow(2.0).mean()
        else:
            RuntimeError(f"Unexpected loss_type: {self.loss_type}")

        return loss

    def add_noise(self, rmat_start: torch.Tensor, timesteps: torch.Tensor):
        """
        对输入的旋转矩阵加噪

        ### args:
            - rmat_start: 旋转矩阵, (B, 3, 3)
            - timesteps: (B,)

        ### return:
            - descaled_noise: v矩阵 (B, 3)
            - noisy_rotation: 加噪后的旋转矩阵 (B, 3, 3)
        """
        eps = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, timesteps.shape)   # shape=(B,) 与timestep对应的 sqrt_one_minus_alphas_cumprod
        noise = IsotropicGaussianSO3(eps).sample()  # (B, 3, 3) 采样噪声旋转矩阵IG(I, sqrt_one_minus_alphas_cumprod)
        rmat_noisy = self.q_sample(x_start=rmat_start, t=timesteps, noise=noise)  # 对rmat_start加噪后的旋转矩阵，包括缩放和乘噪声
        descaled_noise = skew2vec(log_rmat(noise)) * (1 / eps)[..., None]   # real v, (B, 3) * (B, 1) = (B, 3)
        return descaled_noise, rmat_noisy


    def forward(self, x, *args, **kwargs):
        """
        x: (B, 3, 3) Batch个沿z轴旋转90或-90度的旋转矩阵
        """
        b, *_, device = *x.shape, x.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x, t, *args, **kwargs)
