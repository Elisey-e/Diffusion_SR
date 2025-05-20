"""
Полный пайплайн для тренировки диффузионной модели super-resolution на COCO датасете.
Оптимизирован для работы с ограниченной VRAM.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as TF
from torch.cuda.amp import autocast, GradScaler
from PIL import Image
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import logging
import argparse
from pathlib import Path

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# Определение аргументов командной строки
parser = argparse.ArgumentParser(description='Обучение диффузионной модели для Super-Resolution')
parser.add_argument('--data_dir', type=str, default='./coco', help='Директория с COCO датасетом')
parser.add_argument('--output_dir', type=str, default='./output', help='Директория для сохранения результатов')
parser.add_argument('--batch_size', type=int, default=8, help='Размер батча')
parser.add_argument('--epochs', type=int, default=50, help='Количество эпох')
parser.add_argument('--lr', type=float, default=2e-4, help='Скорость обучения')
parser.add_argument('--num_workers', type=int, default=4, help='Количество workers для загрузки данных')
parser.add_argument('--scale_factor', type=int, default=4, help='Коэффициент масштабирования для SR')
parser.add_argument('--image_size', type=int, default=128, help='Размер HR изображения')
parser.add_argument('--save_interval', type=int, default=5, help='Интервал сохранения модели (эпохи)')
parser.add_argument('--grad_accum_steps', type=int, default=1, help='Шаги накопления градиента')
parser.add_argument('--mixed_precision', action='store_true', help='Использовать смешанную точность')
parser.add_argument('--checkpoint', type=str, default=None, help='Путь к checkpoint для продолжения обучения')



# ================================
# Класс датасета для Super-Resolution
# ================================

class SRDataset(Dataset):
    def __init__(self, coco_dataset, hr_size=128, scale_factor=4, transform=None):
        """
        Датасет для задачи Super-Resolution
        
        Args:
            coco_dataset: COCO датасет
            hr_size: размер изображения высокого разрешения
            scale_factor: коэффициент масштабирования
            transform: дополнительные преобразования
        """
        self.coco_dataset = coco_dataset
        self.hr_size = hr_size
        self.lr_size = hr_size // scale_factor
        self.scale_factor = scale_factor
        self.transform = transform
        
    def __len__(self):
        return len(self.coco_dataset)
    
    def __getitem__(self, idx):
        image, _ = self.coco_dataset[idx]
        
        # Проверка и конвертация изображения в RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Если размер изображения меньше требуемого, масштабируем его так,
        # чтобы обе стороны были ≥ hr_size, сохраняя соотношение сторон
        w, h = image.size
        if w < self.hr_size or h < self.hr_size:
            scale = self.hr_size / min(w, h)
            new_w, new_h = int(w * scale + 0.5), int(h * scale + 0.5)
            image = TF.resize(image, (new_h, new_w), interpolation=Image.BICUBIC)
        
        # Случайная обрезка patch'а exact hr_size × hr_size
        i, j, h_crop, w_crop = transforms.RandomCrop.get_params(
            image, output_size=(self.hr_size, self.hr_size)
        )
        hr_image = TF.crop(image, i, j, h_crop, w_crop)
        
        # Применяем дополнительные преобразования или конвертируем в тензор
        if self.transform:
            hr_image = self.transform(hr_image)
        else:
            hr_image = transforms.ToTensor()(hr_image)
        
        # Создаем LR версию изображения
        lr_image = F.interpolate(
            hr_image.unsqueeze(0),
            size=(self.lr_size, self.lr_size),
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        
        # Создаем upscaled версию LR изображения (для условного входа в сеть)
        lr_upscaled = F.interpolate(
            lr_image.unsqueeze(0),
            scale_factor=self.scale_factor,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        
        return {
            'lr': lr_image,
            'hr': hr_image,
            'lr_upscaled': lr_upscaled
        }


# ================================
# Модель U-Net для диффузии
# ================================

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # Убедимся, что количество групп для GroupNorm не превышает количество каналов
        groups = min(32, out_channels)
        
        self.block = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
        )
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        return self.block(x) + self.shortcut(x)




class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        b, c, h, w = x.shape
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x)
        q, k, v = qkv.chunk(3, dim=1)
        
        q = q.reshape(b, c, h * w).transpose(1, 2)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).transpose(1, 2)
        
        scale = 1.0 / math.sqrt(c)
        attn = torch.bmm(q, k) * scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.bmm(attn, v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


# ---------------------------
# 1. Энкодерный Down-блок
# ---------------------------
class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_channels, has_attn=False):
        super().__init__()
        self.res1 = ResidualBlock(in_channels,  out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels)
        )

        self.has_attn = has_attn
        if has_attn:
            self.attn = AttentionBlock(out_channels)

        # ↓2: AvgPool → Conv
        self.downsample = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, time_emb):
        x = self.res1(x)
        x = self.res2(x) + self.time_emb(time_emb).unsqueeze(-1).unsqueeze(-1)
        if self.has_attn:
            x = self.attn(x)

        skip = x              # (B, C, H, W) — ДО downsample
        x    = self.downsample(x)
        return x, skip        # x ↓2,  skip — оригинал




class UpBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        skip_channels: int,
        out_channels: int,
        time_channels: int,
        has_attn: bool = False,
    ):
        super().__init__()

        self.skip_proj = (
            nn.Identity()
            if skip_channels == out_channels
            else nn.Conv2d(skip_channels, out_channels, kernel_size=1)
        )

        self.res1 = ResidualBlock(in_channels + out_channels, out_channels)
        self.res2 = ResidualBlock(out_channels, out_channels)

        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_channels, out_channels),
        )

        self.has_attn = has_attn
        if has_attn:
            self.attn = AttentionBlock(out_channels)

        self.upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )

    def forward(self, x, skip, time_emb):
        # 0. если spatial-размеры не совпадают → подгоняем skip
        if skip.shape[2:] != x.shape[2:]:
            skip = F.interpolate(skip, size=x.shape[2:], mode="nearest")

        # 1. выравниваем каналы
        skip = self.skip_proj(skip)

        # 2. объединяем + два Res-блока
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x)
        x = self.res2(x) + self.time_emb(time_emb).unsqueeze(-1).unsqueeze(-1)

        # 3. (опц.) attention
        if self.has_attn:
            x = self.attn(x)

        # 4. апсемпл
        return self.upsample(x)




class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class UNetSR(nn.Module):
    """
    U-Net для условной (LR→HR) диффузии Super-Resolution, построенный
    на исправленных Down/Up-блоках.
    """
    def __init__(
        self,
        in_channels: int = 6,            # 3 канала LR + 3 канала cond
        cond_channels: int = 3,
        base_channels: int = 64,
        channel_mults: tuple = (1, 2, 4),
        time_emb_dim: int = 256,
    ):
        super().__init__()

        # ---------- time-embedding ----------
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # ---------- входной блок ----------
        self.input_conv = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.cond_proj  = nn.Conv2d(cond_channels, base_channels, kernel_size=3, padding=1)

        # ---------- энкодер ----------
        self.down_blocks  = nn.ModuleList()
        self.skip_channels = []                      # будем запоминать C каждого skip
        channels = base_channels

        for mult in channel_mults:                   # (1, 2, 4) …
            out_channels = base_channels * mult
            self.down_blocks.append(
                DownBlock(
                    in_channels  = channels,
                    out_channels = out_channels,
                    time_channels= time_emb_dim,
                    has_attn     = mult >= 2,        # attention на средних/глубоких уровнях
                )
            )
            self.skip_channels.append(out_channels)  # пригодится декодеру
            channels = out_channels                  # для следующего уровня

        # ---------- bottleneck ----------
        self.middle_block = nn.Sequential(
            ResidualBlock(channels, channels),
            AttentionBlock(channels),
            ResidualBlock(channels, channels),
        )

        # ---------- декодер ----------
        self.up_blocks = nn.ModuleList()
        for mult in reversed(channel_mults):         # (4, 2, 1)
            skip_ch   = self.skip_channels.pop()     # соответствующий энкодерный skip
            out_ch    = skip_ch                      # хотим выйти на те же каналы
            self.up_blocks.append(
                UpBlock(
                    in_channels   = channels,        # из предыдущего уровня
                    skip_channels = skip_ch,
                    out_channels  = out_ch,
                    time_channels = time_emb_dim,
                    has_attn      = mult >= 2,
                )
            )
            channels = out_ch                        # для следующего up-уровня

        # ---------- выход ----------
        self.output_conv = nn.Sequential(
            nn.GroupNorm(min(32, channels), channels),
            nn.SiLU(),
            nn.Conv2d(channels, 3, kernel_size=3, padding=1),
        )

    # -------- forward ---------
    def forward(self, x_lr, time, cond):
        """
        x_lr  : (B, 3,  H,  W)  — зашумл. HR-тензор (в процессе диффузии)
        cond  : (B, 3,  H,  W)  — bicubic-upsampled LR (условие)
        time  : (B,)            — шаг t
        """
        # time-embedding
        t_emb = self.time_embedding(time)

        # объединяем вход с условием
        x = torch.cat([x_lr, cond], dim=1)           # 6 каналов
        x = self.input_conv(x) + self.cond_proj(cond)

        # -------- энкодер --------
        skips = []
        for down in self.down_blocks:
            x, skip = down(x, t_emb)
            skips.append(skip)

        # -------- bottleneck --------
        x = self.middle_block(x)

        # -------- декодер --------
        for up in self.up_blocks:
            skip = skips.pop()                       # соответствующий skip-коннект
            x = up(x, skip, t_emb)

        # -------- выход --------
        return self.output_conv(x)





# ================================
# Шумовой планировщик (Scheduler)
# ================================

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Косинусное расписание шума, как в улучшенном DDPM
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class NoiseScheduler:
    def __init__(self, num_timesteps=1000, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device
        
        # Определяем бета-расписание
        self.betas = cosine_beta_schedule(num_timesteps)
        
        # Определяем альфа и накопленную альфа
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Константы для диффузионного процесса
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        
        # Константы для предсказания шума
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        # Перемещаем все тензоры на нужное устройство
        self._move_to_device()
    
    def _move_to_device(self):
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
        self.log_one_minus_alphas_cumprod = self.log_one_minus_alphas_cumprod.to(self.device)
        self.sqrt_recip_alphas_cumprod = self.sqrt_recip_alphas_cumprod.to(self.device)
        self.sqrt_recipm1_alphas_cumprod = self.sqrt_recipm1_alphas_cumprod.to(self.device)
        self.posterior_variance = self.posterior_variance.to(self.device)
        self.posterior_log_variance_clipped = self.posterior_log_variance_clipped.to(self.device)
        self.posterior_mean_coef1 = self.posterior_mean_coef1.to(self.device)
        self.posterior_mean_coef2 = self.posterior_mean_coef2.to(self.device)
    
    def q_sample(self, x_start, t, noise=None):
        """
        Добавляет шум к x_start согласно расписанию шума
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """
        Предсказывает исходное изображение x_0 из зашумленного изображения x_t и предсказанного шума
        """
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def q_posterior(self, x_start, x_t, t):
        """
        Вычисляет параметры апостериорного распределения q(x_{t-1} | x_t, x_0)
        """
        posterior_mean_coef1_t = self.posterior_mean_coef1[t].reshape(-1, 1, 1, 1)
        posterior_mean_coef2_t = self.posterior_mean_coef2[t].reshape(-1, 1, 1, 1)
        
        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t
        posterior_variance_t = self.posterior_variance[t].reshape(-1, 1, 1, 1)
        posterior_log_variance_t = self.posterior_log_variance_clipped[t].reshape(-1, 1, 1, 1)
        
        return posterior_mean, posterior_variance_t, posterior_log_variance_t
    
    def p_mean_variance(self, model, x_t, t, cond, clip_denoised=True):
        """
        Вычисляет параметры распределения p(x_{t-1} | x_t)
        """
        # Предсказание шума с помощью модели
        pred_noise = model(x_t, t, cond)
        
        # Предсказание x_0
        x_recon = self.predict_start_from_noise(x_t, t, pred_noise)
        
        if clip_denoised:
            x_recon = torch.clamp(x_recon, -1., 1.)
        
        # Вычисление параметров для p(x_{t-1} | x_t)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_recon, x_t, t)
        
        return model_mean, posterior_variance, posterior_log_variance, x_recon
    
    def p_sample(self, model, x_t, t, cond, clip_denoised=True):
        """
        Один шаг сэмплирования из p(x_{t-1} | x_t)
        """
        model_mean, _, posterior_log_variance, x_recon = self.p_mean_variance(
            model, x_t, t, cond, clip_denoised=clip_denoised
        )
        
        noise = torch.randn_like(x_t)
        # Без шума на последнем шаге
        nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1)
        
        return model_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise, x_recon
    
    def p_sample_loop(self, model, shape, cond, device, return_all_timesteps=False):
        """
        Генерирует изображение с помощью последовательного сэмплирования
        """
        batch_size = shape[0]
        img = torch.randn(shape, device=device)
        imgs = [img]
        
        # Итерирование от T до 0
        for i in tqdm(reversed(range(0, self.num_timesteps)), total=self.num_timesteps, desc="Sampling"):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img, _ = self.p_sample(model, img, t, cond)
            imgs.append(img)
            
        return imgs if return_all_timesteps else imgs[-1]
    
    def p_losses(self, denoise_model, x_start, t, cond, noise=None):
        """
        Вычисляет функцию потерь для обучения модели
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Получаем зашумленное изображение на шаге t
        x_noisy = self.q_sample(x_start, t, noise)
        
        # Предсказание шума моделью
        predicted_noise = denoise_model(x_noisy, t, cond)
        
        # Считаем MSE между настоящим и предсказанным шумом
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss


# ================================
# Функции для обучения и оценки модели
# ================================

def load_coco_dataset(data_dir, transforms=None):
    """Загружает COCO датасет и готовит его к использованию"""
    coco_train = CocoDetection(
        root=os.path.join(data_dir, 'train2017'),
        annFile=os.path.join(data_dir, 'annotations/instances_train2017.json'),
        transform=transforms
    )
    
    coco_val = CocoDetection(
        root=os.path.join(data_dir, 'val2017'),
        annFile=os.path.join(data_dir, 'annotations/instances_val2017.json'),
        transform=transforms
    )
    
    return coco_train, coco_val


def prepare_dataloaders(args):
    """Подготавливает датасеты и DataLoader'ы для обучения"""
    # Загружаем COCO датасет
    coco_train, coco_val = load_coco_dataset(args.data_dir)
    
    # Создаем датасеты для Super-Resolution
    train_dataset = SRDataset(coco_train, hr_size=args.image_size, scale_factor=args.scale_factor)
    val_dataset = SRDataset(coco_val, hr_size=args.image_size, scale_factor=args.scale_factor)
    
    # Создаем DataLoader'ы
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader


def setup_training(args):
    """Настраивает модель, оптимизатор и планировщик шума"""
    # Определяем устройство для обучения
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Создаем модель
    model = UNetSR(in_channels=6, base_channels=64).to(device)
    
    # Создаем оптимизатор
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.)
    
    # Создаем планировщик шума
    noise_scheduler = NoiseScheduler(num_timesteps=1000, device=device)
    
    # Создаем scaler для смешанной точности, если указано
    scaler = GradScaler() if args.mixed_precision else None
    
    # Загружаем checkpoint, если указан
    start_epoch = 0
    if args.checkpoint:
        if os.path.exists(args.checkpoint):
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            start_epoch = checkpoint['epoch'] + 1
            logger.info(f"Загружен checkpoint из {args.checkpoint}, начиная с эпохи {start_epoch}")
        else:
            logger.warning(f"Checkpoint {args.checkpoint} не найден, начинаем с начала")
    
    return model, optimizer, noise_scheduler, scaler, device, start_epoch


# ================================
# Добавленная функция train_epoch
# ================================
# Завершение функции train_epoch
def train_epoch(model, train_loader, optimizer, noise_scheduler, device, scaler, epoch, args):
    """Обучает модель на одной эпохе"""
    model.train()
    total_loss = 0
    pbar = tqdm(train_loader, desc=f"Эпоха {epoch+1}/{args.epochs}, Обучение")
    
    # Для накопления градиентов
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(pbar):
        lr_images = batch['lr'].to(device)
        hr_images = batch['hr'].to(device)
        lr_upscaled = batch['lr_upscaled'].to(device)
        
        # Нормализация изображений из [0, 1] в [-1, 1]
        hr_images = hr_images * 2 - 1
        lr_upscaled = lr_upscaled * 2 - 1
        
        batch_size = hr_images.shape[0]
        t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device).long()
        
        # Используем смешанную точность, если указано
        if args.mixed_precision:
            with autocast():
                loss = noise_scheduler.p_losses(model, hr_images, t, lr_upscaled)
                # Масштабируем потери в зависимости от шагов накопления градиента
                loss = loss / args.grad_accum_steps
                
            # Обратное распространение с накоплением градиентов
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                # Оптимизируем параметры модели
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        else:
            # Обычное обучение без смешанной точности
            loss = noise_scheduler.p_losses(model, hr_images, t, lr_upscaled)
            loss = loss / args.grad_accum_steps
            
            loss.backward()
            
            if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Обновляем статистику
        total_loss += loss.item() * args.grad_accum_steps
        
        # Обновляем прогресс-бар
        pbar.set_postfix(loss=loss.item(), avg_loss=total_loss/(batch_idx+1))
    
    # Возвращаем среднюю потерю по эпохе
    return total_loss / len(train_loader)


def validate(model, val_loader, noise_scheduler, device, epoch, args):
    """Валидирует модель на валидационном наборе данных"""
    model.eval()
    total_loss = 0
    total_psnr = 0
    total_samples = 0
    
    pbar = tqdm(val_loader, desc=f"Эпоха {epoch+1}/{args.epochs}, Валидация")
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(pbar):
            lr_images = batch['lr'].to(device)
            hr_images = batch['hr'].to(device)
            lr_upscaled = batch['lr_upscaled'].to(device)
            
            # Нормализация изображений из [0, 1] в [-1, 1]
            hr_images = hr_images * 2 - 1
            lr_upscaled = lr_upscaled * 2 - 1
            
            batch_size = hr_images.shape[0]
            
            # Для валидации используем равномерно распределенные временные шаги
            t = torch.randint(0, noise_scheduler.num_timesteps, (batch_size,), device=device).long()
            
            # Вычисляем функцию потерь
            if args.mixed_precision:
                with autocast():
                    loss = noise_scheduler.p_losses(model, hr_images, t, lr_upscaled)
            else:
                loss = noise_scheduler.p_losses(model, hr_images, t, lr_upscaled)
            
            total_loss += loss.item() * batch_size
            
            # Генерируем SR изображения для оценки качества (используем небольшое количество шагов для скорости)
            # Для валидации используем меньшее число шагов, чтобы ускорить процесс
            if batch_idx % 10 == 0:  # Проверяем только каждый 10-й батч для экономии времени
                with torch.no_grad():
                    # Используем только первое изображение из батча для генерации
                    sample_lr_upscaled = lr_upscaled[0:1]
                    
                    # Генерируем SR изображение с помощью 100 шагов вместо полных 1000
                    sr_image = noise_scheduler.p_sample_loop(
                        model, 
                        shape=(1, 3, args.image_size, args.image_size), 
                        cond=sample_lr_upscaled, 
                        device=device,
                        return_all_timesteps=False
                    )
                    
                    # Денормализуем изображения из [-1, 1] в [0, 1]
                    sr_image = (sr_image + 1) / 2
                    sr_hr_image = (hr_images[0:1] + 1) / 2
                    
                    # Вычисляем PSNR
                    mse = F.mse_loss(sr_image, sr_hr_image).item()
                    psnr = -10 * math.log10(mse + 1e-8)
                    
                    total_psnr += psnr
                    total_samples += 1
            
            # Обновляем прогресс-бар
            pbar.set_postfix(loss=loss.item(), avg_loss=total_loss/total_samples if total_samples > 0 else float('inf'))
    
    # Вычисляем средние метрики
    avg_loss = total_loss / len(val_loader.dataset)
    avg_psnr = total_psnr / total_samples if total_samples > 0 else 0
    
    return avg_loss, avg_psnr


def save_checkpoint(model, optimizer, epoch, loss, psnr, args, is_best=False):
    """Сохраняет чекпоинт модели"""
    # Создаем директорию для сохранения, если она не существует
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Путь для сохранения чекпоинта
    checkpoint_path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch}.pt")
    
    # Сохраняем состояние модели и оптимизатора
    checkpoint = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': loss,
        'psnr': psnr,
        'args': vars(args)
    }
    
    # Сохраняем чекпоинт
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Чекпоинт сохранен в {checkpoint_path}")
    
    # Если это лучшая модель, копируем её отдельно
    if is_best:
        best_path = os.path.join(args.output_dir, "best_model.pt")
        torch.save(checkpoint, best_path)
        logger.info(f"Лучшая модель сохранена в {best_path}")


def visualize_results(model, val_loader, noise_scheduler, device, epoch, args):
    """Визуализирует результаты модели на валидационном наборе"""
    model.eval()
    
    # Получаем батч изображений для визуализации
    batch = next(iter(val_loader))
    lr_images = batch['lr'].to(device)
    hr_images = batch['hr'].to(device)
    lr_upscaled = batch['lr_upscaled'].to(device)
    
    # Используем только несколько изображений для визуализации
    num_samples = min(4, lr_images.size(0))
    lr_images = lr_images[:num_samples]
    hr_images = hr_images[:num_samples]
    lr_upscaled = lr_upscaled[:num_samples]
    
    # Нормализация изображений из [0, 1] в [-1, 1]
    lr_upscaled_norm = lr_upscaled * 2 - 1
    
    # Генерируем SR изображения
    with torch.no_grad():
        sr_images = []
        for i in range(num_samples):
            sample_lr_upscaled = lr_upscaled_norm[i:i+1]
            
            # Используем меньшее количество шагов для визуализации
            sr_image = noise_scheduler.p_sample_loop(
                model, 
                shape=(1, 3, args.image_size, args.image_size), 
                cond=sample_lr_upscaled, 
                device=device,
                return_all_timesteps=False
            )
            
            # Денормализуем изображение из [-1, 1] в [0, 1]
            sr_image = (sr_image + 1) / 2
            sr_images.append(sr_image)
    
    # Создаем сетку изображений для визуализации
    fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
    
    for i in range(num_samples):
        # LR изображение (интерполированное)
        if num_samples > 1:
            axes[i, 0].imshow(lr_upscaled[i].permute(1, 2, 0).cpu().numpy())
            axes[i, 0].set_title("LR (Upscaled)")
            axes[i, 0].axis('off')
            
            # HR изображение (Ground Truth)
            axes[i, 1].imshow(hr_images[i].permute(1, 2, 0).cpu().numpy())
            axes[i, 1].set_title("HR (Ground Truth)")
            axes[i, 1].axis('off')
            
            # SR изображение (Предсказанное)
            axes[i, 2].imshow(sr_images[i][0].permute(1, 2, 0).cpu().numpy())
            axes[i, 2].set_title("SR (Generated)")
            axes[i, 2].axis('off')
        else:
            axes[0].imshow(lr_upscaled[i].permute(1, 2, 0).cpu().numpy())
            axes[0].set_title("LR (Upscaled)")
            axes[0].axis('off')
            
            axes[1].imshow(hr_images[i].permute(1, 2, 0).cpu().numpy())
            axes[1].set_title("HR (Ground Truth)")
            axes[1].axis('off')
            
            axes[2].imshow(sr_images[i][0].permute(1, 2, 0).cpu().numpy())
            axes[2].set_title("SR (Generated)")
            axes[2].axis('off')
    
    plt.tight_layout()
    
    # Создаем директорию для результатов, если она не существует
    os.makedirs(os.path.join(args.output_dir, "results"), exist_ok=True)
    
    # Сохраняем изображение
    plt.savefig(os.path.join(args.output_dir, "results", f"epoch_{epoch}_samples.png"))
    plt.close()


def test_model(model, test_loader, noise_scheduler, device, args):
    """Тестирует модель на тестовом наборе данных"""
    model.eval()
    total_psnr = 0
    total_ssim = 0
    
    # Импортируем библиотеку для вычисления SSIM
    try:
        from skimage.metrics import structural_similarity as ssim
        has_ssim = True
    except ImportError:
        logger.warning("scikit-image не установлен, SSIM не будет вычисляться")
        has_ssim = False
    
    pbar = tqdm(test_loader, desc="Тестирование")
    
    # Создаем директорию для результатов, если она не существует
    results_dir = os.path.join(args.output_dir, "test_results")
    os.makedirs(results_dir, exist_ok=True)
    
    with torch.no_grad():
        processed = 0
        for batch_idx, batch in enumerate(pbar):
            if batch_idx == 500:
                break
            lr_images = batch['lr'].to(device)
            hr_images = batch['hr'].to(device)
            lr_upscaled = batch['lr_upscaled'].to(device)
            
            # Нормализация изображений из [0, 1] в [-1, 1]
            lr_upscaled_norm = lr_upscaled * 2 - 1
            
            batch_size = lr_images.shape[0]
            
            # Для каждого изображения в батче
            for i in range(batch_size):
                # Генерируем SR изображение
                sr_image = noise_scheduler.p_sample_loop(
                    model, 
                    shape=(1, 3, args.image_size, args.image_size), 
                    cond=lr_upscaled_norm[i:i+1], 
                    device=device,
                    return_all_timesteps=False
                )
                
                # Денормализуем изображение из [-1, 1] в [0, 1]
                sr_image = (sr_image + 1) / 2
                
                # Вычисляем PSNR
                mse = F.mse_loss(sr_image[0], hr_images[i]).item()
                psnr = -10 * math.log10(mse + 1e-8)
                total_psnr += psnr
                
                # Вычисляем SSIM, если доступно
                if has_ssim:
                    # Преобразуем изображения в формат для skimage
                    sr_np = sr_image[0].permute(1, 2, 0).cpu().numpy()
                    hr_np = hr_images[i].permute(1, 2, 0).cpu().numpy()
                    
                    # Вычисляем SSIM для каждого канала и усредняем
                    ssim_value = 0
                    for c in range(3):  # RGB
                        ssim_value += ssim(sr_np[:, :, c], hr_np[:, :, c], data_range=1.0)
                    ssim_value /= 3
                    
                    total_ssim += ssim_value
                
                # Сохраняем изображения каждые 20 батчей для экономии дискового пространства 
                if batch_idx % 1 == 0 and i == 0:
                    # Создаем сетку изображений
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    
                    # LR изображение (интерполированное)
                    axes[0].imshow(lr_upscaled[i].permute(1, 2, 0).cpu().numpy())
                    axes[0].set_title("LR (Upscaled)")
                    axes[0].axis('off')
                    
                    # HR изображение (Ground Truth)
                    axes[1].imshow(hr_images[i].permute(1, 2, 0).cpu().numpy())
                    axes[1].set_title("HR (Ground Truth)")
                    axes[1].axis('off')
                    
                    # SR изображение (Предсказанное)
                    axes[2].imshow(sr_image[0].permute(1, 2, 0).cpu().numpy())
                    axes[2].set_title(f"SR (Generated)\nPSNR: {psnr:.2f} dB")
                    if has_ssim:
                        axes[2].set_title(f"SR (Generated)\nPSNR: {psnr:.2f} dB, SSIM: {ssim_value:.4f}")
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(results_dir, f"test_batch_{batch_idx}_sample.png"))
                    plt.close()
            
            # Вычисляем текущие средние метрики
            avg_psnr = total_psnr / ((batch_idx * batch_size) + i + 1)
            if has_ssim:
                avg_ssim = total_ssim / ((batch_idx * batch_size) + i + 1)
                pbar.set_postfix(psnr=avg_psnr, ssim=avg_ssim)
            else:
                pbar.set_postfix(psnr=avg_psnr)
    
    # Вычисляем средние метрики по всему тестовому набору
    total_samples = len(test_loader.dataset)
    avg_psnr = total_psnr / total_samples
    
    results = {
        'psnr': avg_psnr,
    }
    
    if has_ssim:
        avg_ssim = total_ssim / total_samples
        results['ssim'] = avg_ssim
    
    return results


def memory_cleanup():
    """Очищает память CUDA для экономии VRAM"""
    gc.collect()
    torch.cuda.empty_cache()


# ================================
# Основная функция для обучения модели
# ================================

# ================================
# Основная функция для обучения модели
# ================================

def main():
    # Парсим аргументы командной строки
    args = parser.parse_args()
    
    # Создаем директорию для вывода, если она не существует
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Настраиваем логирование в файл
    file_handler = logging.FileHandler(os.path.join(args.output_dir, 'training.log'))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    
    # Логируем параметры запуска
    logger.info(f"Аргументы: {args}")
    
    # Подготавливаем датасеты и DataLoader'ы
    logger.info("Подготавливаем датасеты...")
    train_loader, val_loader = prepare_dataloaders(args)
    
    # Настраиваем модель, оптимизатор и планировщик шума
    logger.info("Настраиваем модель и оптимизатор...")
    model, optimizer, noise_scheduler, scaler, device, start_epoch = setup_training(args)
    
    # Логирование информации о модели
    logger.info(f"Модель использует устройство: {device}")
    logger.info(f"Количество параметров модели: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Отслеживаем лучшую модель
    best_psnr = 0
    
    # Основной цикл обучения
    
    # Загружаем лучшую модель для тестирования
    logger.info("Загружаем лучшую модель для тестирования...")
    best_checkpoint = torch.load(os.path.join(args.output_dir, "best_model.pt"), map_location=device)
    model.load_state_dict(best_checkpoint['model'])
    
    # Создаем тестовый DataLoader (используем валидационный, если нет отдельного тестового)
    test_loader = val_loader
    
    # Тестируем модель
    logger.info("Тестируем модель...")
    test_results = test_model(model, test_loader, noise_scheduler, device, args)
    
    # Логируем результаты тестирования
    logger.info(f"Результаты тестирования: PSNR: {test_results['psnr']:.2f} dB")
    if 'ssim' in test_results:
        logger.info(f"SSIM: {test_results['ssim']:.4f}")
    
    logger.info("Обучение завершено!")


# ================================
# Точка входа для запуска скрипта
# ================================

if __name__ == "__main__":
    # Добавляем проверку наличия CUDA для экономии памяти
    if torch.cuda.is_available():
        torch.cuda.set_device(1)
        # Устанавливаем зерно для воспроизводимости
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Включаем оптимизации для CUDA
        torch.backends.cudnn.benchmark = True
        # Для экономии памяти при доступности меньшего количества VRAM
        torch.backends.cudnn.deterministic = False
        
        # Логируем информацию о доступной GPU
        logger.info(f"Доступно CUDA устройств: {torch.cuda.device_count()}")
        logger.info(f"Текущее CUDA устройство: {torch.cuda.current_device()}")
        logger.info(f"Имя CUDA устройства: {torch.cuda.get_device_name(torch.cuda.current_device())}")
        
        # Добавляем отображение доступной памяти GPU
        logger.info(f"Доступная память CUDA: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        logger.warning("CUDA недоступен, используется CPU (обучение будет очень медленным)")
    
    # Импортируем необходимые модули для очистки памяти
    import gc
    
    main()