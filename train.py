"""MNIST 手写数字识别 —— WideResNet-28 单模型训练。

日志与终端展示统一由 :mod:`log_utils` 提供：

* 终端：rich 彩色面板 / 表格 / 进度条
* 日志文件：``logs/mnist_YYYYmmdd_HHMMSS.log``（纯文本、含时间戳，便于事后复盘）

所有超参数都可用环境变量覆盖，方便先跑一次冒烟测试：

    MNIST_EPOCHS=2 MNIST_VAL_SIZE=512 python train.py
"""

from __future__ import annotations

import datetime
import os
import platform
import random
import time
from dataclasses import dataclass, fields

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from log_utils import (
    LogManager,
    fmt_bytes,
    fmt_count,
    fmt_duration,
    fmt_eta,
    fmt_int,
    fmt_lr,
    fmt_params,
    fmt_pct,
    fmt_signed,
    heat_style,
    model_summary,
    setup_logging,
)

# 旧常量保留为「未登记 / 旧 ckpt」的回退值
EMNIST_MEAN = 0.1722
EMNIST_STD = 0.3309

# torchvision 的 MNIST / EMNIST 数据集类都 **不提供** mean / std 属性或任何内置
# 归一化参数 —— 它们只负责下载、解压、返回 PIL.Image / tensor，归一化必须由使用者
# 通过 transforms.Normalize 显式完成。因此这里维护一张硬编码查找表。
DATASET_NORM: dict[str, tuple[float, float]] = {
    "mnist":           (0.1307, 0.3081),   # 社区广泛采用的标准值
    "emnist_mnist":    (0.1722, 0.3309),
    "emnist_digits":   (0.1722, 0.3309),   # 注意：这组值是 EMNIST Digits 的统计量
    "emnist_letters":  (0.1722, 0.3309),
    "emnist_balanced": (0.1722, 0.3309),
    "emnist_byclass":  (0.1722, 0.3309),
}


def dataset_norm(name: str) -> tuple[float, float]:
    """返回数据集对应的 (mean, std)；未注册的数据集直接报错。"""
    if name not in DATASET_NORM:
        raise ValueError(f"未注册的数据集：{name}")
    return DATASET_NORM[name]


# EMNIST 官方规范（NIST）规定其 idx 图像文件按 **转置** 方式存储，而
# ``torchvision.datasets.EMNIST`` 直接复用 ``MNIST`` 的加载逻辑，未做还原 ——
# 因此它返回的图像是「侧躺」的。这里对所有 EMNIST split 标记为需要转置。
DATASET_TRANSPOSE: dict[str, bool] = {
    "mnist":           False,
    "emnist_mnist":    True,
    "emnist_digits":   True,
    "emnist_letters":  True,
    "emnist_balanced": True,
    "emnist_byclass":  True,
}


def dataset_transpose(name: str) -> bool:
    """返回该数据集加载后是否需要转置还原；未注册的数据集直接报错。"""
    if name not in DATASET_TRANSPOSE:
        raise ValueError(f"未注册的数据集：{name}")
    return DATASET_TRANSPOSE[name]


def _maybe_transpose(tf_list: list, transpose: bool) -> list:
    """需还原朝向时，在 transform 链最前面插入 PIL 转置。

    必须是 ``PIL.Image.transpose``（不是 ``rotate(90)`` / ``rot90``）——它就是
    EMNIST 存储时所用变换的逆变换。转置只能作用于 PIL 图像，所以必须放在
    ``ToTensor`` **之前**。
    """
    if not transpose:
        return tf_list
    return [transforms.Lambda(lambda img: img.transpose(Image.TRANSPOSE))] + tf_list


def model_expects_transposed(trained_on: str | None,
                             trained_transpose: bool | None) -> bool | None:
    """模型期望的输入是否处于「转置（侧躺）」状态；``None`` 表示无法判断。

    一次训练后模型学到的朝向 = ``训练数据原生朝向 XOR 训练时是否施加过 transpose``：

    * 修复后的 train.py：EMNIST 源（原生转置）已在训练时还原成正向，
      于是模型期望 **正向** 输入；
    * 旧 ckpt（``trained_transpose is None``）：无从判断，调用方应保持既有行为。
    """
    if trained_transpose is None:
        return None
    train_native = DATASET_TRANSPOSE.get(trained_on or "", False)
    return bool(train_native) ^ bool(trained_transpose)


def needs_transpose(target: str, trained_on: str | None,
                    trained_transpose: bool | None) -> bool:
    """读取 ``target`` 数据集时是否要施加 transpose，才能与模型期望的朝向对齐。

    判定依据是「把目标数据的原生朝向对齐到模型期望的朝向」，因此结果既取决于
    数据源，也取决于模型的训练方式 —— 两者都由 ckpt 携带的信息推导得出：

    * 新 EMNIST ckpt（期望正向）→ EMNIST 源需转置、MNIST 源不转置；
    * 新 MNIST ckpt（期望正向）→ 同上；
    * 旧 ckpt（未记录标记）→ 返回 ``False``，完全保持修复前的行为。
    """
    expects = model_expects_transposed(trained_on, trained_transpose)
    if expects is None:
        return False
    return dataset_transpose(target) != expects


# ---------- 配置 ----------
@dataclass
class Config:
    """全部可调超参数；对应环境变量 ``MNIST_<字段名大写>`` 可覆盖。"""

    batch_size: int = 128
    eval_batch_size: int = 256
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    val_size: int = 5000
    epochs: int = 30
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    grad_clip: float = 5.0
    widen_factor: int = 2
    num_classes: int = 10
    dataset: str = "emnist_digits"  # 可选：mnist / emnist_digits（见 DATASET_NORM）
    aug_degrees: float = 5.0
    aug_translate: float = 0.05
    seed: int = 42
    log_every: int = 100          # 每 N 个 step 输出一行进度日志（0 = 关闭）
    tta: bool = True
    tta_views: int = 8
    tta_rot_deg: float = 7.0
    tta_translate: float = 0.08
    tta_scale: float = 0.04
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"

    @classmethod
    def from_env(cls) -> "Config":
        config = cls()
        for f in fields(cls):
            raw = os.getenv(f"MNIST_{f.name.upper()}")
            if raw in (None, ""):
                continue
            current = getattr(config, f.name)
            try:
                if isinstance(current, bool):
                    value = raw.strip().lower() in {"1", "true", "yes", "y", "on"}
                else:
                    value = type(current)(raw)
                setattr(config, f.name, value)
            except (TypeError, ValueError):
                pass
        config.log_dir = os.getenv("LOG_DIR", config.log_dir)
        return config

    def norm(self) -> tuple[float, float]:
        """当前 ``dataset`` 对应的训练 / 评估归一化 (mean, std)。"""
        return dataset_norm(self.dataset)

    def orientation_desc(self) -> str:
        """图像朝向的可读描述（与实际 transform 保持同源）。"""
        if dataset_transpose(self.dataset):
            return "已转置还原（EMNIST 规范）"
        return "原样（MNIST）"

    def augmentation_desc(self) -> str:
        """训练集数据增强的可读描述（与实际构建的 transform 保持同源）。"""
        return f"RandomAffine(±{self.aug_degrees:g}°, 平移 {self.aug_translate:.0%})"

    def rows(self) -> list[tuple[str, str]]:
        """转成日志面板用的键值对。"""
        return [
            ("batch_size", str(self.batch_size)),
            ("eval_batch_size", str(self.eval_batch_size)),
            ("num_workers", str(self.num_workers)),
            ("val_size", str(self.val_size)),
            ("epochs", str(self.epochs)),
            ("lr", fmt_lr(self.lr)),
            ("weight_decay", f"{self.weight_decay:.4f}"),
            ("label_smoothing", f"{self.label_smoothing:.4f}"),
            ("grad_clip", f"{self.grad_clip:.1f}"),
            ("widen_factor", str(self.widen_factor)),
            ("num_classes", str(self.num_classes)),
            ("dataset", self.dataset),
            ("归一化", f"mean={self.norm()[0]:.4f}, std={self.norm()[1]:.4f}"),
            ("图像朝向", self.orientation_desc()),
            ("seed", str(self.seed)),
            ("log_every", f"{self.log_every} steps"),
            ("训练增强", self.augmentation_desc()),
            ("tta", f"{'开启' if self.tta else '关闭'} · {self.tta_views} 视图"),
            ("tta 几何", f"rot ±{self.tta_rot_deg}°, trans ±{self.tta_translate}, scale ±{self.tta_scale}"),
        ]


# ---------- 工具函数 ----------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(log: LogManager) -> torch.device:
    """选择计算设备（无 CUDA 时回退 CPU 并给出提示）。"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    log.warning("CUDA 不可用，将使用 [bold]CPU[/] 运行，速度会明显变慢。")
    return torch.device('cpu')


def collect_env(device: torch.device) -> list[tuple[str, str]]:
    """收集运行环境信息，供日志面板展示。"""
    cuda_available = torch.cuda.is_available()
    items: list[tuple[str, str]] = [
        ("Python", platform.python_version()),
        ("操作系统", f"{platform.system()} {platform.release()}"),
        ("PyTorch", torch.__version__),
        ("CUDA 编译版本", torch.version.cuda or "—"),
        ("CUDA 可用", "[ok]是[/]" if cuda_available else "[warn]否[/]"),
    ]
    if device.type == 'cuda':
        props = torch.cuda.get_device_properties(0)
        device_name = props.name if torch.cuda.device_count() == 1 else f"{props.name} ×{torch.cuda.device_count()}"
        items += [
            ("设备", device_name),
            ("显存", fmt_bytes(props.total_memory)),
            ("计算能力", f"sm_{props.major}{props.minor}"),
            ("cuDNN", str(torch.backends.cudnn.version())),
        ]
    else:
        items.append(("设备", "[warn]cpu[/]"))
    items.append(("AMP 混合精度", "[ok]开启[/]" if device.type == 'cuda' else "[muted]关闭（CPU）[/]"))
    items.append(("进程 PID", str(os.getpid())))
    return items


# ---------- 数据 ----------
def build_loaders(config: Config, log: LogManager):
    """构建训练 / 验证 / 测试 DataLoader（数据集与预处理由 ``config.dataset`` 决定）。"""
    mean, std = dataset_norm(config.dataset)
    transpose = dataset_transpose(config.dataset)

    train_tf = transforms.Compose(_maybe_transpose([
        transforms.RandomAffine(degrees=config.aug_degrees,
                                translate=(config.aug_translate, config.aug_translate)),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ], transpose))
    eval_tf = transforms.Compose(_maybe_transpose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ], transpose))

    log.info(
        f"正在载入 / 下载 [accent]{config.dataset}[/] 数据集 · "
        f"归一化 mean={mean:.4f}, std={std:.4f} · "
        f"图像朝向 {'转置还原' if transpose else '原样'} …"
    )
    started = time.perf_counter()
    if config.dataset == "mnist":
        train_full = datasets.MNIST('./data', train=True,  download=True, transform=train_tf)
        val_full   = datasets.MNIST('./data', train=True,  download=True, transform=eval_tf)
        test_set   = datasets.MNIST('./data', train=False, download=True, transform=eval_tf)
        cache_dir = './data/MNIST'
    elif config.dataset == "emnist_digits":
        train_full = datasets.EMNIST('./data', split="digits", train=True,  download=True, transform=train_tf)
        val_full   = datasets.EMNIST('./data', split="digits", train=True,  download=True, transform=eval_tf)
        test_set   = datasets.EMNIST('./data', split="digits", train=False, download=True, transform=eval_tf)
        cache_dir = './data/EMNIST'
    else:
        raise ValueError(f"不支持的 dataset：{config.dataset}")
    log.info(
        f"数据集就绪 · 耗时 [accent]{fmt_duration(time.perf_counter() - started)}[/] · "
        f"缓存目录 [accent]{cache_dir}[/]"
    )

    val_size = min(config.val_size, len(train_full) - 1)
    train_size = len(train_full) - val_size
    perm = torch.randperm(len(train_full),
                          generator=torch.Generator().manual_seed(config.seed)).tolist()
    train_set = Subset(train_full, perm[:train_size])
    val_set   = Subset(val_full,   perm[train_size:])

    common = dict(num_workers=config.num_workers, pin_memory=config.pin_memory,
                  persistent_workers=(config.persistent_workers and config.num_workers > 0))
    train_loader = DataLoader(train_set, batch_size=config.batch_size,
                              shuffle=True, drop_last=True, **common)
    val_loader   = DataLoader(val_set,   batch_size=config.eval_batch_size,
                              shuffle=False, drop_last=False, **common)
    test_loader  = DataLoader(test_set,  batch_size=config.eval_batch_size,
                              shuffle=False, drop_last=False, **common)

    stats = {
        "train_size": train_size,
        "val_size": val_size,
        "test_size": len(test_set),
        "steps_per_epoch": len(train_loader),
        # 记录真正传给 DataLoader 的参数，避免日志与实际行为脱节
        "loader": {
            "num_workers": common["num_workers"],
            "pin_memory": common["pin_memory"],
            "persistent_workers": common["persistent_workers"],
            "drop_last": (train_loader.drop_last, val_loader.drop_last),
        },
    }
    return train_loader, val_loader, test_loader, stats


def log_dataset(log: LogManager, config: Config, stats: dict) -> None:
    """输出数据划分与 DataLoader 配置。"""
    mean, std = dataset_norm(config.dataset)
    transpose = dataset_transpose(config.dataset)
    loader = stats["loader"]
    log.section("数据集")
    log.table(
        f"{config.dataset} 数据划分",
        ["划分", ("样本数", "right"), ("批大小", "right"), "数据增强", "用途"],
        [
            ["训练集", fmt_count(stats['train_size']), config.batch_size, config.augmentation_desc(), "参数更新"],
            ["验证集", fmt_count(stats['val_size']), config.eval_batch_size, "无", "选最优模型"],
            ["测试集", fmt_count(stats['test_size']), config.eval_batch_size, "无", "最终评估"],
        ],
        caption=(f"划分方式：固定随机置换（seed={config.seed}）· "
                 f"归一化 dataset={config.dataset} · mean={mean:.4f}, std={std:.4f} · "
                 f"图像朝向 {'已转置还原（EMNIST 规范）' if transpose else '原样（MNIST）'}"),
    )
    log.kv(
        "DataLoader 设置",
        [
            ("num_workers", str(loader["num_workers"])),
            ("pin_memory", str(loader["pin_memory"])),
            ("persistent_workers", str(loader["persistent_workers"])),
            ("drop_last", f"训练集 {loader['drop_last'][0]} / 验证、测试 {loader['drop_last'][1]}"),
            ("一个 epoch 的 step 数", str(stats['steps_per_epoch'])),
            ("总训练步数", f"{stats['steps_per_epoch'] * config.epochs:,}"),
        ],
        columns=2,
        border_style="muted",
    )



# ---------- 模型 ----------
class WideResNetBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.shortcut = nn.Identity()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out, inplace=True)


class WideResNetMNIST(nn.Module):
    def __init__(self, widen_factor: int = 2, num_classes: int = 10):
        super().__init__()
        n = 32 * widen_factor
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(1, n, 3, 1, 1)
        self.layer1 = WideResNetBlock(n, n)
        self.layer2 = WideResNetBlock(n, n * 2, stride=2)
        self.layer3 = WideResNetBlock(n * 2, n * 4, stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(n * 4, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x), inplace=True)
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.pool(x).flatten(1)
        return self.fc(x)


# ---------- 训练 / 评估 ----------
@torch.no_grad()
def evaluate(model, loader, device, criterion=None) -> tuple[float, float]:
    """返回 (平均损失, 准确率)；criterion 为 None 时只统计准确率。"""
    model.eval()
    use_amp = (device.type == 'cuda')
    total_loss = correct = total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(x)
            loss = criterion(out, y) if criterion is not None else None
        if loss is not None:
            total_loss += loss.item() * y.size(0)
        correct += (out.argmax(1) == y).sum().item()
        total += y.size(0)
    avg_loss = total_loss / total if total else float('nan')
    return avg_loss, (correct / total if total else 0.0)


def log_history(log: LogManager, tag: str, history: list[dict],
                best_epoch: int, best_val: float, total_secs: float) -> None:
    """输出逐轮指标表格与训练摘要面板。"""
    rows = []
    for record in history:
        rows.append([
            record['epoch'],
            f"{record['train_loss']:.4f}",
            fmt_pct(record['train_acc']),
            f"{record['val_loss']:.4f}",
            fmt_pct(record['val_acc']),
            f"{record['grad_norm']:.2f}",
            fmt_lr(record['lr']),
            fmt_duration(record['secs']),
            "[ok]★ 最优[/]" if record['epoch'] == best_epoch else "",
        ])

    log.table(
        f"{tag} · 逐轮指标",
        ["Epoch", "训练损失", "训练准确率", "验证损失", "验证准确率", "|grad|", "学习率", "耗时", "备注"],
        rows,
        caption=(f"最优 epoch {best_epoch} · val_acc {fmt_pct(best_val)} · "
                 f"总耗时 {fmt_duration(total_secs)} · 平均 {fmt_duration(total_secs / max(1, len(history)))}/epoch"),
        zebra=True,
    )

    first, last = history[0], history[-1]
    items = [
        ("最优验证准确率", f"[metric]{fmt_pct(best_val)}[/]（epoch {best_epoch}）"),
        ("最终 epoch", f"{fmt_pct(last['val_acc'])} · val_loss {last['val_loss']:.4f}"),
        ("训练损失", f"{first['train_loss']:.4f} → {last['train_loss']:.4f}"),
        ("吞吐", f"{last['it_s']:.1f} it/s"),
        ("总耗时", f"{fmt_duration(total_secs)} · 平均 {fmt_duration(total_secs / max(1, len(history)))}/epoch"),
    ]
    if torch.cuda.is_available():
        items.append(("显存", f"已分配 {fmt_bytes(torch.cuda.memory_allocated())} · 峰值 {fmt_bytes(torch.cuda.max_memory_allocated())}"))
    log.kv(f"{tag} · 训练完成", items, columns=2, border_style="green")


def train_one_model(model, train_loader, val_loader, device, config: Config,
                    log: LogManager, *, tag: str = "模型"):
    """训练单个模型，返回 (model, best_val, history)。"""
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    use_amp = (device.type == 'cuda')
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

    log.kv(
        "训练设置",
        [
            ("优化器", f"AdamW(lr={fmt_lr(config.lr)}, weight_decay={config.weight_decay})"),
            ("调度器", "CosineAnnealingWarmRestarts(T_0=10, T_mult=2)"),
            ("损失函数", f"CrossEntropyLoss(label_smoothing={config.label_smoothing})"),
            ("梯度处理", f"clip_grad_norm_(max_norm={config.grad_clip})" if config.grad_clip else "不裁剪"),
            ("混合精度", "torch.amp.autocast + GradScaler" if use_amp else "[muted]关闭（CPU）[/]"),
            ("训练规模", f"{config.epochs} epochs × {len(train_loader)} steps"),
        ],
        columns=1,
        border_style="muted",
    )

    history: list[dict] = []
    best_val, best_epoch, best_state = -1.0, 0, None
    model_start = time.perf_counter()

    for epoch in range(config.epochs):
        epoch_start = time.perf_counter()
        epoch_lr = optimizer.param_groups[0]['lr']
        model.train()
        run_loss = correct = total = 0
        grad_sum = 0.0
        description = f"{tag} · Epoch {epoch + 1:02d}/{config.epochs}"

        with log.progress() as progress:
            task = log.add_task(progress, description, total=len(train_loader), stats="等待首个 batch …")
            for step, (x, y) in enumerate(train_loader, start=1):
                x = x.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast('cuda', enabled=use_amp):
                    out = model(x)
                    loss = criterion(out, y)

                scaler.scale(loss).backward()
                if config.grad_clip:
                    scaler.unscale_(optimizer)  # 反缩放后才能统计 / 裁剪真实梯度
                    grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip))
                else:
                    grad_norm = 0.0
                scaler.step(optimizer)
                scaler.update()

                batch = y.size(0)
                run_loss += loss.item() * batch
                correct += (out.argmax(1) == y).sum().item()
                total += batch
                grad_sum += grad_norm

                elapsed = time.perf_counter() - epoch_start
                it_s = step / elapsed if elapsed > 0 else 0.0
                eta = (len(train_loader) - step) / it_s if it_s > 0 else 0.0
                progress.update(
                    task, advance=1,
                    stats=(f"loss [metric]{run_loss / total:.4f}[/] │ "
                           f"acc [metric]{fmt_pct(correct / total)}[/] │ "
                           f"|g| {grad_sum / step:.2f} │ {it_s:.1f} it/s"),
                )

                if config.log_every and step % config.log_every == 0:
                    log.info(
                        f"  step {step:>4}/{len(train_loader)} │ loss {run_loss / total:.4f} │ "
                        f"acc {fmt_pct(correct / total)} │ |g| {grad_sum / step:.2f} │ "
                        f"{it_s:.1f} it/s │ ETA {fmt_eta(eta)}"
                    )
                elif log.debug_enabled:
                    log.debug(f"  step {step:>4} │ loss {loss.item():.4f} │ |g| {grad_norm:.3f}")

        epoch_secs = time.perf_counter() - epoch_start
        scheduler.step()
        val_loss, val_acc = evaluate(model, val_loader, device, criterion)
        improved = val_acc > best_val
        if improved:
            best_val, best_epoch = val_acc, epoch + 1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append({
            "epoch": epoch + 1,
            "train_loss": run_loss / total,
            "train_acc": correct / total,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "grad_norm": grad_sum / max(1, len(train_loader)),
            "lr": epoch_lr,
            "secs": epoch_secs,
            "it_s": len(train_loader) / epoch_secs if epoch_secs > 0 else 0.0,
        })
        log.info(
            f"[bold]Epoch {epoch + 1:>2}/{config.epochs}[/] │ "
            f"train_loss {run_loss / total:.4f} │ train_acc {fmt_pct(correct / total)} │ "
            f"val_loss {val_loss:.4f} │ val_acc [metric]{fmt_pct(val_acc)}[/] │ "
            f"lr {fmt_lr(epoch_lr)} │ |g| {history[-1]['grad_norm']:.2f} │ "
            f"{fmt_duration(epoch_secs)} · {history[-1]['it_s']:.1f} it/s"
            + (" │ [ok]★ 最优[/]" if improved else "")
        )

    if best_state is not None:
        model.load_state_dict(best_state)

    total_secs = time.perf_counter() - model_start
    log_history(log, tag, history, best_epoch, best_val, total_secs)
    return model, best_val, history


# ---------- 推理 ----------
@torch.no_grad()
def predict(model, loader, device):
    """返回 (预测标签, 真实标签)。"""
    model.eval()
    use_amp = (device.type == 'cuda')
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast('cuda', enabled=use_amp):
            out = model(x)
        preds.append(out.float().argmax(1).cpu())
        targets.append(y)
    return torch.cat(preds), torch.cat(targets)


def _affine_view_raw(x_raw, angle=0.0, dx=0.0, dy=0.0, scale=1.0):
    """对 [0, 1] 图像 batch 施加旋转、平移和缩放。

    dx/dy 使用整图比例，与 transforms.RandomAffine 的 translate 语义一致；
    affine_grid 的归一化坐标以半宽/半高为 1，因此传入 theta 前要乘 2。
    """
    angle_rad = torch.as_tensor(angle * torch.pi / 180.0, device=x_raw.device, dtype=x_raw.dtype)
    cos_angle = torch.cos(angle_rad) / scale
    sin_angle = torch.sin(angle_rad) / scale
    theta = torch.zeros((x_raw.size(0), 2, 3), device=x_raw.device, dtype=x_raw.dtype)
    # affine_grid 使用输出到输入的逆映射，因此正角度的视觉旋转方向相反。
    theta[:, 0, 0] = cos_angle
    theta[:, 0, 1] = sin_angle
    theta[:, 0, 2] = 2.0 * dx
    theta[:, 1, 0] = -sin_angle
    theta[:, 1, 1] = cos_angle
    theta[:, 1, 2] = 2.0 * dy
    grid = F.affine_grid(theta, x_raw.size(), align_corners=False)
    return F.grid_sample(x_raw, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def make_tta_views(x_norm, config: Config, mean: float | None = None, std: float | None = None):
    """返回归一化后的原图及多个不改变语义的几何视图。

    ``mean`` / ``std`` 必须与 ``x_norm`` 使用的归一化一致：TTA 会先反归一化回
    [0, 1] 再做几何变换，最后按同一组参数重新归一化。缺省时回退到
    ``config.dataset`` 注册的统计量，保持对旧调用方（如 predict.py）的兼容。
    """
    if mean is None or std is None:
        mean, std = dataset_norm(config.dataset)
    if not config.tta or config.tta_views <= 1:
        return [x_norm]

    x_raw = (x_norm * std + mean).clamp(0.0, 1.0)
    rot = config.tta_rot_deg
    translate = config.tta_translate
    scale = config.tta_scale
    transforms_to_apply = [
        (rot, 0.0, 0.0, 1.0),
        (-rot, 0.0, 0.0, 1.0),
        (0.0, translate, 0.0, 1.0),
        (0.0, -translate, 0.0, 1.0),
        (0.0, 0.0, translate, 1.0),
        (0.0, 0.0, -translate, 1.0),
        (0.0, 0.0, 0.0, 1.0 + scale),
        (0.0, 0.0, 0.0, 1.0 - scale),
        (rot, translate, 0.0, 1.0),
        (-rot, -translate, 0.0, 1.0),
        (rot, 0.0, translate, 1.0),
        (-rot, 0.0, -translate, 1.0),
    ]
    views = [x_norm]
    for angle, dx, dy, view_scale in transforms_to_apply[:config.tta_views - 1]:
        view = _affine_view_raw(x_raw, angle, dx, dy, view_scale)
        views.append((view - mean) / std)
    return views


@torch.no_grad()
def predict_tta(model, loader, device, config: Config,
                mean: float | None = None, std: float | None = None):
    """对测试集的多个视图平均 softmax 概率后预测。

    ``mean`` / ``std`` 为本次评估所用归一化，必须与 loader 的 transform 一致。
    """
    model.eval()
    use_amp = (device.type == 'cuda')
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        probabilities = []
        for view in make_tta_views(x, config, mean, std):
            with torch.amp.autocast('cuda', enabled=use_amp):
                out = model(view)
            probabilities.append(F.softmax(out.float(), dim=1))
        preds.append(torch.stack(probabilities).mean(dim=0).argmax(1).cpu())
        targets.append(y)
    return torch.cat(preds), torch.cat(targets)


def log_test_results(model, best_val, test_loader, device, log: LogManager,
                     config: Config | None = None,
                     mean: float | None = None, std: float | None = None) -> dict:
    """测试集评估：验证/测试表现对比、各类别准确率、混淆矩阵。

    ``mean`` / ``std`` 需与构建 ``test_loader`` 时一致（TTA 反归一化要用）。
    """
    log.section("测试集评估")
    use_tta = config is not None and config.tta and config.tta_views > 1

    if use_tta:
        log.info(
            f"测试集启用 TTA：共 {config.tta_views} 个视图（原图 + {config.tta_views - 1} 个几何变换） · "
            f"rot ±{config.tta_rot_deg}° · trans ±{config.tta_translate} · scale ±{config.tta_scale}"
        )
    with log.progress() as progress:
        task = log.add_task(progress, "测试集 TTA 推理" if use_tta else "测试集推理",
                            total=len(test_loader), stats="")
        if use_tta:
            preds, targets = predict_tta(model, test_loader, device, config, mean, std)
        else:
            preds, targets = predict(model, test_loader, device)
        progress.update(task, advance=len(test_loader), stats=f"样本 {len(targets)}")

    test_acc = (preds == targets).float().mean().item()

    log.table(
        "验证集 / 测试集表现对比",
        ["数据集", ("准确率", "right")],
        [
            ["验证集（最优）", fmt_pct(best_val)],
            [f"测试集（TTA {config.tta_views} 视图）" if use_tta else "测试集",
             f"[metric]{fmt_pct(test_acc)}[/]"],
        ],
        caption="验证集用于选最优权重，测试集仅做最终评估。",
        zebra=True,
    )

    log.metrics([
        ("测试准确率", fmt_pct(test_acc)),
        ("最优验证准确率", fmt_pct(best_val)),
        ("验证 − 测试", f"{fmt_signed((best_val - test_acc) * 100, 2)}%"),
    ])

    # ---- 逐类别准确率 ----
    # 类别数从模型读取（兼容被 DataParallel 包裹的情况），避免与网络定义脱节
    unwrapped = getattr(model, "module", model)
    model_classes = getattr(unwrapped, "num_classes", None) or unwrapped.fc.out_features
    label_classes = int(targets.max().item()) + 1 if targets.numel() else 0
    if label_classes > model_classes:
        log.warning(
            f"标签最大值 {label_classes - 1} 超出模型输出维度 {model_classes}，"
            "混淆矩阵将按标签范围扩展，请检查 num_classes 配置。"
        )
    num_classes = max(model_classes, label_classes)
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.long)
    for target, pred in zip(targets.tolist(), preds.tolist()):
        confusion[target, pred] += 1

    per_class_rows = []
    for digit in range(num_classes):
        row_total = int(confusion[digit].sum().item())
        row_correct = int(confusion[digit, digit].item())
        mistakes = confusion[digit].clone()
        mistakes[digit] = 0
        worst = int(mistakes.max().item())
        hardest = f"被预测为 {int(mistakes.argmax())}（{worst} 次）" if worst else "[ok]零错误[/]"
        per_class_rows.append([
            str(digit), fmt_int(row_total), fmt_int(row_correct),
            fmt_pct(row_correct / row_total if row_total else 0.0), fmt_int(worst), hardest,
        ])
    log.table("各类别准确率",
              ["类别", ("样本数", "right"), ("正确", "right"), ("准确率", "right"), ("错误", "right"), "最易混淆"],
              per_class_rows)

    # ---- 混淆矩阵（热力着色） ----
    vmax = int(confusion.max().item()) or 1
    matrix_rows = []
    for digit in range(num_classes):
        row = [f"[head]{digit}[/]"]
        for predicted in range(num_classes):
            value = int(confusion[digit, predicted].item())
            row.append(f"[muted]{'·':>5}[/]" if value == 0 else f"[{heat_style(value, vmax)}]{value:>5}[/]")
        matrix_rows.append(row)
    log.table("混淆矩阵（行 = 真实标签，列 = 预测标签）",
              [""] + [str(d) for d in range(num_classes)], matrix_rows,
              caption="颜色越亮代表样本越多；对角线即预测正确的样本。")

    return {"test_acc": test_acc, "best_val": best_val}


# ---------- 主程序 ----------
def main() -> int:
    config = Config.from_env()
    log = setup_logging(name="MNIST", level="INFO", log_dir=config.log_dir)

    log.banner(
        "MNIST 手写数字识别",
        f"WideResNet-28(widen={config.widen_factor}) · {config.dataset} · 单模型训练 · rich 日志",
    )
    log.info(f"详细日志文件：[accent]{log.log_path}[/]")
    log.info(
        "小贴士：环境变量 [accent]LOG_LEVEL=DEBUG[/] 可查看逐 step 调试信息，"
        "[accent]MNIST_LOG_EVERY=50[/] 可调整周期日志间隔，"
        "[accent]MNIST_EPOCHS=2 MNIST_VAL_SIZE=512[/] 可先跑一次冒烟测试。"
    )
    log.blank()

    # 提前校验 dataset，避免跑到一半才报错
    try:
        mean, std = dataset_norm(config.dataset)
        transpose = dataset_transpose(config.dataset)
    except ValueError as exc:
        log.error(f"{exc}（可选：{', '.join(DATASET_NORM)}）")
        log.close()
        return 2

    set_seed(config.seed)
    device = get_device(log)

    log.section("运行环境")
    log.kv("环境信息", collect_env(device), columns=2)
    log.section("运行配置")
    log.kv("超参数", config.rows(), columns=2, footer=f"seed={config.seed}")

    if not log.ask_yes_no("开始训练？", default=True):
        log.warning("已取消训练。")
        log.close()
        return 0

    train_loader, val_loader, test_loader, stats = build_loaders(config, log)
    log_dataset(log, config, stats)

    log.section("模型结构")
    log.info("用一个模板模型统计各子模块的形状与参数分布（不计入训练）。")
    template = WideResNetMNIST(widen_factor=config.widen_factor,
                              num_classes=config.num_classes).to(device)
    model_summary(log, template, (1, 1, 28, 28),
                  title=f"WideResNetMNIST(widen_factor={config.widen_factor}, "
                        f"num_classes={config.num_classes})")
    del template
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    log.section("模型训练")
    set_seed(config.seed)  # 再次固定种子，保证模型初始化可复现
    model = WideResNetMNIST(widen_factor=config.widen_factor,
                            num_classes=config.num_classes).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        f"模型已创建 · 参数量 [accent]{fmt_params(n_params)}[/] · "
        f"设备 {device.type} · seed={config.seed}"
    )

    train_started = time.perf_counter()
    model, best_val, history = train_one_model(
        model, train_loader, val_loader, device, config, log, tag="单模型")
    train_secs = time.perf_counter() - train_started
    log.success(f"训练完成 · 最优验证准确率 {fmt_pct(best_val)} · 耗时 {fmt_duration(train_secs)}")

    os.makedirs(config.checkpoint_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(config.checkpoint_dir, f"wrn_mnist_{timestamp}.pt")
    # ckpt 自带训练时的归一化与朝向信息，test.py / predict.py 直接读取，
    # 避免跳数据集评估或推理真实图片时输入分布 / 朝向错位。
    torch.save({
        "model": model,               # 保持 torch.save(model, ...) 的完整模型对象
        "dataset": config.dataset,
        "mean": mean,
        "std": std,
        "transpose": transpose,       # 模型期望的输入朝向（EMNIST 需转置还原）
        "widen_factor": config.widen_factor,
        "num_classes": config.num_classes,
    }, ckpt_path)
    log.debug(f"已保存 {ckpt_path} · {fmt_bytes(os.path.getsize(ckpt_path))}")
    log.success(f"模型权重已保存至 [accent]{ckpt_path}[/]")
    log.info(
        f"ckpt 已记录预处理：dataset={config.dataset} · "
        f"mean={mean:.4f} · std={std:.4f} · transpose={transpose}"
    )

    results = log_test_results(model, best_val, test_loader, device, log, config, mean, std)

    best_record = max(history, key=lambda record: record['val_acc'])
    log.section("运行汇总")
    log.kv(
        "本次运行",
        [
            ("总耗时", fmt_duration(log.elapsed)),
            ("测试准确率", f"[metric]{fmt_pct(results['test_acc'])}[/]"),
            ("最优验证准确率", f"{fmt_pct(best_val)}（epoch {best_record['epoch']}）"),
            ("模型", f"WideResNetMNIST(widen_factor={config.widen_factor}, "
                    f"num_classes={config.num_classes}) · {fmt_params(n_params)} 参数"),
            ("训练耗时", f"{fmt_duration(train_secs)} · 平均 "
                        f"{fmt_duration(train_secs / max(1, len(history)))}/epoch"),
            ("检查点", ckpt_path),
            ("日志文件", str(log.log_path)),
        ],
        columns=1,
        border_style="ok",
    )
    log.success(f"全部流程结束 🎉 测试准确率 {fmt_pct(results['test_acc'])}")
    log.close()
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\n已被用户手动中断。")
        raise SystemExit(130)