#!/usr/bin/env python
"""batch_inference.py —— 对 ./data/Reasoning 下的图片做批量手写数字识别。

典型用法
--------
    # 1) 默认：./data/Reasoning 下所有图片 + ./checkpoints下最新的 .pt
    python scripts/predict.py

    # 2) 显式指定权重与数据目录
    python scripts/predict.py --ckpt checkpoints/wrn_mnist_20260921_120000.pt \
                              --data-dir ./data/Reasoning

    # 3) 图片是白底黑字（需要反色）
    python scripts/predict.py --invert

    # 4) 开启 TTA + 输出 CSV
    python scripts/predict.py --tta --tta-views 8 --output results.csv

    # 5) 强制指定图像朝向（默认从 ckpt 自动读取）
    python scripts/predict.py --transpose yes
"""
from __future__ import annotations

import argparse
import csv
import datetime as _dt
import time
from pathlib import Path
from typing import NamedTuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# ---------------------------------------------------------------------------
# 1) 复用训练脚本中的模型定义与常量
#    train.py 的 main() 在 __main__ 保护下，import 时不会触发训练
# ---------------------------------------------------------------------------
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/..')

from train import (
    Config,
    EMNIST_MEAN,
    EMNIST_STD,
    WideResNetBlock,
    WideResNetMNIST,
    make_tta_views,
    model_expects_transposed,
)

# 2) 兼容 `python train.py` 直接保存出的 ckpt：
#    pickle 里记录的类路径可能是 `__main__.WideResNetMNIST`，
#    这里把同名符号绑定到本脚本的 __main__ 命名空间，
#    反序列化时就能找到。
import __main__ as _main_ns

_main_ns.WideResNetMNIST = WideResNetMNIST
_main_ns.WideResNetBlock = WideResNetBlock

# ---------------------------------------------------------------------------
# 3) 复用日志工具
# ---------------------------------------------------------------------------
from log_utils import (
    fmt_bytes,
    fmt_duration,
    fmt_params,
    fmt_pct,
    setup_logging,
)

IMAGE_EXTS = {
    ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff",
    ".webp", ".gif", ".ppm", ".pgm",
}


# ---------------------------------------------------------------------------
# 设备与 checkpoints发现
# ---------------------------------------------------------------------------
def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def find_latest_ckpt(directory: Path) -> Path | None:
    if not directory.exists():
        return None
    candidates = [p for p in directory.rglob("*.pt") if p.is_file()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


# ---------------------------------------------------------------------------
# 数据读取
# ---------------------------------------------------------------------------
def discover_images(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    )


def filter_valid(paths: list[Path]) -> tuple[list[Path], list[Path]]:
    """预校验图片可读性，避免个别坏文件中断整批推理。"""
    valid, bad = [], []
    for p in paths:
        try:
            with Image.open(p) as im:
                im.verify()
            valid.append(p)
        except Exception:
            bad.append(p)
    return valid, bad


def build_transform(invert: bool, transpose: bool,
                    mean: float = EMNIST_MEAN, std: float = EMNIST_STD) -> transforms.Compose:
    """构建单图预处理：transpose → resize → invert → normalize。

    ``transpose`` 必须是 PIL 转置，所以放在 ``ToTensor`` 之前；它来自 ckpt
    （模型期望的输入朝向），与图片本身无关。
    """
    ops: list = []
    if transpose:
        ops.append(transforms.Lambda(lambda img: img.transpose(Image.TRANSPOSE)))
    ops += [transforms.Resize((28, 28)), transforms.ToTensor()]
    if invert:
        ops.append(transforms.Lambda(lambda t: 1.0 - t))
    ops.append(transforms.Normalize((mean,), (std,)))
    return transforms.Compose(ops)


class ImageList(Dataset):
    def __init__(self, paths: list[Path], transform) -> None:
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        path = self.paths[index]
        with Image.open(path) as im:
            im = im.convert("L")
            x = self.transform(im)
        return x, index


# ---------------------------------------------------------------------------
# 模型加载
# ---------------------------------------------------------------------------
class CkptInfo(NamedTuple):
    """从 ckpt 读出的模型及其自带的预处理信息。"""

    model: torch.nn.Module
    mean: float
    std: float
    dataset: str | None    # 训练所用数据集
    transpose: bool | None  # 训练时是否施加过 transpose；None = 未记录
    norm_recorded: bool     # 归一化是否由 ckpt 提供（否则为回退默认）


def load_model(ckpt_path: Path, device: torch.device, log) -> CkptInfo:
    log.info(f"载入权重 [accent]{ckpt_path}[/] …")
    try:
        obj = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        # PyTorch < 2.6 没有 weights_only 参数
        obj = torch.load(ckpt_path, map_location=device)

    mean, std = EMNIST_MEAN, EMNIST_STD
    transpose: bool | None = None
    dataset: str | None = None
    norm_recorded = False
    if isinstance(obj, torch.nn.Module):
        model = obj
    elif isinstance(obj, dict) and "model" in obj:
        model = obj["model"]
        dataset = obj.get("dataset")
        if "mean" in obj and "std" in obj:
            mean = float(obj["mean"])
            std = float(obj["std"])
            norm_recorded = True
            log.info(
                f"ckpt 自带归一化：dataset={dataset} · "
                f"mean={mean:.4f} · std={std:.4f}"
            )
        if "transpose" in obj:
            transpose = bool(obj["transpose"])
    elif isinstance(obj, dict):
        # 兼容 state_dict 保存方式：从键形状反推 widen_factor / num_classes
        if "conv1.weight" not in obj:
            raise TypeError("无法识别的 checkpoint：既不是 nn.Module 也不是 state_dict。")
        n = int(obj["conv1.weight"].shape[0])
        widen = max(1, n // 32)
        num_classes = int(obj["fc.weight"].shape[0]) if "fc.weight" in obj else 10
        model = WideResNetMNIST(widen_factor=widen, num_classes=num_classes)
        model.load_state_dict(obj)
        log.warning("检测到 state_dict 格式，已按结构重建模型。")
    else:
        raise TypeError(f"无法识别的 checkpoints类型：{type(obj)}")

    if transpose is None:
        log.warning(
            "ckpt 未携带 transpose 标记，按旧行为处理（不对输入做转置）。"
            "若该 ckpt 由修复前的 EMNIST 训练得到，它期望的是侧躺输入。"
        )
    model = model.to(device).eval()
    return CkptInfo(model=model, mean=mean, std=std, dataset=dataset,
                    transpose=transpose, norm_recorded=norm_recorded)


# ---------------------------------------------------------------------------
# 推理
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_inference(model, loader, device, tta_config, log,
                  mean: float = EMNIST_MEAN, std: float = EMNIST_STD) -> torch.Tensor:
    """``mean`` / ``std`` 必须与构建 transform 时一致（TTA 反归一化要用）。"""
    model.eval()
    use_amp = (device.type == "cuda")
    autocast_device = device.type
    probs_all: list[torch.Tensor] = []
    total = len(loader.dataset)
    done = 0
    t0 = time.perf_counter()

    with log.progress() as progress:
        task = log.add_task(progress, "批量推理", total=len(loader), stats="等待首个 batch …")
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            if tta_config is not None and tta_config.tta and tta_config.tta_views > 1:
                views = make_tta_views(x, tta_config, mean, std)
                stacked = []
                for view in views:
                    with torch.amp.autocast(autocast_device, enabled=use_amp):
                        out = model(view)
                    stacked.append(F.softmax(out.float(), dim=1))
                probs = torch.stack(stacked).mean(dim=0)
            else:
                with torch.amp.autocast(autocast_device, enabled=use_amp):
                    out = model(x)
                probs = F.softmax(out.float(), dim=1)

            probs_all.append(probs.cpu())
            done += x.size(0)
            elapsed = max(1e-9, time.perf_counter() - t0)
            it_s = done / elapsed
            progress.update(
                task, advance=1,
                stats=f"{done}/{total} · {it_s:.1f} img/s",
            )

    return torch.cat(probs_all)  # 顺序与 loader 一致（shuffle=False）


# ---------------------------------------------------------------------------
# 结果输出
# ---------------------------------------------------------------------------
def write_csv(path: Path, images, preds, confs, probs, topk: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    num_classes = probs.shape[1]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["filename", "prediction", "confidence", f"top{topk}"]
            + [f"p{i}" for i in range(num_classes)]
        )
        topk = max(1, min(topk, num_classes))
        for p, pred, conf, prob in zip(images, preds.tolist(), confs.tolist(), probs.tolist()):
            order = sorted(range(num_classes), key=lambda i: prob[i], reverse=True)[:topk]
            top_str = " ".join(f"{i}:{prob[i]:.4f}" for i in order)
            writer.writerow(
                [p.name, pred, f"{conf:.6f}", top_str]
                + [f"{v:.6f}" for v in prob]
            )


# ---------------------------------------------------------------------------
# 参数
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="对 ./data/Reasoning 下的图片做批量手写数字识别。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt", type=str, default=None,
                        help="模型 .pt 路径；不指定则取 --ckpt-dir 下最新的 .pt")
    parser.add_argument("--ckpt-dir", type=str, default="./checkpoints",
                        help="checkpoints目录")
    parser.add_argument("--data-dir", type=str, default="./data/Reasoning",
                        help="待识别图片目录")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", type=str, default=None,
                        help="cuda / cpu；默认自动选择")
    parser.add_argument("--invert", action="store_true",
                        help="反色：用于白底黑字图片（MNIST 为黑底白字）")
    parser.add_argument("--transpose", choices=["auto", "yes", "no"], default="auto",
                        help="图像朝向：auto 按 ckpt 推导（默认），yes/no 强制覆盖")
    parser.add_argument("--tta", action="store_true", help="启用测试时增强")
    parser.add_argument("--tta-views", type=int, default=8,
                        help="TTA 视图数（含原图，至少 2）")
    parser.add_argument("--topk", type=int, default=3, help="每张图打印 Top-K 概率")
    parser.add_argument("--output", type=str, default=None,
                        help="CSV 输出路径；默认 predictions_<时间戳>.csv")
    parser.add_argument("--max-show", type=int, default=50,
                        help="终端最多显示多少行逐图结果（0 表示全部）")
    parser.add_argument("--log-dir", type=str, default="logs")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace, log) -> int:
    # -- 权重 --
    if args.ckpt:
        ckpt_path = Path(args.ckpt)
    else:
        ckpt_path = find_latest_ckpt(Path(args.ckpt_dir))
    if ckpt_path is None or not ckpt_path.exists():
        log.error(f"未找到 checkpoint：{args.ckpt or args.ckpt_dir}")
        return 1

    # -- 图片 --
    data_root = Path(args.data_dir)
    all_images = discover_images(data_root)
    if not all_images:
        log.error(f"{data_root} 下未找到可识别图片（支持 {sorted(IMAGE_EXTS)}）")
        return 1
    valid, bad = filter_valid(all_images)
    if bad:
        log.warning(f"跳过 {len(bad)} 张无法读取的图片：{', '.join(p.name for p in bad[:5])}"
                    + (" …" if len(bad) > 5 else ""))
    images = valid
    if not images:
        log.error("所有图片均无法读取，退出。")
        return 1

    device = pick_device(args.device)

    log.banner("MNIST 批量推理", f"WideResNet · {len(images)} 张图片")
    log.info(f"详细日志：[accent]{log.log_path}[/]")

    # -- 模型（先加载，才能知道 ckpt 期望的预处理） --
    ckpt = load_model(ckpt_path, device, log)
    model = ckpt.model

    # -- 朝向：默认从 ckpt 推导（图片本身是正常朝向）；--transpose 可强制覆盖 --
    if args.transpose == "auto":
        expects = model_expects_transposed(ckpt.dataset, ckpt.transpose)
        transpose = bool(expects)          # 图片自身是正向；模型要侧躺就转置
        if expects is None:
            transpose_src = "ckpt 未记录，按旧行为不转置"
        else:
            transpose_src = (f"来自 ckpt（dataset={ckpt.dataset}, "
                             f"transpose={ckpt.transpose}）")
    else:
        transpose = (args.transpose == "yes")
        transpose_src = "命令行强制指定"

    n_params = sum(p.numel() for p in model.parameters())
    log.info(
        f"模型就绪 · 参数量 [accent]{fmt_params(n_params)}[/] · "
        f"文件 [accent]{fmt_bytes(ckpt_path.stat().st_size)}[/]"
    )
    log.info(
        f"预处理：transpose={transpose}（{transpose_src}） · "
        f"mean={ckpt.mean:.4f} · std={ckpt.std:.4f}"
    )

    log.kv("运行环境",
           [
               ("设备", f"{device.type}" + (f" · {torch.cuda.get_device_name(0)}"
                                            if device.type == "cuda" else "")),
               ("Checkpoint", str(ckpt_path)),
               ("图片目录", str(data_root)),
               ("图片数量", f"{len(images)}（有效）"),
               ("批大小", str(args.batch_size)),
               ("预处理", f"transpose={transpose}（{transpose_src}） · "
                        f"mean={ckpt.mean:.4f} · std={ckpt.std:.4f}"),
               ("TTA", f"开启 · {args.tta_views} 视图" if args.tta else "关闭"),
               ("反色", "是（白底黑字→黑底白字）" if args.invert else "否"),
           ],
           columns=2, border_style="accent")

    # -- DataLoader --
    transform = build_transform(args.invert, transpose, ckpt.mean, ckpt.std)
    dataset = ImageList(images, transform)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    # -- TTA 配置 --
    tta_config: Config | None = None
    if args.tta:
        tta_config = Config()       # 取默认超参
        tta_config.tta = True
        tta_config.tta_views = max(2, args.tta_views)

    # -- 推理 --
    log.section("批量推理")
    t0 = time.perf_counter()
    probs = run_inference(model, loader, device, tta_config, log, ckpt.mean, ckpt.std)
    elapsed = time.perf_counter() - t0

    preds = probs.argmax(dim=1)
    confs = probs.gather(1, preds.unsqueeze(1)).squeeze(1)

    # -- 结果表 --
    log.section("推理结果")
    topk = max(1, args.topk)
    rows = []
    for path, pred, conf, prob in zip(images, preds.tolist(), confs.tolist(), probs.tolist()):
        order = sorted(range(len(prob)), key=lambda i: prob[i], reverse=True)[:topk]
        top_str = " ".join(f"{i}:{prob[i] * 100:.1f}%" for i in order)
        rows.append([path.name, str(pred), fmt_pct(conf), top_str])

    show_n = len(rows) if args.max_show <= 0 else min(args.max_show, len(rows))
    if show_n < len(rows):
        log.info(f"结果较多（共 {len(rows)} 行），终端只显示前 {show_n} 行；"
                 "完整结果见日志文件与 CSV。")
    avg_conf = float(confs.mean().item()) if len(confs) else 0.0
    log.table(
        "逐图预测",
        ["文件", ("预测", "right"), ("置信度", "right"), f"Top-{topk}"],
        rows[:show_n],
        caption=(f"共 {len(rows)} 张 · 平均置信度 {fmt_pct(avg_conf)} · "
                 f"耗时 {fmt_duration(elapsed)} · "
                 f"{len(rows) / max(1e-9, elapsed):.1f} img/s"),
        zebra=True,
    )

    # -- 预测分布 --
    dist = torch.bincount(preds, minlength=10).tolist()
    log.kv(
        "预测分布",
        [(str(d), f"{dist[d]}（{fmt_pct(dist[d] / max(1, len(preds)))}）") for d in range(10)],
        columns=5, border_style="muted",
    )

    # -- CSV --
    out_path = Path(args.output) if args.output else Path(
        f"predictions_{_dt.datetime.now():%Y%m%d_%H%M%S}.csv"
    )
    write_csv(out_path, images, preds, confs, probs, topk)
    log.success(f"逐图结果已写入 [accent]{out_path}[/]")

    # -- KPI --
    log.metrics([
        ("图片数量", str(len(images))),
        ("平均置信度", fmt_pct(avg_conf)),
        ("推理耗时", fmt_duration(elapsed)),
    ])
    log.success(
        f"批量推理结束 · 共 {len(images)} 张 · "
        f"耗时 {fmt_duration(elapsed)} · 平均 {fmt_pct(avg_conf)} 置信度"
    )
    return 0


def main() -> int:
    args = parse_args()
    log = setup_logging(name="Inference", level="INFO", log_dir=args.log_dir)
    try:
        return run(args, log)
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())