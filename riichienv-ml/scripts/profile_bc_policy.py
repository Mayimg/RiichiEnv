"""Profile offline BC policy trainer step breakdown.

This script intentionally does not save checkpoints or log to wandb.  It builds
the same dataset, encoder, model, optimizer, and scheduler used by
``train_bc.py`` for offline ``behavior_cloning`` configs, then reports where
time is spent inside the training loop.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from riichienv_ml.config import import_class, load_config
from riichienv_ml.trainers.bc_policy import _move_to_device
from riichienv_ml.utils import build_encoder, load_model_weights
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

PHASES = (
    "h2d",
    "zero_grad",
    "forward",
    "mask_loss",
    "backward",
    "grad_clip",
    "optimizer",
    "metrics",
)


@dataclass
class Meter:
    total: float = 0.0
    count: int = 0
    min_value: float = field(default_factory=lambda: float("inf"))
    max_value: float = field(default_factory=lambda: float("-inf"))

    def update(self, value: float) -> None:
        self.total += value
        self.count += 1
        self.min_value = min(self.min_value, value)
        self.max_value = max(self.max_value, value)

    @property
    def avg(self) -> float:
        return self.total / self.count if self.count else 0.0

    @property
    def min(self) -> float:
        return self.min_value if self.count else 0.0

    @property
    def max(self) -> float:
        return self.max_value if self.count else 0.0


@dataclass
class ProfileMeters:
    wall: dict[str, Meter] = field(default_factory=dict)
    cuda: dict[str, Meter] = field(default_factory=dict)
    losses: Meter = field(default_factory=Meter)
    accuracies: Meter = field(default_factory=Meter)
    examples: int = 0

    def __post_init__(self) -> None:
        for name in ("data_wait", "step", "total_iter", "cuda_sum", "host_overhead", *PHASES):
            self.wall[name] = Meter()
        for name in PHASES:
            self.cuda[name] = Meter()

    def update(
        self,
        *,
        batch_size: int,
        data_wait_ms: float,
        step_wall_ms: float,
        phase_wall_ms: dict[str, float],
        phase_cuda_ms: dict[str, float],
        loss: float,
        acc: float,
    ) -> None:
        cuda_sum = sum(phase_cuda_ms.values())
        self.examples += batch_size
        self.wall["data_wait"].update(data_wait_ms)
        self.wall["step"].update(step_wall_ms)
        self.wall["total_iter"].update(data_wait_ms + step_wall_ms)
        self.wall["cuda_sum"].update(cuda_sum)
        self.wall["host_overhead"].update(max(step_wall_ms - cuda_sum, 0.0))
        for name, value in phase_wall_ms.items():
            self.wall[name].update(value)
        for name, value in phase_cuda_ms.items():
            self.cuda[name].update(value)
        self.losses.update(loss)
        self.accuracies.update(acc)


@dataclass
class StepTiming:
    phase_wall_ms: dict[str, float]
    phase_cuda_ms: dict[str, float]
    step_wall_ms: float
    loss: float
    acc: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile BC policy trainer loop")
    parser.add_argument("-c", "--config", required=True, help="Path to BC config YAML")
    parser.add_argument("--data_glob", default=None, help="Override training data glob")
    parser.add_argument("--load_model", default=None, help="Override initial model weights")
    parser.add_argument("--device", default=None, help="Override device, e.g. cuda or cuda:0")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--num_workers", type=int, default=None, help="Override DataLoader workers")
    parser.add_argument("--steps", type=int, default=500, help="Measured training steps")
    parser.add_argument("--warmup", type=int, default=100, help="Warmup training steps excluded from report")
    parser.add_argument("--prefetch_factor", type=int, default=None, help="Override DataLoader prefetch factor")
    parser.add_argument("--persistent_workers", action="store_true", default=None)
    parser.add_argument("--no_persistent_workers", action="store_false", dest="persistent_workers")
    parser.add_argument("--pin_memory", action="store_true", default=None)
    parser.add_argument("--no_pin_memory", action="store_false", dest="pin_memory")
    parser.add_argument("--matmul_precision", choices=("highest", "high", "medium"), default=None)
    parser.add_argument("--compile", choices=("none", "default", "reduce-overhead", "max-autotune"), default="none")
    parser.add_argument("--amp_dtype", choices=("none", "bf16", "fp16"), default="none")
    parser.add_argument("--no_grad_clip", action="store_true", help="Skip grad clipping for comparison")
    parser.add_argument("--dataloader_only", action="store_true", help="Only measure DataLoader batch wait time")
    parser.add_argument("--report_every", type=int, default=100, help="Progress report interval in measured steps")
    parser.add_argument("--json_output", default=None, help="Optional path to write summary JSON")
    parser.add_argument("--torch_profiler_steps", type=int, default=0, help="Run PyTorch profiler for N active steps")
    parser.add_argument("--profiler_dir", default="profiles/bc_policy", help="Profiler trace output directory")
    parser.add_argument("--profiler_record_shapes", action="store_true")
    parser.add_argument("--profiler_profile_memory", action="store_true")
    parser.add_argument("--profiler_with_stack", action="store_true")
    parser.add_argument("--nvtx", action="store_true", help="Emit NVTX ranges for nsys/nsight")
    parser.add_argument("--num_threads", type=int, default=None, help="Set torch CPU thread count")
    return parser.parse_args()


def mask_logits(logits: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    legal = masks.bool() if masks.dtype != torch.bool else masks
    return logits.masked_fill(~legal, torch.finfo(logits.dtype).min)


def create_dataloader(dataset, *, batch_size: int, num_workers: int, args: argparse.Namespace) -> DataLoader:
    pin_memory = args.pin_memory
    if pin_memory is None:
        pin_memory = True

    kwargs: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        persistent_workers = args.persistent_workers
        if persistent_workers is None:
            persistent_workers = True
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = args.prefetch_factor if args.prefetch_factor is not None else 4
    return DataLoader(**kwargs)


def resolve_amp_dtype(name: str) -> torch.dtype | None:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return None


def profile_context(name: str, *, enabled: bool):
    if not enabled or not torch.cuda.is_available():
        return nullcontext()
    return torch.cuda.nvtx.range(name)


def run_step(  # noqa: PLR0915
    *,
    batch,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
    device: torch.device,
    label_smoothing: float,
    max_grad_norm: float,
    amp_dtype: torch.dtype | None,
    no_grad_clip: bool,
    collect_cuda_events: bool,
    nvtx: bool,
) -> StepTiming:
    phase_wall_ms: dict[str, float] = {}
    phase_cuda_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
    use_cuda_events = collect_cuda_events and device.type == "cuda"

    def begin(name: str) -> tuple[float, torch.cuda.Event | None]:
        start_event = None
        if use_cuda_events:
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        return time.perf_counter(), start_event

    def end(name: str, wall_start: float, start_event: torch.cuda.Event | None) -> None:
        phase_wall_ms[name] = (time.perf_counter() - wall_start) * 1000.0
        if use_cuda_events and start_event is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record()
            phase_cuda_events[name] = (start_event, end_event)

    step_start = time.perf_counter()
    features_cpu, actions_cpu, masks_cpu = batch

    with profile_context("h2d", enabled=nvtx):
        wall_start, start_event = begin("h2d")
        features = _move_to_device(features_cpu, device)
        actions = actions_cpu.long().to(device, non_blocking=True)
        masks = masks_cpu.to(device, non_blocking=True)
        end("h2d", wall_start, start_event)

    with profile_context("zero_grad", enabled=nvtx):
        wall_start, start_event = begin("zero_grad")
        optimizer.zero_grad()
        end("zero_grad", wall_start, start_event)

    autocast_context = (
        torch.autocast(device_type=device.type, dtype=amp_dtype)
        if amp_dtype is not None and device.type == "cuda"
        else nullcontext()
    )
    with autocast_context:
        with profile_context("forward", enabled=nvtx):
            wall_start, start_event = begin("forward")
            outputs = model(features)
            logits = outputs[0] if isinstance(outputs, tuple) else outputs
            end("forward", wall_start, start_event)

        with profile_context("mask_loss", enabled=nvtx):
            wall_start, start_event = begin("mask_loss")
            logits = mask_logits(logits, masks)
            loss = F.cross_entropy(logits, actions, label_smoothing=label_smoothing)
            end("mask_loss", wall_start, start_event)

    with profile_context("backward", enabled=nvtx):
        wall_start, start_event = begin("backward")
        loss.backward()
        end("backward", wall_start, start_event)

    with profile_context("grad_clip", enabled=nvtx):
        wall_start, start_event = begin("grad_clip")
        if not no_grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
        end("grad_clip", wall_start, start_event)

    with profile_context("optimizer", enabled=nvtx):
        wall_start, start_event = begin("optimizer")
        optimizer.step()
        scheduler.step()
        end("optimizer", wall_start, start_event)

    with profile_context("metrics", enabled=nvtx):
        wall_start, start_event = begin("metrics")
        predictions = logits.argmax(dim=1)
        acc_tensor = (predictions == actions).float().mean()
        acc = float(acc_tensor.item())
        loss_value = float(loss.item())
        end("metrics", wall_start, start_event)

    phase_cuda_ms: dict[str, float] = {name: 0.0 for name in PHASES}
    if use_cuda_events:
        torch.cuda.synchronize(device)
        for name, (start_event, end_event) in phase_cuda_events.items():
            phase_cuda_ms[name] = float(start_event.elapsed_time(end_event))
    elif device.type == "cuda":
        torch.cuda.synchronize(device)
        phase_cuda_ms.update({name: phase_wall_ms.get(name, 0.0) for name in PHASES})
    else:
        phase_cuda_ms.update({name: phase_wall_ms.get(name, 0.0) for name in PHASES})

    step_wall_ms = (time.perf_counter() - step_start) * 1000.0
    for name in PHASES:
        phase_wall_ms.setdefault(name, 0.0)
        phase_cuda_ms.setdefault(name, 0.0)

    return StepTiming(
        phase_wall_ms=phase_wall_ms,
        phase_cuda_ms=phase_cuda_ms,
        step_wall_ms=step_wall_ms,
        loss=loss_value,
        acc=acc,
    )


def build_profile_objects(args: argparse.Namespace):
    cfg = load_config(args.config).bc
    if cfg.online or cfg.offline_algorithm != "behavior_cloning":
        raise ValueError("profile_bc_policy.py supports offline BC configs with offline_algorithm=behavior_cloning")

    overrides = {}
    for field_name in ("data_glob", "load_model", "device", "batch_size", "num_workers"):
        value = getattr(args, field_name)
        if value is not None:
            overrides[field_name] = value
    if overrides:
        cfg = cfg.model_copy(update=overrides)

    if args.matmul_precision is not None:
        torch.set_float32_matmul_precision(args.matmul_precision)
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)

    device = torch.device(cfg.device)
    game = cfg.game
    encoder_class = import_class(cfg.encoder_class)
    encoder = build_encoder(encoder_class, tile_dim=game.tile_dim, model_config=cfg.model.model_dump())

    train_files = sorted(glob.glob(cfg.data_glob, recursive=True))
    if not train_files:
        raise ValueError(
            f"No data found at data_glob={cfg.data_glob!r}. "
            "Pass --data_glob with the path used on this machine."
        )
    print(f"Found {len(train_files)} training files")

    dataset_class = import_class(cfg.dataset_class)
    train_dataset = dataset_class(
        train_files,
        is_train=True,
        n_players=game.n_players,
        replay_rule=game.replay_rule,
        encoder=encoder,
    )
    dataloader = create_dataloader(
        train_dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        args=args,
    )

    model_class = import_class(cfg.model_class)
    model = model_class(**cfg.model.model_dump()).to(device)
    if cfg.load_model:
        load_model_weights(model, cfg.load_model, map_location=device)
    if args.compile != "none":
        compile_kwargs = {}
        if args.compile != "default":
            compile_kwargs["mode"] = args.compile
        model = torch.compile(model, **compile_kwargs)

    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=max(cfg.limit, 1), eta_min=cfg.lr_min)
    model.train()
    return cfg, device, dataloader, model, optimizer, scheduler, len(train_files)


def profile_dataloader_only(args: argparse.Namespace, dataloader: DataLoader, batch_size: int) -> dict[str, Any]:
    iterator = iter(dataloader)
    meters = ProfileMeters()
    total = args.warmup + args.steps
    for step_idx in range(total):
        start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        data_wait_ms = (time.perf_counter() - start) * 1000.0
        if step_idx >= args.warmup:
            measured_batch = batch[1].shape[0] if hasattr(batch[1], "shape") else batch_size
            meters.examples += int(measured_batch)
            meters.wall["data_wait"].update(data_wait_ms)
            meters.wall["total_iter"].update(data_wait_ms)
        if args.report_every > 0 and step_idx >= args.warmup:
            measured = step_idx - args.warmup + 1
            if measured % args.report_every == 0:
                print(f"measured {measured}/{args.steps} dataloader batches")
    summary = summarize_meters(meters, batch_size=batch_size, device=None, cuda_available=False)
    if summary["measured_steps"] == 0:
        raise RuntimeError(
            "No measured DataLoader batches were collected. "
            "The dataset ended before --warmup completed; reduce --warmup or check --data_glob."
        )
    return summary


def profile_training_loop(
    args: argparse.Namespace,
    *,
    cfg,
    device: torch.device,
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
) -> dict[str, Any]:
    iterator = iter(dataloader)
    meters = ProfileMeters()
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    total = args.warmup + args.steps

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step_idx in range(total):
        if step_idx == args.warmup and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        data_start = time.perf_counter()
        try:
            batch = next(iterator)
        except StopIteration:
            break
        data_wait_ms = (time.perf_counter() - data_start) * 1000.0

        timing = run_step(
            batch=batch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            label_smoothing=cfg.label_smoothing,
            max_grad_norm=cfg.max_grad_norm,
            amp_dtype=amp_dtype,
            no_grad_clip=args.no_grad_clip,
            collect_cuda_events=True,
            nvtx=args.nvtx,
        )

        if step_idx >= args.warmup:
            batch_size = int(batch[1].shape[0])
            meters.update(
                batch_size=batch_size,
                data_wait_ms=data_wait_ms,
                step_wall_ms=timing.step_wall_ms,
                phase_wall_ms=timing.phase_wall_ms,
                phase_cuda_ms=timing.phase_cuda_ms,
                loss=timing.loss,
                acc=timing.acc,
            )

            if args.report_every > 0:
                measured = step_idx - args.warmup + 1
                if measured % args.report_every == 0:
                    avg_ms = meters.wall["total_iter"].avg
                    print(
                        f"measured {measured}/{args.steps} batches: "
                        f"{avg_ms:.3f} ms/batch, loss={meters.losses.avg:.4f}, acc={meters.accuracies.avg:.4f}"
                    )

    summary = summarize_meters(
        meters,
        batch_size=cfg.batch_size,
        device=device,
        cuda_available=device.type == "cuda" and torch.cuda.is_available(),
    )
    if summary["measured_steps"] == 0:
        raise RuntimeError(
            "No measured training batches were collected. "
            "The dataset ended before --warmup completed; reduce --warmup or check --data_glob."
        )
    return summary


def run_torch_profiler(
    args: argparse.Namespace,
    *,
    cfg,
    device: torch.device,
    dataloader: DataLoader,
    model: torch.nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler.LRScheduler,
) -> None:
    if args.torch_profiler_steps <= 0:
        return

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    profiler_dir = Path(args.profiler_dir)
    profiler_dir.mkdir(parents=True, exist_ok=True)
    iterator = iter(dataloader)
    amp_dtype = resolve_amp_dtype(args.amp_dtype)
    total_steps = 2 + args.torch_profiler_steps

    print(f"\nRunning torch.profiler for {args.torch_profiler_steps} active steps...")
    with torch.profiler.profile(
        activities=activities,
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=args.torch_profiler_steps, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profiler_dir)),
        record_shapes=args.profiler_record_shapes,
        profile_memory=args.profiler_profile_memory,
        with_stack=args.profiler_with_stack,
    ) as profiler:
        for _ in range(total_steps):
            try:
                batch = next(iterator)
            except StopIteration:
                break
            run_step(
                batch=batch,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=device,
                label_smoothing=cfg.label_smoothing,
                max_grad_norm=cfg.max_grad_norm,
                amp_dtype=amp_dtype,
                no_grad_clip=args.no_grad_clip,
                collect_cuda_events=False,
                nvtx=args.nvtx,
            )
            profiler.step()

    sort_key = "cuda_time_total" if device.type == "cuda" else "cpu_time_total"
    print("\nTop profiler operators:")
    print(profiler.key_averages().table(sort_by=sort_key, row_limit=30))
    print(f"Profiler traces written to: {profiler_dir}")


def summarize_meters(
    meters: ProfileMeters,
    *,
    batch_size: int,
    device: torch.device | None,
    cuda_available: bool,
) -> dict[str, Any]:
    total_ms = meters.wall["total_iter"].avg
    step_ms = meters.wall["step"].avg
    measured_steps = meters.wall["total_iter"].count
    total_wall = meters.wall["total_iter"].total
    samples_per_sec = 1000.0 * meters.examples / total_wall if total_wall else 0.0
    batches_per_sec = 1000.0 * measured_steps / total_wall if total_wall else 0.0

    summary: dict[str, Any] = {
        "measured_steps": measured_steps,
        "measured_examples": meters.examples,
        "nominal_batch_size": batch_size,
        "avg_loss": meters.losses.avg,
        "avg_acc": meters.accuracies.avg,
        "batches_per_sec": batches_per_sec,
        "samples_per_sec": samples_per_sec,
        "wall_ms": {
            name: {
                "avg": meter.avg,
                "min": meter.min,
                "max": meter.max,
                "pct_total": (meter.avg / total_ms * 100.0) if total_ms else 0.0,
            }
            for name, meter in meters.wall.items()
        },
        "cuda_ms": {
            name: {
                "avg": meter.avg,
                "min": meter.min,
                "max": meter.max,
                "pct_total": (meter.avg / total_ms * 100.0) if total_ms else 0.0,
                "pct_step": (meter.avg / step_ms * 100.0) if step_ms else 0.0,
            }
            for name, meter in meters.cuda.items()
        },
    }

    if cuda_available and device is not None:
        summary["cuda_memory_mb"] = {
            "max_allocated": torch.cuda.max_memory_allocated(device) / 1024 / 1024,
            "max_reserved": torch.cuda.max_memory_reserved(device) / 1024 / 1024,
        }
        summary["cuda_device"] = torch.cuda.get_device_name(device)
    return summary


def print_summary(
    summary: dict[str, Any],
    *,
    args: argparse.Namespace,
    cfg,
    device: torch.device,
    train_file_count: int,
) -> None:
    print("\n=== BC Policy Trainer Profile ===")
    print(f"config: {args.config}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"cuda_device: {summary.get('cuda_device', torch.cuda.get_device_name(device))}")
    print(f"model_class: {cfg.model_class}")
    print(f"dataset_class: {cfg.dataset_class}")
    print(f"encoder_class: {cfg.encoder_class}")
    print(f"data_glob: {cfg.data_glob}")
    print(f"train_files: {train_file_count}")
    print(f"batch_size: {cfg.batch_size}")
    print(f"num_workers: {cfg.num_workers}")
    print(f"warmup_steps: {args.warmup}")
    print(f"measured_steps: {summary['measured_steps']}")
    print(f"compile: {args.compile}")
    print(f"amp_dtype: {args.amp_dtype}")
    print(f"matmul_precision: {torch.get_float32_matmul_precision()}")
    print(f"avg_loss: {summary['avg_loss']:.6f}")
    print(f"avg_acc: {summary['avg_acc']:.6f}")
    print(f"batches/sec: {summary['batches_per_sec']:.3f}")
    print(f"samples/sec: {summary['samples_per_sec']:.1f}")

    print("\nWall-time breakdown per batch:")
    print(f"{'phase':<20} {'avg_ms':>10} {'pct_total':>10} {'min_ms':>10} {'max_ms':>10}")
    for name in ("data_wait", "step", "cuda_sum", "host_overhead", "total_iter"):
        item = summary["wall_ms"][name]
        print(f"{name:<20} {item['avg']:>10.3f} {item['pct_total']:>9.1f}% {item['min']:>10.3f} {item['max']:>10.3f}")

    print("\nCUDA/event phase breakdown per batch:")
    print(f"{'phase':<20} {'cuda_ms':>10} {'pct_total':>10} {'pct_step':>10} {'wall_ms':>10}")
    for name in PHASES:
        cuda_item = summary["cuda_ms"][name]
        wall_item = summary["wall_ms"][name]
        print(
            f"{name:<20} {cuda_item['avg']:>10.3f} {cuda_item['pct_total']:>9.1f}% "
            f"{cuda_item['pct_step']:>9.1f}% {wall_item['avg']:>10.3f}"
        )

    if "cuda_memory_mb" in summary:
        mem = summary["cuda_memory_mb"]
        print("\nCUDA memory:")
        print(f"max_allocated_mb: {mem['max_allocated']:.1f}")
        print(f"max_reserved_mb: {mem['max_reserved']:.1f}")


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.warmup < 0:
        raise ValueError("--warmup must be >= 0")

    cfg, device, dataloader, model, optimizer, scheduler, train_file_count = build_profile_objects(args)
    if args.dataloader_only:
        summary = profile_dataloader_only(args, dataloader, cfg.batch_size)
    else:
        summary = profile_training_loop(
            args,
            cfg=cfg,
            device=device,
            dataloader=dataloader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        run_torch_profiler(
            args,
            cfg=cfg,
            device=device,
            dataloader=dataloader,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    summary["data_glob"] = cfg.data_glob
    summary["train_file_count"] = train_file_count
    print_summary(summary, args=args, cfg=cfg, device=device, train_file_count=train_file_count)

    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
        print(f"\nJSON summary written to: {output_path}")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn", force=True)
    main()
