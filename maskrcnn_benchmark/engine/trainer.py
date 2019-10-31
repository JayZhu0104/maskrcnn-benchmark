# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import datetime
import logging
import os
import time

import torch
import torch.distributed as dist
from tqdm import tqdm

from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.utils.comm import get_world_size, synchronize
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.engine.inference import inference

from apex import amp

def reduce_loss_dict(loss_dict):
    """
    Reduce the loss dictionary from all processes so that process with rank
    0 has the averaged results. Returns a dict with the same fields as
    loss_dict, after reduction.
    """
    world_size = get_world_size()
    if world_size < 2:
        return loss_dict
    with torch.no_grad():
        loss_names = []
        all_losses = []
        for k in sorted(loss_dict.keys()):
            loss_names.append(k)
            all_losses.append(loss_dict[k])
        all_losses = torch.stack(all_losses, dim=0)
        dist.reduce(all_losses, dst=0)
        if dist.get_rank() == 0:
            # only main process gets accumulated, so only divide by
            # world_size in this case
            all_losses /= world_size
        reduced_losses = {k: v for k, v in zip(loss_names, all_losses)}
    return reduced_losses


def do_train(
    cfg,
    model,
    data_loader,
    data_loader_val,
    optimizer,
    scheduler,
    checkpointer,
    device,
    checkpoint_period,
    test_period,
    arguments,
):
    logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(data_loader)
    start_iter = arguments["iteration"]
    model.train()
    start_training_time = time.time()
    end = time.time()

    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    dataset_names = cfg.DATASETS.TEST

    for iteration, (images, targets, _) in enumerate(data_loader, start_iter):
        
        if any(len(target) < 1 for target in targets):
            logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            continue
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        images = images.to(device)
        targets = [target.to(device) for target in targets]
        # 计算损失，model会计算images与targets的损失，并返回
        # 实际调用的是GeneralizedRCNN类的forward方法
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        # 多GPU的时候，会有无效损失，去除这些损失
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        meters.update(loss=losses_reduced, **loss_dict_reduced)

        optimizer.zero_grad()
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # amp的用法
        with amp.scale_loss(losses, optimizer) as scaled_losses:
            scaled_losses.backward()
        optimizer.step()
        # scheduler也需要step
        scheduler.step()
        # 每个batch要多久
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        # 还需要多久
        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

        # 每20次迭代或者最好一次迭代了，输出信息
        if iteration % 20 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string, # 还需要多久
                    iter=iteration, # 第几次迭代，是从maskrcnn_benchmark.utils.checkpoint记录的迭代次数开始算起
                    meters=str(meters), # 包含loss、loss_classifier、loss_box_reg、loss_objectness、loss_rpn_box_reg
                    lr=optimizer.param_groups[0]["lr"], # 学习率
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0, # 占用最大的GPU内存
                )
            )
        # 每隔checkpoint_period保存一次checkpoint
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if data_loader_val is not None and test_period > 0 and iteration % test_period == 0:
            meters_val = MetricLogger(delimiter="  ")
            synchronize()
            _ = inference(  # The result can be used for additional logging, e. g. for TensorBoard
                model,
                # The method changes the segmentation mask format in a data loader,
                # so every time a new data loader is created:
                make_data_loader(cfg, is_train=False, is_distributed=(get_world_size() > 1), is_for_period=True),
                dataset_name="[Validation]",
                iou_types=iou_types,
                box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
                device=cfg.MODEL.DEVICE,
                expected_results=cfg.TEST.EXPECTED_RESULTS,
                expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
                output_folder=None,
            )
            synchronize()
            model.train()
            with torch.no_grad():
                # Should be one image for each GPU:
                for iteration_val, (images_val, targets_val, _) in enumerate(tqdm(data_loader_val)):
                    images_val = images_val.to(device)
                    targets_val = [target.to(device) for target in targets_val]
                    loss_dict = model(images_val, targets_val)
                    losses = sum(loss for loss in loss_dict.values())
                    loss_dict_reduced = reduce_loss_dict(loss_dict)
                    losses_reduced = sum(loss for loss in loss_dict_reduced.values())
                    meters_val.update(loss=losses_reduced, **loss_dict_reduced)
            synchronize()
            logger.info(
                meters_val.delimiter.join(
                    [
                        "[Validation]: ",
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                        "lr: {lr:.6f}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    eta=eta_string,
                    iter=iteration,
                    meters=str(meters_val),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
        # 最后一次迭代的时候保存checkpoint
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)
    # train总共花了多少时间
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
