import os
from itertools import combinations

import numpy as np
import argparse
import errno
import math
import pickle
import tensorboardX
from tqdm import tqdm
from time import time
import copy
import random
import prettytable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from lib.utils.tools import *
from lib.utils.learning import *
from lib.utils.utils_data import flip_data
from lib.data.dataset_motion_2d import PoseTrackDataset2D, InstaVDataset2D
from lib.data.dataset_motion_3d import MotionDataset3D, SkiPoseDataset3D
from lib.data.augmentation import Augmenter2D
from lib.data.datareader_h36m import DataReaderH36M
from lib.model.loss import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/pretrain.yaml", help="Path to the config file.")
    parser.add_argument('-c', '--checkpoint', default='checkpoint', type=str, metavar='PATH', help='checkpoint directory')
    parser.add_argument('-p', '--pretrained', default='checkpoint', type=str, metavar='PATH', help='pretrained checkpoint directory')
    parser.add_argument('-r', '--resume', default='', type=str, metavar='FILENAME', help='checkpoint to resume (file name)')
    parser.add_argument('-e', '--evaluate', default='', type=str, metavar='FILENAME', help='checkpoint to evaluate (file name)')
    parser.add_argument('-ms', '--selection', default='latest_epoch.bin', type=str, metavar='FILENAME', help='checkpoint to finetune (file name)')
    parser.add_argument('-sd', '--seed', default=0, type=int, help='random seed')
    opts = parser.parse_args()
    return opts

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss):
    print('Saving checkpoint to', chk_path)
    torch.save({
        'epoch': epoch + 1,
        'lr': lr,
        'optimizer': optimizer.state_dict(),
        'model_pos': model_pos.state_dict(),
        'min_loss' : min_loss
    }, chk_path)


def denormalise(pose: Tensor, resolution: Tensor) -> Tensor:
    pose, height, width = pose.clone().T, *resolution.T
    pose[0] += 1
    pose[1] += height / width
    pose *= width / 2
    return pose.T

def evaluate(args, model_pos, test_loader, datareader):
    print('INFO: Testing')
    results_all = []
    gts = []
    activities = []
    model_pos.eval()
    actions = np.array(['Direction', 'Discuss', 'Eating', 'Greet', 'Phone', 'Pose', 'Purchase', 'Sitting', 'SittingDown', 'Smoke', 'Photo',
     'Wait', 'Walk', 'WalkDog', 'WalkTwo'])
    with torch.no_grad():
        for pose_2d, _, performer, resolution, factor, gt, action_index in tqdm(test_loader):
            as_batched = (-1, *pose_2d.shape[2:])
            if torch.cuda.is_available():
                pose_2d = pose_2d.cuda()
                resolution = resolution.cuda()
            if args.no_conf:
                pose_2d = pose_2d[..., :2]

            predicted_pose = model_pos(pose_2d.view(as_batched)).view(pose_2d.shape)
            if args.flip:
                predicted_pose_flipped = model_pos(flip_data(pose_2d).view(as_batched)).view(pose_2d.shape)
                predicted_pose = (predicted_pose + flip_data(predicted_pose_flipped)) / 2

            if args.rootrel:
                predicted_pose[..., 0, :] = 0     # [N,T,17,3]

            if args.gt_2d:
                predicted_pose[..., :2] = pose_2d[..., :2]

            predicted_pose = denormalise(predicted_pose, resolution).T
            predicted_pose *= factor.T.to(predicted_pose)
            results_all.append(predicted_pose.T.cpu().numpy())
            gts.append(gt.cpu().numpy())
            activities.append(actions[action_index])
    results_all = np.concatenate(results_all)
    gts = np.concatenate(gts)

    predictions = results_all.copy()
    predictions -= predictions[..., 0:1, :]
    gts -= gts[..., 0:1, :]
    mpjpe_total = mpjpe(predictions, gts)
    p_mpjpe_total = p_mpjpe(predictions, gts)

    print('Protocol #1 Error (MPJPE):', mpjpe_total, 'mm')
    print('Protocol #2 Error (P-MPJPE):', p_mpjpe_total, 'mm')
    print('----------')
    summary_table = prettytable.PrettyTable()
    try:
        activities = np.concatenate(activities)
        mpjpe_action = {action: mpjpe(predictions[activities == action], gts[activities == action]) for action in actions}
        p_mpjpe_action = {action: p_mpjpe(predictions[activities == action], gts[activities == action]) for action in actions}
        summary_table.field_names = ['test_name'] + actions.tolist()
        summary_table.add_row(['P1'] + [mpjpe_action[action] for action in actions])
        summary_table.add_row(['P2'] + [p_mpjpe_action[action] for action in actions])
        print(summary_table)
    except:
        pass
    return mpjpe_total, p_mpjpe_total, results_all

        
def train_epoch(
        args,
        model_pos: torch.nn.Module,
        train_loader: DataLoader,
        losses: dict[str, AverageMeter],
        optimizer: torch.optim.Optimizer,
        has_3d: bool,
        has_gt: bool,
        no_confidence: bool,
        relative_to_root: bool,
        accumulate_gradients: int = 1,
):
    assert has_3d == True, "We do not support no 3D data yet."

    model_pos.train()
    for idx, batch in enumerate(tqdm(train_loader)):
        if len(batch) == 3:
            (batch_input, target, performer_id) = batch
            frame_exist = torch.ones_like(target[..., 0, 0], dtype=torch.float)
        else:
            (batch_input, target, performer_id, frame_exist, pred_2d) = batch
        batch_size = len(batch_input)
        if torch.cuda.is_available():
            batch_input = batch_input.cuda()
            target = target.cuda()
            frame_exist = frame_exist.cuda()
        with torch.no_grad():
            if no_confidence:
                batch_input = batch_input[..., :2]
            if relative_to_root:
                target = target - target[..., 0:1, :]
            else:
                target[..., 2] = target[..., 2] - target[..., 0:1, 0:1, 2] # Place the depth of first frame root to 0.
            if args.mask or args.noise:
                batch_input = args.aug.augment2D(batch_input, noise=(args.noise and has_gt), mask=args.mask)
        # Predict 3D poses
        as_batched = (-1, *batch_input.shape[2:])
        predicted_pose = model_pos(batch_input.view(as_batched)).view(batch_input.shape)    # (B*, T, 17, 3)

        special_performer = performer_id == 1

        function_weight_label = (
            (loss_mpjpe, args.mpjpe_weight, "3d_pos"), (n_mpjpe, args.lambda_scale, "3d_scale"),
            (loss_velocity, args.lambda_3d_velocity, "3d_velocity"), (lambda x, t: loss_limb_var(x), args.lambda_lv, "lv"),
            (loss_limb_gt, args.lambda_lg, "lg"), (loss_angle, args.lambda_a, "angle"),
            (loss_angle_velocity, args.lambda_av, "angle_velocity")
        )
        loss_total = torch.zeros(1, device=predicted_pose.device)
        if special_performer.any():
            for loss, weight, name in function_weight_label:
                value = loss(predicted_pose[special_performer], target[special_performer], frame_exist[special_performer])
                losses[name].update(value.item(), batch_size)
                loss_total += weight * value

        if (~special_performer).any():
            loss_2d_proj = loss_2d_weighted(
                predicted_pose[~special_performer],
                target[~special_performer],
                batch_input[~special_performer, ..., -1]
            )
            loss_total += loss_2d_proj * args.mpjpe_2d_weight
            losses['2d_proj'].update(loss_2d_proj.item(), batch_size)

        if predicted_pose.size(1) > 1:
            view_pairs = (*zip(*combinations(range(predicted_pose.size(1)), 2)),)
            consistency = loss_consistency(*predicted_pose[:, view_pairs, ...].unbind(1), frame_exist[:, view_pairs].unbind(1))
            losses["consistency"].update(consistency.item(), batch_size)
            loss_total += args.lambda_consistency * consistency

        losses['total'].update(loss_total.item(), batch_size)
        loss_total.backward()
        if (idx + 1) % accumulate_gradients == 0 or idx == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)


def train_with_config(args, opts):
    print(args)
    try:
        os.makedirs(opts.checkpoint, exist_ok=True)
    except OSError as e:
        raise RuntimeError('Unable to create checkpoint directory:', opts.checkpoint)
    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.checkpoint, "logs"))


    print('Loading dataset...')
    trainloader_params = {
          'batch_size': args.batch_size,
          'shuffle': True,
          'num_workers': args.num_workers,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }
    
    testloader_params = {
          'batch_size': args.batch_size,
          'shuffle': False,
          'num_workers': args.num_workers,
          'pin_memory': True,
          'prefetch_factor': 4,
          'persistent_workers': True
    }

    if args.get("dataset") == "skipose":
        train_dataset = SkiPoseDataset3D(args, args.subset_list, 'train', train_views=args.train_views)
        test_dataset = SkiPoseDataset3D(args, args.subset_list, 'test', train_views=args.train_views)
    else:
        train_dataset = MotionDataset3D(args, args.subset_list, 'train')
        test_dataset = MotionDataset3D(args, args.subset_list, 'test')
    train_loader_3d = DataLoader(train_dataset, **trainloader_params)
    test_loader = DataLoader(test_dataset, **testloader_params)
    
    if args.train_2d:
        posetrack = PoseTrackDataset2D()
        posetrack_loader_2d = DataLoader(posetrack, **trainloader_params)
        instav = InstaVDataset2D()
        instav_loader_2d = DataLoader(instav, **trainloader_params)
        
    datareader = DataReaderH36M(n_frames=args.clip_len, sample_stride=args.sample_stride, data_stride_train=args.data_stride, data_stride_test=args.clip_len, dt_root = 'data/motion3d', dt_file=args.dt_file)
    min_loss = 100000
    model_backbone = load_backbone(args)
    model_params = 0
    for parameter in model_backbone.parameters():
        model_params = model_params + parameter.numel()
    print('INFO: Trainable parameter count:', model_params)

    if torch.cuda.is_available():
        model_backbone = nn.DataParallel(model_backbone)
        model_backbone = model_backbone.cuda()

    if args.finetune:
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone
        else:
            chk_filename = os.path.join(opts.pretrained, opts.selection)
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
            model_pos = model_backbone            
    else:
        chk_filename = os.path.join(opts.checkpoint, "latest_epoch.bin")
        if os.path.exists(chk_filename):
            opts.resume = chk_filename
        if opts.resume or opts.evaluate:
            chk_filename = opts.evaluate if opts.evaluate else opts.resume
            print('Loading checkpoint', chk_filename)
            checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
            model_backbone.load_state_dict(checkpoint['model_pos'], strict=True)
        model_pos = model_backbone
        
    if args.partial_train:
        model_pos = partial_train_layers(model_pos, args.partial_train)

    if not opts.evaluate:        
        lr = args.learning_rate
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model_pos.parameters()), lr=lr, weight_decay=args.weight_decay)
        lr_decay = args.lr_decay
        st = 0
        if args.train_2d:
            print('INFO: Training on {}(3D)+{}(2D) batches'.format(len(train_loader_3d), len(instav_loader_2d) + len(posetrack_loader_2d)))
        else:
            print('INFO: Training on {}(3D) batches'.format(len(train_loader_3d)))
        if opts.resume:
            st = checkpoint['epoch']
            if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
                optimizer.load_state_dict(checkpoint['optimizer'])
            else:
                print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')            
            lr = checkpoint['lr']
            if 'min_loss' in checkpoint and checkpoint['min_loss'] is not None:
                min_loss = checkpoint['min_loss']
                
        args.mask = (args.mask_ratio > 0 and args.mask_T_ratio > 0)
        if args.mask or args.noise:
            args.aug = Augmenter2D(args)

       # evaluate(args, model_pos, test_loader, datareader)
        # Training
        print("wtf RANY why not training")
        for epoch in range(st, args.epochs):
            print('Training epoch %d.' % epoch)
            start_time = time()
            losses = {}
            losses['3d_pos'] = AverageMeter()
            losses['3d_scale'] = AverageMeter()
            losses['2d_proj'] = AverageMeter()
            losses['lg'] = AverageMeter()
            losses['lv'] = AverageMeter()
            losses['total'] = AverageMeter()
            losses['3d_velocity'] = AverageMeter()
            losses['angle'] = AverageMeter()
            losses['angle_velocity'] = AverageMeter()
            losses["consistency"] = AverageMeter()
            N = 0

            # Curriculum Learning
            optimizer.zero_grad()
            train = partial(
                train_epoch,
                args=args, model_pos=model_pos, losses=losses, optimizer=optimizer,
                no_confidence=args.no_conf, relative_to_root=args.rootrel,
                accumulate_gradients=args.accumulate_gradients,
            )
            if args.train_2d and (epoch >= args.pretrain_3d_curriculum):
                train(train_loader=posetrack_loader_2d, has_3d=False, has_gt=True)
                train(train_loader=instav_loader_2d, has_3d=False, has_gt=False)
            train(train_loader=train_loader_3d, has_3d=True, has_gt=True)
            elapsed = (time() - start_time) / 60

            if args.no_eval:
                print('[%d] time %.2f lr %f 3d_train %f' % (
                    epoch + 1,
                    elapsed,
                    lr,
                   losses['3d_pos'].avg))
                continue

            e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)
            print('[%d] time %.2f lr %f 3d_train %f e1 %f e2 %f' % (
                epoch + 1,
                elapsed,
                lr,
                losses['3d_pos'].avg,
                e1, e2))
            train_writer.add_scalar('Error P1', e1, epoch + 1)
            train_writer.add_scalar('Error P2', e2, epoch + 1)
            train_writer.add_scalar('loss_3d_pos', losses['3d_pos'].avg, epoch + 1)
            train_writer.add_scalar('loss_2d_proj', losses['2d_proj'].avg, epoch + 1)
            train_writer.add_scalar('loss_3d_scale', losses['3d_scale'].avg, epoch + 1)
            train_writer.add_scalar('loss_3d_velocity', losses['3d_velocity'].avg, epoch + 1)
            train_writer.add_scalar('loss_lv', losses['lv'].avg, epoch + 1)
            train_writer.add_scalar('loss_lg', losses['lg'].avg, epoch + 1)
            train_writer.add_scalar('loss_a', losses['angle'].avg, epoch + 1)
            train_writer.add_scalar('loss_av', losses['angle_velocity'].avg, epoch + 1)
            train_writer.add_scalar('loss_total', losses['total'].avg, epoch + 1)

            # Decay learning rate exponentially
            lr *= lr_decay
            for param_group in optimizer.param_groups:
                param_group['lr'] *= lr_decay

            # Save checkpoints
            chk_path = os.path.join(opts.checkpoint, 'epoch_{}.bin'.format(epoch))
            chk_path_latest = os.path.join(opts.checkpoint, 'latest_epoch.bin')
            chk_path_best = os.path.join(opts.checkpoint, 'best_epoch.bin'.format(epoch))

            save_checkpoint(chk_path_latest, epoch, lr, optimizer, model_pos, min_loss)
            if (epoch + 1) % args.checkpoint_frequency == 0:
                save_checkpoint(chk_path, epoch, lr, optimizer, model_pos, min_loss)
            if e1 < min_loss:
                min_loss = e1
                save_checkpoint(chk_path_best, epoch, lr, optimizer, model_pos, min_loss)
                
    if opts.evaluate:
        e1, e2, results_all = evaluate(args, model_pos, test_loader, datareader)

if __name__ == "__main__":
    opts = parse_args()
    set_random_seed(opts.seed)
    args = get_config(opts.config)
    train_with_config(args, opts)
