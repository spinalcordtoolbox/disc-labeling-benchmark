#===================================================
## code from https://github.com/bearpaw/pytorch-pose 
# Revised by Reza Azad (rezazad68@gmail.com)
# Revised by Nathan Molinier (nathan.molinier@gmail.com)
#===================================================

from __future__ import print_function, absolute_import
import os
import argparse
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
from progress.bar import Bar
import pickle
from torch.utils.data import DataLoader 
import copy
import wandb

from dlh.models.hourglass import hg
from dlh.models.atthourglass import atthg
from dlh.models import JointsMSELoss
from dlh.models.utils import AverageMeter, adjust_learning_rate, accuracy, dice_loss
from dlh.utils.train_utils import image_Dataset, SaveOutput, save_epoch_res_as_image2, save_attention
from dlh.utils.skeleton import create_skeleton

# select proper device to run
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  

def main(args):
    '''
    Training hourglass network
    '''
    best_acc = 0
    weight_folder = args.weight_folder
    vis_folder = args.visual_folder
    
    ## Load the prepared dataset
    with open(args.datapath, 'rb') as file_pi:
        full = pickle.load(file_pi)
    
    # Split training dataset into training and validation
    train_idx = int(np.round(len(full[0]) *0.9))
    validation_idx = int(np.round(len(full[0])))
    full[0] = full[0][:, :, :, :, 0]
    
    ## Create a dataset loader
    full_dataset_train = image_Dataset(image_paths=full[0][0:train_idx], 
                                       target_paths=full[1][:train_idx], 
                                       num_channel=args.ndiscs
                                       )
    full_dataset_val = image_Dataset(image_paths=full[0][train_idx:validation_idx], 
                                     target_paths=full[1][train_idx:validation_idx],
                                     num_channel=args.ndiscs, 
                                     use_flip = False
                                     )

    MRI_train_loader = DataLoader(full_dataset_train, batch_size= args.train_batch,
                                shuffle=True,
                                num_workers=0)
    MRI_val_loader = DataLoader(full_dataset_val, batch_size=args.val_batch,
                                shuffle=False,
                                num_workers=0)
    
    # idx is the index of joints used to compute accuracy (we detect N discs starting from C1 to args.ndiscs) 
    idx = [(i+1) for i in range(args.ndiscs)]

    # create model
    print("==> creating model stacked hourglass, stacks={}, blocks={}".format(args.stacks, args.blocks))
    if args.att:
        model = atthg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.ndiscs)
    else:
        model = hg(num_stacks=args.stacks, num_blocks=args.blocks, num_classes=args.ndiscs)
    model = torch.nn.DataParallel(model).to(device)

    # define loss function (criterion) and optimizer
    criterion = JointsMSELoss().to(device)

    if args.solver == 'rms':
        optimizer = torch.optim.RMSprop(
                                        model.parameters(),
                                        lr=args.lr,
                                        momentum=args.momentum,
                                        weight_decay=args.weight_decay
                                        )
    elif args.solver == 'adam':
        optimizer = torch.optim.Adam(
                                    model.parameters(),
                                    lr=args.lr,
        )
    else:
        print('Unknown solver: {}'.format(args.solver))
        assert False
    
    # optionally resume from a checkpoint
    if args.resume:
       print("=> loading checkpoint to continue learing process")
       if args.att:
            model.load_state_dict(torch.load(f'{weight_folder}/model_{args.contrast}_att_stacks_{args.stacks}_ndiscs_{args.ndiscs}', map_location='cpu')['model_weights'])
       else:
            model.load_state_dict(torch.load(f'{weight_folder}/model_{args.contrast}_stacks_{args.stacks}_ndiscs_{args.ndiscs}', map_location='cpu')['model_weights'])

    # evaluation only
    if args.evaluate:
        print('\nEvaluation only')
        print('loading the pretrained weight')
        if args.att:
            model.load_state_dict(torch.load(f'{weight_folder}/model_{args.contrast}_att_stacks_{args.stacks}_ndiscs_{args.ndiscs}', map_location='cpu')['model_weights'])
        else:
            model.load_state_dict(torch.load(f'{weight_folder}/model_{args.contrast}_stacks_{args.stacks}_ndiscs_{args.ndiscs}', map_location='cpu')['model_weights'])

        if args.attshow:
            loss, acc = show_attention(MRI_val_loader, model)
        else:
            loss, acc = validate(MRI_val_loader, model, criterion, epoch, idx, vis_folder)
        return
    
    # train and eval
    lr = args.lr
    for epoch in range(args.start_epoch, args.epochs):
        lr = adjust_learning_rate(optimizer, epoch, lr, args.schedule, args.gamma)
        print('\nEpoch: %d | LR: %.8f' % (epoch + 1, lr))

        # decay sigma
        if args.sigma_decay > 0:
            MRI_train_loader.dataset.sigma *=  args.sigma_decay
            MRI_val_loader.dataset.sigma *=  args.sigma_decay

        # train for one epoch
        train_loss, train_acc = train(MRI_train_loader, model, criterion, optimizer, idx)

        # evaluate on validation set
        valid_loss, valid_acc  = validate(MRI_val_loader, model, criterion, epoch, idx, vis_folder)

        # remember best acc and save checkpoint
        if valid_acc > best_acc:
           state = copy.deepcopy({'model_weights': model.state_dict()})
           if args.att:
                torch.save(state, f'{weight_folder}/model_{args.contrast}_att_stacks_{args.stacks}_ndiscs_{args.ndiscs}')
           else:
                torch.save(state, f'{weight_folder}/model_{args.contrast}_stacks_{args.stacks}_ndiscs_{args.ndiscs}')
           best_acc = valid_acc
                

def train(train_loader, model, criterion, optimizer, idx):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    loss_dices = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()

    
    bar = Bar('Train', max=len(train_loader))
    
    for i, (input, target, vis) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        input, target = input.to(device), target.to(device, non_blocking=True)
        vis = vis.to(device, non_blocking=True)
        # compute output
        output = model(input) 
        if type(output) == list:  # multiple output
            loss = 0
            for o in output:
                loss += criterion(o, target, vis)
            output = output[-1]
        else:  # single output
            loss = criterion(output, target, vis)
        acc = accuracy(output, target, idx)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))
        acces.update(acc[0], input.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # plot progress
        bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
                    batch=i + 1,
                    size=len(train_loader),
                    data=data_time.val,
                    bt=batch_time.val,
                    total=bar.elapsed_td,
                    eta=bar.eta_td,
                    loss=losses.avg*100,
                    acc=acces.avg
                    )
        bar.next()

    return losses.avg, acces.avg


def validate(val_loader, model, criterion, ep, idx, out_folder):
    Flag_visualize = True
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    loss_dices = AverageMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()
    bar = Bar('Eval ', max=len(val_loader))
    with torch.no_grad():
        for i, (input, target, vis) in enumerate(val_loader):
            # measure data loading time
            data_time.update(time.time() - end)

            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            vis = vis.to(device, non_blocking=True)
            # compute output
            output = model(input)
            output = output[-1]
        
            if type(output) == list:  # multiple output
                loss = 0
                for o in output:
                    loss += criterion(o, target, vis)
                output = output[-1]
            else:  # single output
                loss = criterion(output, target, vis)

            acc = accuracy(output.cpu(), target.cpu(), idx)
            loss_dice = dice_loss(output, target)

            if Flag_visualize:
                # save the visualization only for the first batch of the validation
                save_epoch_res_as_image2(input, output, target, out_folder, epoch_num=ep, target_th=0.5)
                Flag_visualize = False

            # measure accuracy and record loss
            losses.update(loss.item(), input.size(0))
            acces.update(acc[0], input.size(0))
            loss_dices.update(loss_dice.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # plot progress
            bar.suffix  = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}| dice: {dice:.4f}'.format(
                        batch=i + 1,
                        size=len(val_loader),
                        data=data_time.val,
                        bt=batch_time.avg,
                        total=bar.elapsed_td,
                        eta=bar.eta_td,
                        loss=losses.avg*100,
                        acc=acces.avg,
                        dice=loss_dices.avg*100
                        )
            bar.next()

        bar.finish()
    return losses.avg, acces.avg



def show_attention(val_loader, model):
    ## define the attention layer output
    save_output = SaveOutput()
    for layer in model.modules():
        if isinstance(layer, torch.nn.modules.conv.Conv2d):
            if layer.weight.size()[0]==1:     
                layer.register_forward_hook(save_output)
                break
    # switch to evaluate mode
    N = 1
    Sel= 0
    model.eval()
    with torch.no_grad():
        for i, (input, target, vis) in enumerate(val_loader):
            if i==Sel:
               input  = input [N:N+1]
               target = target[N:N+1]
               input  = input.to(device, non_blocking=True)
               target = target.to(device, non_blocking=True)
               output = model(input)
               att = save_output.outputs[0][0,0]
               output = output[-1]
               save_attention(input, output, target, att, target_th=0.6)
            
    return 0, 0         
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Training hourglass network')
    
    ## Parameters
    parser.add_argument('--datapath', type=str, required=True,
                        help='Path to trainset')
    parser.add_argument('-c', '--contrast', type=str, metavar='N', required=True,
                        help='MRI contrast')               
    parser.add_argument('--ndiscs', type=int, required=True,
                        help='Number of discs to detect')
    
    parser.add_argument('--resume', default= False, type=bool,
                        help='Resume the training from the last checkpoint')  
    parser.add_argument('--attshow', default= False, type=bool,
                        help=' Show the attention map') 
    parser.add_argument('--epochs', default=120, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--train_batch', default=3, type=int, metavar='N', 
                        help='train batchsize')
    parser.add_argument('--val-batch', default=4, type=int, metavar='N',
                        help='validation batchsize')
    parser.add_argument('--solver', metavar='SOLVER', default='rms',
                        choices=['rms', 'adam'],
                        help='optimizers')
    parser.add_argument('--lr', '--learning-rate', default=2.5e-4, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--weight-decay', '--wd', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)')
    parser.add_argument('--schedule', type=int, nargs='+', default=[60, 90],
                        help='Decrease learning rate at these epochs.')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='LR is multiplied by gamma on schedule.')
    parser.add_argument('-e', '--evaluate', default=False, type=bool,
                        help='evaluate model on validation set')
    parser.add_argument('--att', default=True, type=bool, 
                        help='Use attention or not')
    parser.add_argument('-s', '--stacks', default=2, type=int, metavar='N',
                        help='Number of hourglasses to stack')
    parser.add_argument('--features', default=256, type=int, metavar='N',
                        help='Number of features in the hourglass')
    parser.add_argument('-b', '--blocks', default=1, type=int, metavar='N',
                        help='Number of residual modules at each location in the hourglass')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--sigma-decay', type=float, default=0,
                        help='Sigma decay rate for each epoch.')
    parser.add_argument('--weight-folder', type=str, default='src/dlh/weights',
                        help='Folder where hourglass weights are stored and loaded')
    parser.add_argument('--visual-folder', type=str, default='test/visualize',
                        help='Folder where visuals are stored')

    # Create weights folder to store training weights
    if not os.path.exists(parser.parse_args().weight_folder):
        os.mkdir(parser.parse_args().weight_folder)
        
    # Create visualize folder to images created during training
    if not os.path.exists(parser.parse_args().visual_folder):
        os.mkdir(parser.parse_args().visual_folder)
        
    main(parser.parse_args())  # Train the hourglass network
    create_skeleton(parser.parse_args())  # Create skeleton file to improve hourglass accuracy during testing
    
