#coding=utf8
from __future__ import print_function, division

import os,time,datetime
import numpy as np
from math import ceil
import datetime

import torch
from torch import nn
from torch.autograd import Variable
#from torchvision.utils import make_grid, save_image

from utils.utils import LossRecord, clip_gradient
from models.focal_loss import FocalLoss
from utils.eval_model import eval_turn
from utils.Asoftmax_loss import AngleLoss

import pdb

def dt():
    return datetime.datetime.now().strftime("%Y-%m-%d-%H_%M_%S")


def train(Config,
          model,
          epoch_num,
          start_epoch,
          optimizer,
          exp_lr_scheduler,
          data_loader,
          save_dir,
          data_size=448,
          savepoint=500,
          checkpoint=1000
          ):
    # savepoint: save without evalution
    # checkpoint: save with evaluation


    eval_train_flag = False
    rec_loss = []
    checkpoint_list = []

    train_batch_size = data_loader['train'].batch_size
    train_size = data_loader['train'].total_item_len
    train_epoch_step = data_loader['train'].__len__()
    train_loss_recorder = LossRecord(train_batch_size)

    if savepoint > train_epoch_step:
        savepoint = 1*train_epoch_step
        checkpoint = savepoint

    date_suffix = dt()
    log_file = open(os.path.join(Config.log_folder, 'formal_log_r50_dcl_%s_%s.log'%(str(data_size), date_suffix)), 'a')

    add_loss = nn.L1Loss()
    get_ce_loss = nn.CrossEntropyLoss()
    get_focal_loss = FocalLoss()
    get_angle_loss = AngleLoss()

    sig_acclist = []
    sig_wrlist = []
    for i in range(13):
        sig_acclist.append([])
        sig_wrlist.append([])
    for epoch in range(start_epoch,epoch_num):
        exp_lr_scheduler.step(epoch)
        model.train(True)
        step = 0
        train_corrects1 = 0
        train_corrects2 = 0
        train_corrects3 = 0
        save_grad = []
        for batch_cnt, data in enumerate(data_loader['train']):
            step += 1
            loss = 0
            model.train(True)
            if Config.use_backbone:
                inputs, labels, img_names = data
                inputs = Variable(inputs.cuda())
                labels = Variable(torch.from_numpy(np.array(labels)).cuda())

            if Config.use_dcl:
                inputs, labels, labels_swap, swap_law, img_names = data

                inputs = Variable(inputs.cuda())
                labels = Variable(torch.from_numpy(np.array(labels)).cuda())
                labels_swap = Variable(torch.from_numpy(np.array(labels_swap)).cuda())
                swap_law = Variable(torch.from_numpy(np.array(swap_law)).float().cuda())

            optimizer.zero_grad()

            if inputs.size(0) < 2*train_batch_size:
                outputs = model(inputs, inputs[0:-1:2])
            else:
                outputs = model(inputs, None)
            print('labels:',labels)
            sig_val = (outputs[3][0] + outputs[3][1])/2.0
            sig_val = np.array(sig_val.cpu().detach().numpy())

            for i in range(13):
                if i == labels[0].cpu().detach().numpy():
                    sig_acclist[i].append(sig_val[i])
                else:
                    sig_wrlist[i].append(sig_val[i])
            print('acclist:',sig_acclist,'\nwrlist:',sig_wrlist)
            #查看top3准确率
            outputs_pred = outputs[0] + outputs[1][:, 0:Config.numcls] + outputs[1][:, Config.numcls:2 * Config.numcls]
            top3_val, top3_pos = torch.topk(outputs_pred, 3)
            #print('top3val:',top3_val,'\npos:',top3_pos[:,0])
            batch_corrects1 = torch.sum((top3_pos[:, 0] == labels)).data.item()
            train_corrects1 += batch_corrects1
            #print('corrects 1:', batch_corrects1)
            batch_corrects2 = torch.sum((top3_pos[:, 1] == labels)).data.item()+batch_corrects1
            train_corrects2 += batch_corrects2
            #print('corrects 2:', batch_corrects2)
            batch_corrects3 = torch.sum((top3_pos[:, 2] == labels)).data.item()+batch_corrects2
            train_corrects3 += batch_corrects3
            #print('corrects 3:', batch_corrects3)

            if Config.use_focal_loss:
                ce_loss = get_focal_loss(outputs[0], labels)
            else:
                #print(labels)
                ce_loss = get_ce_loss(outputs[0], labels.long())

            if Config.use_Asoftmax:
                fetch_batch = labels.size(0)
                if batch_cnt % (train_epoch_step // 5) == 0:
                    angle_loss = get_angle_loss(outputs[3], labels[0:fetch_batch:2], decay=0.9)
                else:
                    angle_loss = get_angle_loss(outputs[3], labels[0:fetch_batch:2])
                loss += angle_loss

            loss += ce_loss

            alpha_ = 1
            beta_ = 1
            gamma_ = 0.01 if Config.dataset == 'STCAR' or Config.dataset == 'AIR' else 1
            if Config.use_dcl:
                swap_loss = get_ce_loss(outputs[1], labels_swap.long()) * beta_
                loss += swap_loss
                law_loss = add_loss(outputs[2], swap_law) * gamma_
                #print(law_loss)
                loss += law_loss

            loss.backward()
            torch.cuda.synchronize()

            optimizer.step()
            torch.cuda.synchronize()

            if Config.use_dcl:
                print('step: {:-8d} / {:d} acc1={:6.4f}, acc2={:6.4f}, acc3={:6.4f}'.format(step, train_epoch_step, batch_corrects1/labels.size(0), batch_corrects2/labels.size(0), batch_corrects3/labels.size(0)))
                print('step: {:-8d} / {:d} loss=ce_loss+swap_loss+law_loss: {:6.4f} = {:6.4f} + {:6.4f} + {:6.4f} '.format(step, train_epoch_step, loss.detach().item(), ce_loss.detach().item(), swap_loss.detach().item(), law_loss.detach().item()), flush=True)
            if Config.use_backbone:
                print('step: {:-8d} / {:d} loss=ce_loss+swap_loss+law_loss: {:6.4f} = {:6.4f} '.format(step, train_epoch_step, loss.detach().item(), ce_loss.detach().item()), flush=True)
            rec_loss.append(loss.detach().item())

            train_loss_recorder.update(loss.detach().item())

            # evaluation & save
            if step % checkpoint == 0:
                rec_loss = []
                print(32*'-', flush=True)
                print('step: {:d} / {:d} global_step: {:8.2f} train_epoch: {:04d} rec_train_loss: {:6.4f}'.format(step, train_epoch_step, 1.0*step/train_epoch_step, epoch, train_loss_recorder.get_val()), flush=True)
                print('current lr:%s' % exp_lr_scheduler.get_lr(), flush=True)
                if eval_train_flag:
                    trainval_acc1, trainval_acc2, trainval_acc3 = eval_turn(Config, model, data_loader['trainval'], 'trainval', epoch, log_file)
                    if abs(trainval_acc1 - trainval_acc3) < 0.01:
                        eval_train_flag = False

                val_acc1, val_acc2, val_acc3 = eval_turn(Config, model, data_loader['val'], 'val', epoch, log_file)

                save_path = os.path.join(save_dir, 'weights_%d_%d_%.4f_%.4f.pth'%(epoch, batch_cnt, val_acc1, val_acc3))
                torch.cuda.synchronize()
                torch.save(model.state_dict(), save_path)
                print('saved model to %s' % (save_path), flush=True)
                torch.cuda.empty_cache()

            # save only
            elif step % savepoint == 0:
                train_loss_recorder.update(rec_loss)
                rec_loss = []
                save_path = os.path.join(save_dir, 'savepoint_weights-%d-%s.pth'%(step, dt()))

                checkpoint_list.append(save_path)
                if len(checkpoint_list) == 6:
                    os.remove(checkpoint_list[0])
                    del checkpoint_list[0]
                torch.save(model.state_dict(), save_path)
                torch.cuda.empty_cache()

        print('train_size:',train_size,'train_corrects3:',train_corrects3)
        print('epoch: {:-8d} / {:d} acc1={:6.4f}, acc2={:6.4f}, acc3={:6.4f}'.format(epoch + 1, epoch_num, train_corrects1/train_size, train_corrects2/train_size, train_corrects3/train_size))


    log_file.close()



