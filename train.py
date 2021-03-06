
import os
import sys

import multiprocessing  

import argparse

import numpy as np
import matplotlib.pyplot as plt

import torch
import torchvision.transforms as transforms
import torch.optim as optim


from TRD import TRD, TRDLoss
from bbox_tr import plot_bbox

from PairFileDataset import PairFileDataset


def my_collate_fn(batch):
    images = [item[0] for item in batch]
    images = torch.stack(images,0)
    targets_np = [item[1] for item in batch]
    targets = []
    for target_np in targets_np:
        target = {key: torch.tensor(target_np[key]) for key in target_np}
        targets.append(target)
    return images, targets_np

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def train_net(net,
              save_path,
              dataset_path,              
              image_ext='.png',
              start_epoch=0,
              epochs=1000,
              batch_size=1,             
              lr=0.01,
              momentum=0.9):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    net.to(device)
    criterion = TRDLoss(net.bboxw_range,net.image_size)
    # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    # optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=momentum)
    optimizer = optim.Adam(net.parameters(), lr=lr,betas=(momentum,0.999), weight_decay=1e-8)

    transform = transforms.Compose([
        # transforms.Resize([net.image_size,net.image_size]),
        transforms.ToTensor()])
    
    trainset = PairFileDataset(dataset_path,image_ext,transform= transform)
    
    image, target = trainset[3]
    bboxes = target['bboxes']
    cids = target['labels']
    image = image*255    # unnormalize
    image = image.numpy()
    image = np.transpose(image, (1, 2, 0))
    plot_bbox(image, bboxes, labels=cids,absolute_coordinates=False)
    plt.show()

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=True, num_workers=1,collate_fn = my_collate_fn)   

    # dataiter = iter(trainloader)
    # images, targets = dataiter.next()
    # imshow(torchvision.utils.make_grid(images))
    # images = images.to(device)
    # pred = net.detect(images,score_thresh=0.4)
    # image = images[0,:,:,:]
    # image = image*255    # unnormalize
    # image = image.to('cpu')
    # image = image.numpy()
    # image = np.transpose(image, (1, 2, 0))
    # plot_bbox(image, pred)
    # plt.show()    

    log_batchs = 1
    # 正例置信度
    score_loss_p_log = 0.0            
    # 负例置信度
    score_loss_n_log = 0.0         
    # 范围框数值部分
    bboxv_loss_log = 0.0 
    # 范围框符号部分
    bboxs_loss_log = 0.0 
    # 类别标签
    label_loss_log = 0.0 

    for epoch in range(start_epoch,start_epoch+epochs):  # loop over the dataset multiple times
        net.train()
        for i, (images, targets) in enumerate(trainloader, 0):
            # imshow(torchvision.utils.make_grid(images))

            # get the inputs
            images = images.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()
    
            # forward + backward + optimize
            outputs = net(images)

            loss = criterion(outputs, targets)

            score_loss_p_log = score_loss_p_log + criterion.score_loss_p_log
            score_loss_n_log = score_loss_n_log + criterion.score_loss_n_log
            bboxv_loss_log = bboxv_loss_log + criterion.bboxv_loss_log
            bboxs_loss_log = bboxs_loss_log + criterion.bboxs_loss_log
            label_loss_log = label_loss_log + criterion.label_loss_log

            loss.backward()
            optimizer.step()

            # print log
            if i % log_batchs == (log_batchs-1):    # print every log_batchs mini-batches
                log_msg = '[%d, %7d] sum_loss: %.5f score_p: %.5f score_n: %.5f bboxv: %.5f bboxs: %.5f label: %.5f' %(
                    epoch+1, i + 1, 
                    loss, 
                    score_loss_p_log/log_batchs, 
                    score_loss_n_log/log_batchs,
                    bboxv_loss_log/log_batchs,
                    bboxs_loss_log/log_batchs, 
                    label_loss_log/log_batchs)
                print(log_msg)
                # log_file.writeline(log_msg)
                # 正例置信度
                score_loss_p_log = 0.0            
                # 负例置信度
                score_loss_n_log = 0.0         
                # 范围框数值部分
                bboxv_loss_log = 0.0 
                # 范围框符号部分
                bboxs_loss_log = 0.0 
                # 类别标签
                label_loss_log = 0.0 
        
        if (epoch+1) % 50 == 0:
            model_path = os.path.join(save_path,'TRD_%d.pth'%(epoch+1))
            torch.save(net.state_dict(), model_path)
    
    model_path = os.path.join(save_path,'TRD_final.pth')
    torch.save(net.state_dict(), model_path)
    print('Finished Training')

def get_args():
    parser = argparse.ArgumentParser(description='Train the TRD on splited images and TRA lablels',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-s', '--save-path', metavar='S', type=str, default=r'.\lan4',
                        help='Model saving path', dest='save_path')                                 
    parser.add_argument('-d', '--dataset-path', metavar='D', type=str, default=r'D:\cvImageSamples\lan4\splited_imgs',
                        help='Dataset path', dest='dataset_path')
    parser.add_argument('-iz', '--image-size', metavar='IZ', type=int, default=608,
                        help='Image size', dest='image_size')
    parser.add_argument('-ie', '--image-ext', metavar='IE', type=str, default='.bmp',
                        help='Image extension name', dest='image_ext')

    parser.add_argument('-se', '--start-epoch', metavar='SE', type=int, default=0,
                        help='Start epoch', dest='start_epoch')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=1000,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=1,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-c', '--num-classes', metavar='C', type=int, default=1,
                        help='number of classes', dest='num_classes')


    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-m', '--momentum', metavar='M', type=float, default=0.9,
                        help='Momentum', dest='momentum')
    parser.add_argument('-f', '--load', type=str, default=r"E:\SourceCode\Python\pytorch_test\resnet50-19c8e357.pth",
                        help='Load model from a .pth file', dest='load')
    
    return parser.parse_args()

if __name__ == '__main__':
    torch.multiprocessing.freeze_support()

    args = get_args()
    
    # 
    bboxw_range = [(48,320),(24,160),(12,80)]
    net = TRD(bboxw_range,args.image_size,args.num_classes)  

    # if args.load:
    #     net.load_state_dict(
    #         torch.load(args.load)
    #     )
    
    if args.load:
        # 加载预训练的resnet参数
        pretrained_dict = torch.load(args.load)
        model_dict = net.state_dict()  
        #将pretrained_dict里不属于model_dict的键剔除掉 
        pretrained_dict =  {k: v for k, v in pretrained_dict.items() if k in model_dict}         
        # 更新现有的model_dict 
        model_dict.update(pretrained_dict)         
        # 加载我们真正需要的state_dict 
        net.load_state_dict(model_dict)

    try:
        train_net(net=net,
                  save_path=args.save_path,
                  dataset_path=args.dataset_path,
                  image_ext=args.image_ext,                  
                  start_epoch=args.start_epoch,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  momentum=args.momentum)
    except Exception:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')

