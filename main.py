from dataset import PascalDataset
from evaluation import SegmentationMetric
import numpy as np
import os.path as osp
import os
import sys
import torch
import einops
import torch.nn as nn
import torch.nn.functional as F

def block_images_einops(x, patch_size):
  """Image to patches."""
  batch, height, width, channels = x.shape
  grid_height = height // patch_size[0]
  grid_width = width // patch_size[1]
  x = einops.rearrange(
      x, "n (gh fh) (gw fw) c -> n (gh gw) (fh fw) c",
      gh=grid_height, gw=grid_width, fh=patch_size[0], fw=patch_size[1])
  return x


def unblock_images_einops(x, grid_size, patch_size):
  """patches to images."""
  x = einops.rearrange(
      x, "n (gh gw) (fh fw) c -> n (gh fh) (gw fw) c",
      gh=grid_size[0], gw=grid_size[1], fh=patch_size[0], fw=patch_size[1])
  return x


class GridGatingUnit(nn.Module):
  """A SpatialGatingUnit as defined in the gMLP paper.
  The 'spatial' dim is defined as the second last.
  If applied on other dims, you should swapaxes first.
  """
  
  def __init__(self, in_channel, spatial):
        # bn_layer not used
        super(GridGatingUnit, self).__init__()
        self.LN1 = nn.LayerNorm(in_channel // 2)
        self.linear = nn.Linear(spatial, spatial)

  def forward(self, x):
    u, v = torch.chunk(x, 2, dim=-1)
    v = self.LN1(v)
    #n = x.shape[-3]   # get spatial dim
    v = v.transpose(-1, -3)
    v = self.linear(v)
    v = v.transpose(-1, -3)
    return u * (v + 1.)


class GridGmlpLayer(nn.Module):
  """Grid gMLP layer that performs global mixing of tokens."""
  def __init__(self, in_channel, out_channel, grid_size, factor = 2, dropout_rate = 0.0):
      # bn_layer not used
      super(GridGmlpLayer, self).__init__()
      self.grid_size = grid_size
      self.LN1 = nn.LayerNorm(in_channel)
      self.linear1 = nn.Linear(in_channel, in_channel * factor)
      self.gelu = nn.GELU()
      self.ggu = GridGatingUnit(in_channel * factor, grid_size[0] * grid_size[1])
      self.linear2 = nn.Linear(in_channel, in_channel)
      self.linear3 = nn.Linear(in_channel, out_channel)
      self.dropout_rate = dropout_rate

  def forward(self, x, deterministic=True):
    x = x.permute(0, 2, 3, 1)
    n, h, w, num_channels = x.shape
    gh, gw = self.grid_size
    fh, fw = h // gh, w // gw
    x = block_images_einops(x, patch_size=(fh, fw))
    # gMLP1: Global (grid) mixing part, provides global grid communication.
    y = self.LN1(x)
    y = self.linear1(y)
    y = self.gelu(y)
    y = self.ggu(y)
    y = self.linear2(y)
    #y = nn.Dropout(self.dropout_rate)(y, deterministic)
    x = x + y
    x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
    x = self.linear3(x)
    x = x.permute(0, 3, 1, 2)
    return x


class BlockGatingUnit(nn.Module):
  """A SpatialGatingUnit as defined in the gMLP paper.
  The 'spatial' dim is defined as the **second last**.
  If applied on other dims, you should swapaxes first.
  """
  def __init__(self, in_channel, spatial):
        # bn_layer not used
        super(BlockGatingUnit, self).__init__()
        self.LN1 = nn.LayerNorm(in_channel // 2)
        self.linear = nn.Linear(spatial, spatial)

  def forward(self, x):
    u, v = torch.chunk(x, 2, dim=-1)
    v = self.LN1(v)
    #n = x.shape[-3]   # get spatial dim
    v = v.transpose(-1, -2)
    v = self.linear(v)
    v = v.transpose(-1, -2)
    return u * (v + 1.)


class BlockGmlpLayer(nn.Module):
  """Block gMLP layer that performs local mixing of tokens."""
  def __init__(self, in_channel, out_channel, block_size, factor = 2, dropout_rate = 0.0):
      # bn_layer not used
      super(BlockGmlpLayer, self).__init__()
      self.block_size = block_size
      self.LN1 = nn.LayerNorm(in_channel)
      self.linear1 = nn.Linear(in_channel, in_channel * factor)
      self.gelu = nn.GELU()
      self.bgu = BlockGatingUnit(in_channel * factor, block_size[0] * block_size[1])
      self.linear2 = nn.Linear(in_channel, in_channel)
      self.linear3 = nn.Linear(in_channel, out_channel)
      self.dropout_rate = dropout_rate

  def forward(self, x, deterministic=True):
    x = x.permute(0, 2, 3, 1)
    n, h, w, num_channels = x.shape
    fh, fw = self.block_size
    gh, gw = h // fh, w // fw
    x = block_images_einops(x, patch_size=(fh, fw))
    # gMLP1: Global (grid) mixing part, provides global grid communication.
    y = self.LN1(x)
    y = self.linear1(y)
    y = self.gelu(y)
    y = self.bgu(y)
    y = self.linear2(y)
    #y = nn.Dropout(self.dropout_rate)(y, deterministic)
    x = x + y
    x = unblock_images_einops(x, grid_size=(gh, gw), patch_size=(fh, fw))
    x = self.linear3(x)
    x = x.permute(0, 3, 1, 2)
    return x

class BasicBlock(nn.Module):
    def __init__(self, in_channel):
        super(BasicBlock, self).__init__()
        self.LN = nn.LayerNorm(in_channel)
        self.gelu = nn.GELU()
        self.Linear1 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        self.Local = BlockGmlpLayer(in_channel = in_channel // 2, out_channel = in_channel // 2, block_size = (8, 8))
        self.Global = GridGmlpLayer(in_channel = in_channel // 2, out_channel = in_channel // 2, grid_size = (8, 8))
        self.Linear2 = nn.Conv2d(in_channel, in_channel, 1, 1, 0)
        
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.LN(x)
        x = x.permute(0, 3, 1, 2)
        x = self.Linear1(x)
        x = self.gelu(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.Local(x1)
        x2 = self.Global(x2)
        x = torch.cat((x1, x2), 1)
        x = self.Linear2(x)
        return x

class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, 3, 1, 1) 
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.act = F.relu
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.pool(x)
        return x
        
class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, skip=True):
        super(UpBlock, self).__init__()
        if skip:
            self.conv1 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                conv3x3(in_channels * 2, out_channels))
        else:
            self.conv1 = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                conv3x3(in_channels, out_channels))
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.act = F.relu
        self.basic = BasicBlock(out_channel)
        
    def forward(self, x):
        if type(x) is tuple:
            x1, x2 = x 
            x = self.act(self.bn1(self.conv1(torch.cat((x1, x2), 1))))
        else:
            x = self.act(self.bn1(self.conv1(x)))
        x = self.basic(x)
        return x

class Unet(nn.Module):
    def __init__(self, in_channels=3, out_channels=7, basic_channel=32, n_layer=4):
        super(Unet, self).__init__()
        self.down = []
        self.up = []
        outs = None
        for i in range(n_layer):
            ins = in_channels if i == 0 else outs
            outs = basic_channel * 2 ** i
            down = DownBlock(ins, outs)
            up = UpBlock(outs, outs // 2) if i != n_layer - 1 else: UpBlock(outs, outs // 2, skip=False)
            self.down.append(down)
            self.up.append(up)
        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)
        self.final = nn.Sequential(nn.Conv2d(basic_channel // 2, basic_channel // 2, 3, 1, 1), nn.Conv2d(basic_channel // 2, out_channels, 1, 1, 0))
        
    def forward(self, x):
        encoder_outs = []
        for i in range(len(self.down)):
            x = self.down[i](x)
            encoder_outs.append(x)
        for i in range(len(self.up)):
            if i == 0:
                x = self.up[-(i+1)](x)
            else:
                x = self.up(x, encoder_outs[-(i+1)])
        x = self.final(x)
        return x

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
if __name__ == '__main__':
    train_batch = 16
    val_batch = 8
    train_loader = torch.utils.data.DataLoader(PascalDataset(True),batch_size=train_batch, shuffle=True,num_workers=4, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(PascalDataset(False),batch_size=val_batch, shuffle=True,num_workers=4, pin_memory=True)
    model = Unet().to(torch.device('cuda'))
    optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
    for epoch in range(100):
        print("Epoch", epoch + 1, 'starts training!')
        total_num = len(train_loader) // train_batch
        loss = 0
        total_loss = 0
        count = 0
        model.train()
        for i, batches in enumerate(self.train_loader):
            current_id = i + 1
            img = batches['img'].to(torch.device('cuda'))
            seg = batches['seg'].to(torch.device('cuda'))
            seg_pred = model(img)
            optimizer.zero_grad()
            B,C,H,W = seg_pred.shape
            seg_loss = nn.CrossEntropyLoss(seg_pred.view(B, C, -1), seg.squeeze().view(B, -1))
            seg_loss.backward()
            optimizer.step()
            total_loss += (seg_loss * B)
            count += B
            loss /= count
            if current_id % 100 == 1:
                print(current_id, "/", total_num, ':', loss)
        print("Epoch", epoch + 1, 'finishes training!')
        print("Epoch", epoch + 1, 'starts validating!')
        model.eval()
        best = 0
        best_epoch = 0
        with torch.no_grad():
            SM = SegmentationMetric(7)
            for i, batches in enumerate(self.val_loader):
                current_id = i + 1
                img = batches['img'].to(torch.device('cuda'))
                seg = batches['seg'].to(torch.device('cuda'))
                seg_pred = model(img)
                labels = torch.argmax(seg_pred, dim=1).unsqueeze(1)
                SM.addBatch(labels.reshape(-1), seg.reshape(-1))
                pa = SM.pixelAccuracy()
                cpa = SM.classPixelAccuracy()
                mpa = SM.meanPixelAccuracy()
                mIoU = SM.meanIntersectionOverUnion()
                if current_id % 100 == 1:
                    print(current_id, "/", total_num, ':', 'PA:', pa, 'CPA:', cpa, 'MPA:', mpa, 'mIoU:' mIoU)
            print("Total:")
            print('PA:', pa, 'CPA:', cpa, 'MPA:', mpa, 'mIoU:' mIoU)
            print("Epoch", epoch + 1, 'finishes validating!')
            if mIoU > best:
                torch.save(model.state_dict(), 'model/best.pt')
                best = mIoU
                best_epoch = epoch
                print("Save the best model on epoch", epoch + 1, "with the best mIoU", mIoU)
            print("The best model is on epoch", best_epoch + 1, "and the best mIoU is", mIoU)
                
        
            