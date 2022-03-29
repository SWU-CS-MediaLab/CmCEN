from __future__ import print_function
import argparse
import sys
import re
import time
import torch
from torch import autograd
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from data_loader import SYSUData, RegDBData, TestData
from data_manager import *
from eval_metrics import eval_sysu, eval_regdb
from model import embed_net
from utils import *
from loss import OriTripletLoss, TripletLoss_WRT
from heterogeneity_loss import hetero_loss
from tensorboardX import SummaryWriter
from dis_model import DIS

parser = argparse.ArgumentParser(description='PyTorch Cross-Modality Training')
parser.add_argument('--dataset', default='sysu', help='dataset name: regdb or sysu]')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate, 0.00035 for adam')
parser.add_argument('--optim', default='sgd', type=str, help='optimizer')
parser.add_argument('--arch', default='resnet50', type=str,
                    help='network baseline:resnet18 or resnet50')
parser.add_argument('--resume', '-r', default='', type=str,
                    help='resume from checkpoint')
parser.add_argument('--test-only', action='store_true', help='test only')
parser.add_argument('--model_path', default='save_model/', type=str,
                    help='model save path')
parser.add_argument('--save_epoch', default=20, type=int,
                    metavar='s', help='save model every 10 epochs')
parser.add_argument('--log_path', default='log/', type=str,
                    help='log save path')
parser.add_argument('--vis_log_path', default='log/vis_log/', type=str,
                    help='log save path')
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--img_w', default=144, type=int,
                    metavar='imgw', help='img width')
parser.add_argument('--img_h', default=288, type=int,
                    metavar='imgh', help='img height')
parser.add_argument('--batch-size', default=8, type=int,
                    metavar='B', help='training batch size')
parser.add_argument('--test-batch', default=64, type=int,
                    metavar='tb', help='testing batch size')
parser.add_argument('--method', default='CmCEN', type=str,
                    metavar='m', help='method type: base or CmCEN')
parser.add_argument('--margin', default=0.3, type=float,
                    metavar='margin', help='triplet loss margin')
parser.add_argument('--num_pos', default=4, type=int,
                    help='num of pos per identity in each modality')
parser.add_argument('--trial', default=1, type=int,
                    metavar='t', help='trial (only for RegDB dataset)')
parser.add_argument('--seed', default=0, type=int,
                    metavar='t', help='random seed')
parser.add_argument('--gpu', default='2', type=str,
                    help='gpu device ids for CUDA_VISIBLE_DEVICES')
parser.add_argument('--mode', default='all', type=str, help='all or indoor')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

set_seed(args.seed)

dataset = args.dataset
if dataset == 'sysu':
    data_path = '/home/wusong/Datasets/SYSU-MM01/'
    log_path = args.log_path + 'sysu_log/'
    test_mode = [1, 2]  # thermal to visible
elif dataset == 'regdb':
    data_path = '/home/wusong/RegDB/RegDB/'
    log_path = args.log_path + 'regdb_log/'
    test_mode = [2, 1]  # visible to thermal

checkpoint_path = args.model_path

if not os.path.isdir(log_path):
    os.makedirs(log_path)
if not os.path.isdir(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.isdir(args.vis_log_path):
    os.makedirs(args.vis_log_path)

suffix = dataset
if args.method == 'CmCEN':
    suffix = suffix + '_CmCEN_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)
else:
    suffix = suffix + '_base_p{}_n{}_lr_{}_seed_{}'.format(args.num_pos, args.batch_size, args.lr, args.seed)

if not args.optim == 'sgd':
    suffix = suffix + '_' + args.optim

if dataset == 'regdb':
    suffix = suffix + '_trial_{}'.format(args.trial)

sys.stdout = Logger(log_path + suffix + '_os.txt')

vis_log_dir = args.vis_log_path + suffix + '/'

if not os.path.isdir(vis_log_dir):
    os.makedirs(vis_log_dir)
writer = SummaryWriter(vis_log_dir)
print("==========\nArgs:{}\n==========".format(args))
device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu")  # 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0

print('==> Loading data..')
# Data loading code
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Pad(10),
    transforms.RandomCrop((args.img_h, args.img_w)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])
transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((args.img_h, args.img_w)),
    transforms.ToTensor(),
    normalize,
])

end = time.time()
if dataset == 'sysu':
    # training set
    trainset = SYSUData(data_path, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label, query_cam = process_query_sysu(data_path, mode=args.mode)
    gall_img, gall_label, gall_cam = process_gallery_sysu(data_path, mode=args.mode, trial=0)

elif dataset == 'regdb':
    # training set
    trainset = RegDBData(data_path, args.trial, transform=transform_train)
    # generate the idx of each person identity
    color_pos, thermal_pos = GenIdx(trainset.train_color_label, trainset.train_thermal_label)

    # testing set
    query_img, query_label = process_test_regdb(data_path, trial=args.trial, modal='visible')
    gall_img, gall_label = process_test_regdb(data_path, trial=args.trial, modal='thermal')

gallset = TestData(gall_img, gall_label, transform=transform_test, img_size=(args.img_w, args.img_h))
queryset = TestData(query_img, query_label, transform=transform_test, img_size=(args.img_w, args.img_h))

# testing data loader
gall_loader = data.DataLoader(gallset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)
query_loader = data.DataLoader(queryset, batch_size=args.test_batch, shuffle=False, num_workers=args.workers)

n_class = len(np.unique(trainset.train_color_label))
nquery = len(query_label)
ngall = len(gall_label)

print('Dataset {} statistics:'.format(dataset))
print('  ------------------------------')
print('  subset   | # ids | # images')
print('  ------------------------------')
print('  visible  | {:5d} | {:8d}'.format(n_class, len(trainset.train_color_label)))
print('  thermal  | {:5d} | {:8d}'.format(n_class, len(trainset.train_thermal_label)))
print('  ------------------------------')
print('  query    | {:5d} | {:8d}'.format(len(np.unique(query_label)), nquery))
print('  gallery  | {:5d} | {:8d}'.format(len(np.unique(gall_label)), ngall))
print('  ------------------------------')
print('Data Loading Time:\t {:.3f}'.format(time.time() - end))

print('==> Building model..')
if args.method == 'base':
    net = embed_net(n_class, no_local='off', gm_pool='off', arch=args.arch)
else:
    net = embed_net(n_class, no_local='on', gm_pool='on', arch=args.arch)

net.to(device)
cudnn.benchmark = True

if len(args.resume) > 0:
    model_path = checkpoint_path + args.resume
    if os.path.isfile(model_path):
        print('==> loading checkpoint {}'.format(args.resume))
        checkpoint = torch.load(model_path)
        start_epoch = checkpoint['epoch']
        net.load_state_dict(checkpoint['net'])
        print('==> loaded checkpoint {} (epoch {})'
              .format(args.resume, checkpoint['epoch']))
    else:
        print('==> no checkpoint found at {}'.format(args.resume))

# define loss function
criterion_id = nn.CrossEntropyLoss()
if args.method == 'CmCEN':
    criterion_tri = TripletLoss_WRT()
    criterion_het = hetero_loss()
else:
    loader_batch = args.batch_size * args.num_pos
    criterion_tri = OriTripletLoss(batch_size=loader_batch, margin=args.margin)

discriminator = DIS(2048, 1024, 64)

discriminator.to(device)
criterion_id.to(device)
criterion_tri.to(device)
criterion_id.to(device)

if args.optim == 'sgd':
    ignored_params = list(map(id, net.bottleneck.parameters())) \
                     + list(map(id, net.classifier.parameters()))

    base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

    optimizer = optim.SGD([
        {'params': base_params, 'lr': 0.1 * args.lr},
        {'params': net.bottleneck.parameters(), 'lr': args.lr},
        {'params': net.classifier.parameters(), 'lr': args.lr}],
        weight_decay=5e-4, momentum=0.9, nesterov=True)
    optimizer_dis = optim.SGD([{'params': discriminator.feature_dis.parameters(), 'lr': args.lr}], weight_decay=5e-4,
                              momentum=0.9, nesterov=True)
    # optimizer_dis = {
    #     'feature': optim.Adam(discriminator.feature_dis.parameters(), lr=args.lr, betas=(0.5, 0.9),
    #                           weight_decay=0.0001),
    # }
    # optimizer_dis = {
    #     'params': optim.Adam(base_params, lr=args.lr, betas=(0.5, 0.9), weight_decay=0.0001),
    #     'params': optim.Adam(net.bottleneck.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=0.0001),
    #     'params': optim.Adam(net.classifier.parameters(), lr=args.lr, betas=(0.5, 0.9), weight_decay=0.0001)
    # }


# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr * 0.1
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def adjust_dis_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        lr = args.lr
    elif epoch >= 20 and epoch < 50:
        lr = args.lr * 0.1
    elif epoch >= 50:
        lr = args.lr * 0.01

    optimizer.param_groups[0]['lr'] = 0.1 * lr
    for i in range(len(optimizer.param_groups) - 1):
        optimizer.param_groups[i + 1]['lr'] = lr

    return lr


def adjust_dis_learning_rate(optimizer_dis, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if epoch < 10:
        dis_lr = args.lr * (epoch + 1) / 10
    elif epoch >= 10 and epoch < 20:
        dis_lr = args.lr * 0.1
    elif epoch >= 20 and epoch < 50:
        dis_lr = args.lr * 0.1
    elif epoch >= 50:
        dis_lr = args.lr * 0.01

    optimizer_dis.param_groups[0]['lr'] = 0.1 * dis_lr
    for i in range(len(optimizer_dis.param_groups) - 1):
        optimizer_dis.param_groups[i + 1]['lr'] = dis_lr
    return dis_lr


def L1_penalty(var):  # 稀疏性約束
    return torch.abs(var).sum()


def train(epoch, slim_params,bn_threshold=7e-1):
    current_lr = adjust_learning_rate(optimizer, epoch)
    dis_current_lr = adjust_dis_learning_rate(optimizer_dis, epoch)
    train_loss = AverageMeter()
    id_loss = AverageMeter()
    het_loss = AverageMeter()
    tri_loss = AverageMeter()
    data_time = AverageMeter()
    batch_time = AverageMeter()
    correct = 0
    total = 0

    # switch to train mode
    net.train()
    end = time.time()

    for batch_idx, (input1, input2, label1, label2) in enumerate(trainloader):
        l1_norm = (L1_penalty(m).cuda() for m in slim_params)
        labels = torch.cat((label1, label2), 0)

        input1 = Variable(input1.cuda())
        input2 = Variable(input2.cuda())

        labels = Variable(labels.cuda())
        data_time.update(time.time() - end)

        feat, out0, real_feature, fake_feature = net(input1, input2)

        # train feature discriminator
        # train with rgb  real
        D_real_feature = discriminator.dis_feature(real_feature.detach())
        D_real_feature = -1 * torch.log(torch.sigmoid(D_real_feature)).mean()
        # print(D_real_feature)
        # D_real_feature = -D_real_feature.mean()
        optimizer_dis.zero_grad()
        D_real_feature.backward()
        # trian with fake
        D_fake_feature = discriminator.dis_feature(fake_feature.detach())
        D_fake_feature = -1 * torch.log(
            torch.ones(args.batch_size * 4).to(device) - torch.sigmoid(D_fake_feature)).mean()
        optimizer_dis.zero_grad()
        D_fake_feature.backward()
        # train with gradient penalty
        alpha = torch.randn(args.batch_size * 4, 2048).to(device)
        interpolates = alpha * D_real_feature.detach() + (1 - alpha) * D_fake_feature.detach()
        interpolates.requires_grad_()
        disc_interpolates = discriminator.dis_feature(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        feature_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        feature_gradient_penalty.backward()
        optimizer_dis.step()
        ##############################################################################################
        # train with ir  real
        D_ir_real_feature = discriminator.dis_feature(fake_feature.detach())
        D_ir_real_feature = -1 * torch.log(torch.sigmoid(D_ir_real_feature)).mean()
        # print(D_real_feature)
        # D_real_feature = -D_real_feature.mean()
        optimizer_dis.zero_grad()
        D_ir_real_feature.backward()
        # trian with fake
        D_rgb_fake_feature = discriminator.dis_feature(real_feature.detach())
        D_rgb_fake_feature = -1 * torch.log(
            torch.ones(args.batch_size * 4).to(device) - torch.sigmoid(D_rgb_fake_feature)).mean()
        optimizer_dis.zero_grad()
        D_rgb_fake_feature.backward()
        # train with gradient penalty
        alpha = torch.randn(args.batch_size * 4, 2048).to(device)
        interpolates = alpha * D_ir_real_feature.detach() + (1 - alpha) * D_rgb_fake_feature.detach()
        interpolates.requires_grad_()
        disc_interpolates = discriminator.dis_feature(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(device), create_graph=True,
                                  retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        feature_gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10
        feature_gradient_penalty.backward()
        optimizer_dis.step()
        ###########################################################################################
        het_out = out0.chunk(2, 0)
        loss_id = criterion_id(out0, labels)
        loss_tri, batch_acc = criterion_tri(feat, labels)
        loss_het = 0.001 * criterion_het(het_out[0], het_out[1], label1, label2)
        correct += (batch_acc / 2)
        _, predicted = out0.max(1)
        correct += (predicted.eq(labels).sum().item() / 2)

        loss = loss_id + loss_tri + loss_het

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update P
        train_loss.update(loss.item(), 2 * input1.size(0))
        id_loss.update(loss_id.item(), 2 * input1.size(0))
        tri_loss.update(loss_tri.item(), 2 * input1.size(0))
        het_loss.update(loss_het.item(), 2 * input1.size(0))
        total += labels.size(0)
        slim_params_list = []
        for slim_param in slim_params:
            slim_params_list.extend(slim_param.cpu().data.numpy())
        slim_params_list = np.array(sorted(slim_params_list))
        print('Epoch %d, 3%% smallest slim_params: %.4f' % (epoch, \
                                                            slim_params_list[len(slim_params_list) // 33]), flush=True)
        print('Epoch %d, portion of slim_params < %.e: %.4f' % (epoch, bn_threshold, \
                                                                sum(slim_params_list < bn_threshold) / len(
                                                                    slim_params_list)), flush=True)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if batch_idx % 50 == 0:
            print('Epoch: [{}][{}/{}] '
                  'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                  'lr:{:.3f} '
                  'Loss: {train_loss.val:.4f} ({train_loss.avg:.4f}) '
                  'iLoss: {id_loss.val:.4f} ({id_loss.avg:.4f}) '
                  'TLoss: {tri_loss.val:.4f} ({tri_loss.avg:.4f}) '
                  'Accu: {:.2f}'.format(
                epoch, batch_idx, len(trainloader), current_lr,
                100. * correct / total, batch_time=batch_time,
                train_loss=train_loss, id_loss=id_loss, tri_loss=tri_loss))

    writer.add_scalar('total_loss', train_loss.avg, epoch)
    writer.add_scalar('id_loss', id_loss.avg, epoch)
    writer.add_scalar('tri_loss', tri_loss.avg, epoch)
    writer.add_scalar('lr', current_lr, epoch)
    writer.add_scalar('dis_lr', dis_current_lr, epoch)


def test(epoch):
    # switch to evaluation mode
    net.eval()
    print('Extracting Gallery Feature...')
    start = time.time()
    ptr = 0
    gall_feat = np.zeros((ngall, 2048))
    gall_feat_att = np.zeros((ngall, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(gall_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[0])
            gall_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            gall_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    # switch to evaluation
    net.eval()
    print('Extracting Query Feature...')
    start = time.time()
    ptr = 0
    query_feat = np.zeros((nquery, 2048))
    query_feat_att = np.zeros((nquery, 2048))
    with torch.no_grad():
        for batch_idx, (input, label) in enumerate(query_loader):
            batch_num = input.size(0)
            input = Variable(input.cuda())
            feat, feat_att = net(input, input, test_mode[1])
            query_feat[ptr:ptr + batch_num, :] = feat.detach().cpu().numpy()
            query_feat_att[ptr:ptr + batch_num, :] = feat_att.detach().cpu().numpy()
            ptr = ptr + batch_num
    print('Extracting Time:\t {:.3f}'.format(time.time() - start))

    start = time.time()
    # compute the similarity
    distmat = np.matmul(query_feat, np.transpose(gall_feat))
    distmat_att = np.matmul(query_feat_att, np.transpose(gall_feat_att))

    # evaluation
    if dataset == 'regdb':
        cmc, mAP, mINP = eval_regdb(-distmat, query_label, gall_label)
        cmc_att, mAP_att, mINP_att = eval_regdb(-distmat_att, query_label, gall_label)
    elif dataset == 'sysu':
        cmc, mAP, mINP = eval_sysu(-distmat, query_label, gall_label, query_cam, gall_cam)
        cmc_att, mAP_att, mINP_att = eval_sysu(-distmat_att, query_label, gall_label, query_cam, gall_cam)
    print('Evaluation Time:\t {:.3f}'.format(time.time() - start))

    writer.add_scalar('rank1', cmc[0], epoch)
    writer.add_scalar('mAP', mAP, epoch)
    writer.add_scalar('mINP', mINP, epoch)
    writer.add_scalar('rank1_att', cmc_att[0], epoch)
    writer.add_scalar('mAP_att', mAP_att, epoch)
    writer.add_scalar('mINP_att', mINP_att, epoch)
    return cmc, mAP, mINP, cmc_att, mAP_att, mINP_att


# training
print('==> Start Training...')
for epoch in range(start_epoch, 81 - start_epoch):
    enc_params, dec_params, slim_params = [], [], []
    for name, param in net.named_parameters():
        if bool(re.match('.*conv1.*|.*bn1.*|.*layer.*|.*bn_.*', name)):
            enc_params.append(param)
            # print('enc.parameter:{}'.format(name))
        else:
            dec_params.append(param)
        if param.requires_grad and name.endswith('weight') and 'bn_' in name:
            # print('slim:{}'.format(name))
            if len(slim_params) % 2 == 0:
                slim_params.append(param[:len(param) // 2])
            else:
                slim_params.append(param[len(param) // 2:])

    print('==> Preparing Data Loader...')
    # identity sampler
    sampler = IdentitySampler(trainset.train_color_label, \
                              trainset.train_thermal_label, color_pos, thermal_pos, args.num_pos, args.batch_size,
                              epoch)

    trainset.cIndex = sampler.index1  # color index
    trainset.tIndex = sampler.index2  # thermal index
    print(epoch)
    print(trainset.cIndex)
    print(trainset.tIndex)

    loader_batch = args.batch_size * args.num_pos

    trainloader = data.DataLoader(trainset, batch_size=loader_batch, \
                                  sampler=sampler, num_workers=args.workers, drop_last=True)

    # training
    train(epoch,slim_params)

    if epoch > 0 and epoch % 2 == 0:
        print('Test Epoch: {}'.format(epoch))

        # testing
        cmc, mAP, mINP, cmc_att, mAP_att, mINP_att = test(epoch)
        # save model
        if cmc_att[0] > best_acc:  # not the real best for sysu-mm01
            best_acc = cmc_att[0]
            best_epoch = epoch
            state = {
                'net': net.state_dict(),
                'cmc': cmc_att,
                'mAP': mAP_att,
                'mINP': mINP_att,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_best.t')

        # save model
        if epoch > 10 and epoch % args.save_epoch == 0:
            state = {
                'net': net.state_dict(),
                'cmc': cmc,
                'mAP': mAP,
                'epoch': epoch,
            }
            torch.save(state, checkpoint_path + suffix + '_epoch_{}.t'.format(epoch))

        print(
            'POOL:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc[0], cmc[4], cmc[9], cmc[19], mAP, mINP))
        print(
            'FC:   Rank-1: {:.2%} | Rank-5: {:.2%} | Rank-10: {:.2%}| Rank-20: {:.2%}| mAP: {:.2%}| mINP: {:.2%}'.format(
                cmc_att[0], cmc_att[4], cmc_att[9], cmc_att[19], mAP_att, mINP_att))
        print('Best Epoch [{}]'.format(best_epoch))
