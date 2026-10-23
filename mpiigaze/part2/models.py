"""실험에 쓰는 모델 정의. 모두 nn.py의 NumPy 레이어만 사용."""
import numpy as np
import nn

IN_H, IN_W = 36, 60


def _feat_dim(feat):
    nn.set_train(feat, False)
    d = feat.forward(np.zeros((1, IN_H, IN_W, 1), nn.DTYPE)).shape[1]
    nn.set_train(feat, True)
    return d


def mlp(rng, hidden=(512, 256), dropout=0.0, use_pose=True):
    feat = nn.Sequential(nn.Flatten())
    layers, d = [], IN_H * IN_W + 2 * use_pose
    for h in hidden:
        layers += [nn.Affine(d, h, rng), nn.BatchNorm(h), nn.ReLU()]
        if dropout:
            layers.append(nn.Dropout(dropout, rng))
        d = h
    layers.append(nn.Affine(d, 2, rng))
    return nn.GazeNet(feat, nn.Sequential(*layers), use_pose)


def lenet(rng, c1=20, c2=50, fc=500, use_pose=True):
    """MPIIGaze 원 논문(Zhang et al., 2015)의 LeNet 계열 구조: conv5-pool-conv5-pool-FC500-[pose]-FC2."""
    feat = nn.Sequential(nn.Conv2D(1, c1, 5, rng, pad=0), nn.MaxPool2(),
                         nn.Conv2D(c1, c2, 5, rng, pad=0), nn.MaxPool2(),
                         nn.Flatten(), nn.Affine(6 * 12 * c2, fc, rng), nn.ReLU())
    head = nn.Sequential(nn.Affine(fc + 2 * use_pose, 2, rng))
    return nn.GazeNet(feat, head, use_pose)


def cnn_bn(rng, ch=(32, 64, 128), fc=256, dropout=0.0, use_pose=True):
    """평범한 VGG식 CNN (conv-BN-ReLU 두 번 + pool) x 3, 잔차 없음. 잔차 연결 효과 비교용."""
    layers, cin = [], 1
    for c in ch:
        layers += [nn.Conv2D(cin, c, 3, rng, bias=False), nn.BatchNorm(c), nn.ReLU(),
                   nn.Conv2D(c, c, 3, rng, bias=False), nn.BatchNorm(c), nn.ReLU(), nn.MaxPool2()]
        cin = c
    layers.append(nn.Flatten())
    feat = nn.Sequential(*layers)
    d = _feat_dim(feat)
    head = [nn.Affine(d + 2 * use_pose, fc, rng), nn.BatchNorm(fc), nn.ReLU()]
    if dropout:
        head.append(nn.Dropout(dropout, rng))
    head.append(nn.Affine(fc, 2, rng))
    return nn.GazeNet(feat, nn.Sequential(*head), use_pose)


def resnet(rng, ch=(32, 64, 128), blocks=(2, 2, 2), fc=256, dropout=0.0, use_pose=True, gap=False, stem_pool=False, skip=True):
    """작은 ResNet: stem conv3 -> stage마다 ResBlock (첫 블록 stride 2, 단 첫 stage는 stride 1)."""
    layers = [nn.Conv2D(1, ch[0], 3, rng, bias=False), nn.BatchNorm(ch[0]), nn.ReLU()]
    if stem_pool:
        layers.append(nn.MaxPool2())
    cin = ch[0]
    for si, (c, nb) in enumerate(zip(ch, blocks)):
        for bi in range(nb):
            stride = 2 if (bi == 0 and si > 0) else 1
            layers.append(nn.ResBlock(cin, c, rng, stride, skip))
            cin = c
    layers.append(nn.GlobalAvgPool() if gap else nn.Flatten())
    feat = nn.Sequential(*layers)
    d = _feat_dim(feat)
    head = [nn.Affine(d + 2 * use_pose, fc, rng), nn.BatchNorm(fc), nn.ReLU()]
    if dropout:
        head.append(nn.Dropout(dropout, rng))
    head.append(nn.Affine(fc, 2, rng))
    return nn.GazeNet(feat, nn.Sequential(*head), use_pose)


BUILDERS = {'mlp': mlp, 'lenet': lenet, 'cnn_bn': cnn_bn, 'resnet': resnet}


def build(name, rng, **kw):
    return BUILDERS[name](rng, **kw)
