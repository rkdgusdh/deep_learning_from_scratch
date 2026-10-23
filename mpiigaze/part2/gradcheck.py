"""수치 미분과 역전파 기울기 비교 (float64). 모든 레이어 종류를 포함한 작은 모델로 검사."""
import numpy as np
import nn
nn.set_dtype(np.float64)
rng = np.random.default_rng(0)

def build():
    r = np.random.default_rng(1)
    feat = nn.Sequential(nn.Conv2D(1, 4, 3, r), nn.BatchNorm(4), nn.ReLU(), nn.MaxPool2(),
                         nn.ResBlock(4, 4, r), nn.ResBlock(4, 6, r, stride=2),
                         nn.Conv2D(6, 5, 5, r, pad=2), nn.ReLU(),
                         nn.Flatten())
    return feat

x = rng.standard_normal((3, 9, 11, 1))
pose = rng.standard_normal((3, 2))
t = rng.standard_normal((3, 2))
feat = build()
fdim = feat.forward(x).shape[1]
model = nn.GazeNet(feat, nn.Sequential(nn.Affine(fdim + 2, 7, rng), nn.ReLU(), nn.Affine(7, 2, rng)))
model2 = nn.GazeNet(nn.Sequential(nn.Conv2D(1, 3, 3, rng, stride=2), nn.ReLU(), nn.GlobalAvgPool()),
                    nn.Sequential(nn.Affine(5, 2, rng)))

for name, m in [('cnn+res+bn+pool', model), ('stride2+gap', model2)]:
 for shift_min in [1, 1000]:  # 1: 시프트 합성곱 경로, 1000: im2col 경로
  nn.SHIFT_MIN_CIN = shift_min
  for lossname, loss in [('mse', nn.MSELoss()), ('smoothl1', nn.SmoothL1Loss(0.5))]:
    def f():
        return loss(m.forward(x, pose), t)
    f(); m.backward(loss.backward())
    worst = 0
    for l, k in nn.param_list(m):
        p, g = l.params[k], l.grads[k].copy()
        idx = rng.choice(p.size, min(20, p.size), replace=False)
        for i in idx:
            o = p.flat[i]; h = 1e-6
            p.flat[i] = o + h; fp = f(); p.flat[i] = o - h; fm = f(); p.flat[i] = o
            num = (fp - fm) / (2 * h)
            rel = abs(num - g.flat[i]) / max(1e-6, abs(num) + abs(g.flat[i]))
            worst = max(worst, rel)
    # 입력 기울기
    f(); dx = m.backward(loss.backward())
    for i in rng.choice(x.size, 20, replace=False):
        o = x.flat[i]; h = 1e-6
        x.flat[i] = o + h; fp = f(); x.flat[i] = o - h; fm = f(); x.flat[i] = o
        num = (fp - fm) / (2 * h)
        worst = max(worst, abs(num - dx.flat[i]) / max(1e-6, abs(num) + abs(dx.flat[i])))
    print(f'{name:16s} {"shift" if shift_min==1 else "im2col":6s} {lossname:9s} max relative error = {worst:.2e}')
