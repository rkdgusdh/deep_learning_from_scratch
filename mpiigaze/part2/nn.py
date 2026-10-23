"""순수 NumPy 신경망 프레임워크 (교재 스타일: 각 레이어가 forward/backward를 직접 구현).

텐서 배치는 NHWC. 모든 파라미터는 layer.params[name], 기울기는 layer.grads[name].
"""
import numpy as np

DTYPE = np.float32
SHIFT_MIN_CIN = 16  # 입력 채널이 적으면 im2col이 더 빠르다


def set_dtype(dt):
    global DTYPE
    DTYPE = dt


class Layer:
    def __init__(self):
        self.params, self.grads = {}, {}
        self.train = True

    def children(self):
        return []

    def all_layers(self):
        out = [self]
        for c in self.children():
            out += c.all_layers()
        return out


def he(shape, fan_in, rng):
    return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(DTYPE)


# ---------------------------------------------------------------- 기본 레이어
class Affine(Layer):
    def __init__(self, din, dout, rng, bias=True):
        super().__init__()
        self.params['W'] = he((din, dout), din, rng)
        if bias:
            self.params['b'] = np.zeros(dout, DTYPE)

    def forward(self, x):
        self.x = x
        y = x @ self.params['W']
        if 'b' in self.params:
            y = y + self.params['b']
        return y

    def backward(self, dy):
        self.grads['W'] = self.x.T @ dy
        if 'b' in self.params:
            self.grads['b'] = dy.sum(0)
        return dy @ self.params['W'].T


class ReLU(Layer):
    def forward(self, x):
        self.mask = x > 0
        return x * self.mask

    def backward(self, dy):
        return dy * self.mask


class Flatten(Layer):
    def forward(self, x):
        self.shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dy):
        return dy.reshape(self.shape)


class Dropout(Layer):
    def __init__(self, p, rng):
        super().__init__()
        self.p, self.rng = p, rng

    def forward(self, x):
        if not self.train or self.p == 0:
            return x
        self.mask = (self.rng.random(x.shape) >= self.p).astype(x.dtype) / (1 - self.p)
        return x * self.mask

    def backward(self, dy):
        if not self.train or self.p == 0:
            return dy
        return dy * self.mask


class Conv2D(Layer):
    """합성곱. W: (KH, KW, Cin, Cout).
    stride 1: '평탄화 시프트' 방식. 패딩된 입력을 (N*Hp*Wp, Cin) 행렬로 펴면 커널 위치 (i, j)는
    행 오프셋 i*Wp + j 만큼의 연속 구간이 되므로, im2col 복사 없이 k*k번의 행렬곱과 연속 덧셈으로 계산한다.
    stride 2: 교재의 im2col 방식."""

    def __init__(self, cin, cout, k, rng, stride=1, pad=None, bias=True):
        super().__init__()
        self.k, self.s = k, stride
        self.pad = k // 2 if pad is None else pad
        self.params['W'] = he((k, k, cin, cout), k * k * cin, rng)
        if bias:
            self.params['b'] = np.zeros(cout, DTYPE)

    def forward(self, x):
        N, H, W, C = x.shape
        k, s, p = self.k, self.s, self.pad
        xp = np.pad(x, ((0, 0), (p, p), (p, p), (0, 0))) if p else x
        Hp, Wp = H + 2 * p, W + 2 * p
        Ho, Wo = (Hp - k) // s + 1, (Wp - k) // s + 1
        Wt = self.params['W']
        self.shift = s == 1 and k <= 3 and C >= SHIFT_MIN_CIN
        self.cache = (x.shape, Ho, Wo)
        if self.shift:
            xf = xp.reshape(-1, C)
            L = xf.shape[0] - ((k - 1) * Wp + (k - 1))
            Y = np.zeros((N * Hp * Wp, Wt.shape[-1]), x.dtype)
            for i in range(k):
                for j in range(k):
                    o = i * Wp + j
                    Y[:L] += xf[o:o + L] @ Wt[i, j]
            self.xf = xf
            y = Y.reshape(N, Hp, Wp, -1)[:, :Ho, :Wo]
        else:
            cols = np.concatenate([xp[:, i:i + s * Ho:s, j:j + s * Wo:s, :]
                                   for i in range(k) for j in range(k)], axis=-1)
            self.cols = cols.reshape(N * Ho * Wo, -1)
            y = (self.cols @ Wt.reshape(-1, Wt.shape[-1])).reshape(N, Ho, Wo, -1)
        if 'b' in self.params:
            y = y + self.params['b']
        return y

    def backward(self, dy):
        (N, H, W, C), Ho, Wo = self.cache
        k, s, p = self.k, self.s, self.pad
        Hp, Wp = H + 2 * p, W + 2 * p
        Wt = self.params['W']
        Co = Wt.shape[-1]
        if 'b' in self.params:
            self.grads['b'] = dy.sum(axis=(0, 1, 2))
        if self.shift:
            dY = np.zeros((N, Hp, Wp, Co), dy.dtype)
            dY[:, :Ho, :Wo] = dy
            dY = dY.reshape(-1, Co)
            xf = self.xf
            L = xf.shape[0] - ((k - 1) * Wp + (k - 1))
            dW = np.empty_like(Wt)
            dxp = np.zeros_like(xf)
            for i in range(k):
                for j in range(k):
                    o = i * Wp + j
                    dW[i, j] = xf[o:o + L].T @ dY[:L]
                    dxp[o:o + L] += dY[:L] @ Wt[i, j].T
            self.grads['W'] = dW
            self.xf = None
            dxp = dxp.reshape(N, Hp, Wp, C)
        else:
            dy2 = dy.reshape(-1, Co)
            self.grads['W'] = (self.cols.T @ dy2).reshape(Wt.shape)
            dcols = (dy2 @ Wt.reshape(-1, Co).T).reshape(N, Ho, Wo, k, k, C)
            dxp = np.zeros((N, Hp, Wp, C), dy.dtype)
            for i in range(k):
                for j in range(k):
                    dxp[:, i:i + s * Ho:s, j:j + s * Wo:s, :] += dcols[:, :, :, i, j, :]
            self.cols = None
        return dxp[:, p:p + H, p:p + W, :] if p else dxp


class MaxPool2(Layer):
    """2x2, stride 2 최대 풀링 (홀수 크기는 마지막 행/열을 버림)."""

    def forward(self, x):
        N, H, W, C = x.shape
        H2, W2 = H // 2, W // 2
        self.xshape = x.shape
        xr = x[:, :H2 * 2, :W2 * 2, :].reshape(N, H2, 2, W2, 2, C)
        y = xr.max(axis=(2, 4))
        self.mask = (xr == y[:, :, None, :, None, :])
        # 동률이면 기울기를 동률 위치들에 균등 분배한다
        self.mask = self.mask / self.mask.sum(axis=(2, 4), keepdims=True)
        return y

    def backward(self, dy):
        N, H, W, C = self.xshape
        H2, W2 = H // 2, W // 2
        dx = np.zeros(self.xshape, dy.dtype)
        dx[:, :H2 * 2, :W2 * 2, :] = (self.mask * dy[:, :, None, :, None, :]).reshape(N, H2 * 2, W2 * 2, C)
        return dx


class GlobalAvgPool(Layer):
    def forward(self, x):
        self.shape = x.shape
        return x.mean(axis=(1, 2))

    def backward(self, dy):
        N, H, W, C = self.shape
        return np.broadcast_to(dy[:, None, None, :] / (H * W), self.shape).copy()


class BatchNorm(Layer):
    """채널(마지막 축) 기준 배치 정규화. 2D(N,C)와 4D(N,H,W,C) 모두 지원."""

    def __init__(self, c, momentum=0.9, eps=1e-5):
        super().__init__()
        self.params['gamma'] = np.ones(c, DTYPE)
        self.params['beta'] = np.zeros(c, DTYPE)
        self.run_mean = np.zeros(c, DTYPE)
        self.run_var = np.ones(c, DTYPE)
        self.m, self.eps = momentum, eps

    def forward(self, x):
        axes = tuple(range(x.ndim - 1))
        if self.train:
            mu = x.mean(axes)
            var = x.var(axes)
            self.run_mean = self.m * self.run_mean + (1 - self.m) * mu
            self.run_var = self.m * self.run_var + (1 - self.m) * var
        else:
            mu, var = self.run_mean, self.run_var
        self.std = np.sqrt(var + self.eps)
        self.xhat = (x - mu) / self.std
        return self.params['gamma'] * self.xhat + self.params['beta']

    def backward(self, dy):
        axes = tuple(range(dy.ndim - 1))
        m = dy.size // dy.shape[-1]
        self.grads['gamma'] = (dy * self.xhat).sum(axes)
        self.grads['beta'] = dy.sum(axes)
        dxhat = dy * self.params['gamma']
        return (dxhat - dxhat.mean(axes) - self.xhat * (dxhat * self.xhat).mean(axes)) / self.std


# ---------------------------------------------------------------- 조합 레이어
class Sequential(Layer):
    def __init__(self, *layers):
        super().__init__()
        self.layers = list(layers)

    def children(self):
        return self.layers

    def forward(self, x):
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, dy):
        for l in reversed(self.layers):
            dy = l.backward(dy)
        return dy


class ResBlock(Layer):
    """He et al. (2016)의 basic block: conv-BN-ReLU-conv-BN + shortcut, 이후 ReLU.
    채널이나 해상도가 바뀌면 shortcut에 1x1 conv + BN을 쓴다."""

    def __init__(self, cin, cout, rng, stride=1, skip=True):
        super().__init__()
        self.skip = skip  # False면 같은 층 구성에서 shortcut만 뺀 '평범한' 블록 (잔차 효과 비교용)
        self.main = Sequential(Conv2D(cin, cout, 3, rng, stride, bias=False), BatchNorm(cout), ReLU(),
                               Conv2D(cout, cout, 3, rng, 1, bias=False), BatchNorm(cout))
        self.short = None
        if skip and (stride != 1 or cin != cout):
            self.short = Sequential(Conv2D(cin, cout, 1, rng, stride, pad=0, bias=False), BatchNorm(cout))
        self.relu = ReLU()

    def children(self):
        return [self.main, self.relu] + ([self.short] if self.short else [])

    def forward(self, x):
        y = self.main.forward(x)
        if self.skip:
            y = y + (self.short.forward(x) if self.short else x)
        return self.relu.forward(y)

    def backward(self, dy):
        dy = self.relu.backward(dy)
        dx = self.main.backward(dy)
        if not self.skip:
            return dx
        return dx + (self.short.backward(dy) if self.short else dy)


class GazeNet(Layer):
    """이미지 특징 추출기(feat) -> [특징, head pose] 결합 -> 회귀 헤드(head).
    MPIIGaze 원 논문처럼 head pose 각도를 FC 단에 이어 붙인다."""

    def __init__(self, feat, head, use_pose=True):
        super().__init__()
        self.feat, self.head, self.use_pose = feat, head, use_pose

    def children(self):
        return [self.feat, self.head]

    def forward(self, x, pose):
        f = self.feat.forward(x)
        self.fdim = f.shape[1]
        if self.use_pose:
            f = np.concatenate([f, pose.astype(f.dtype)], 1)
        return self.head.forward(f)

    def backward(self, dy):
        df = self.head.backward(dy)
        return self.feat.backward(df[:, :self.fdim])


def set_train(model, flag):
    for l in model.all_layers():
        l.train = flag


def param_list(model):
    """(layer, name) 쌍 목록. 옵티마이저와 저장에 사용."""
    out = []
    for l in model.all_layers():
        for k in l.params:
            out.append((l, k))
    return out


def n_params(model):
    return sum(l.params[k].size for l, k in param_list(model))


def state_dict(model):
    sd = {}
    for i, l in enumerate(model.all_layers()):
        for k, v in l.params.items():
            sd[f'{i}.{k}'] = v.copy()
        if isinstance(l, BatchNorm):
            sd[f'{i}.rm'] = l.run_mean.copy()
            sd[f'{i}.rv'] = l.run_var.copy()
    return sd


def load_state(model, sd):
    for i, l in enumerate(model.all_layers()):
        for k in l.params:
            l.params[k][...] = sd[f'{i}.{k}']
        if isinstance(l, BatchNorm):
            l.run_mean = sd[f'{i}.rm'].copy()
            l.run_var = sd[f'{i}.rv'].copy()


# ---------------------------------------------------------------- 손실
def angles_to_vec(a):
    p, y = a[:, 0], a[:, 1]
    return np.stack([-np.cos(p) * np.sin(y), -np.sin(p), -np.cos(p) * np.cos(y)], 1)


def angular_error_deg(pred, true):
    a, b = angles_to_vec(pred.astype(np.float64)), angles_to_vec(true.astype(np.float64))
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


class MSELoss:
    def __call__(self, pred, t):
        self.d = pred - t
        return float((self.d ** 2).sum(1).mean())

    def backward(self):
        return (2 * self.d / self.d.shape[0]).astype(self.d.dtype)


class L1Loss:
    def __call__(self, pred, t):
        self.d = pred - t
        return float(np.abs(self.d).sum(1).mean())

    def backward(self):
        return (np.sign(self.d) / self.d.shape[0]).astype(self.d.dtype)


class SmoothL1Loss:
    def __init__(self, beta):
        self.beta = beta

    def __call__(self, pred, t):
        d = pred - t
        self.d = d
        a = np.abs(d)
        l = np.where(a < self.beta, 0.5 * d * d / self.beta, a - 0.5 * self.beta)
        return float(l.sum(1).mean())

    def backward(self):
        d = self.d
        g = np.where(np.abs(d) < self.beta, d / self.beta, np.sign(d))
        return (g / d.shape[0]).astype(d.dtype)


# ---------------------------------------------------------------- 옵티마이저
class Adam:
    """Adam (Kingma & Ba, 2015). wd>0이면 decoupled weight decay (AdamW)를 W에만 적용."""

    def __init__(self, model, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8, wd=0.0):
        self.plist = param_list(model)
        self.lr, self.b1, self.b2, self.eps, self.wd = lr, b1, b2, eps, wd
        self.m = [np.zeros_like(l.params[k]) for l, k in self.plist]
        self.v = [np.zeros_like(l.params[k]) for l, k in self.plist]
        self.t = 0

    def step(self):
        self.t += 1
        c1 = 1 - self.b1 ** self.t
        c2 = 1 - self.b2 ** self.t
        for i, (l, k) in enumerate(self.plist):
            g = l.grads[k]
            p = l.params[k]
            self.m[i] *= self.b1
            self.m[i] += (1 - self.b1) * g
            self.v[i] *= self.b2
            self.v[i] += (1 - self.b2) * g * g
            if self.wd and k == 'W':
                p -= self.lr * self.wd * p
            p -= (self.lr * (self.m[i] / c1) / (np.sqrt(self.v[i] / c2) + self.eps)).astype(p.dtype)


class SGD:
    def __init__(self, model, lr=0.01, momentum=0.9, wd=0.0):
        self.plist = param_list(model)
        self.lr, self.mom, self.wd = lr, momentum, wd
        self.v = [np.zeros_like(l.params[k]) for l, k in self.plist]

    def step(self):
        for i, (l, k) in enumerate(self.plist):
            g = l.grads[k]
            if self.wd and k == 'W':
                g = g + self.wd * l.params[k]
            self.v[i] = self.mom * self.v[i] - self.lr * g
            l.params[k] += self.v[i]
