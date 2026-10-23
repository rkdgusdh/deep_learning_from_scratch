"""nn.py 레이어가 교재(밑바닥부터 시작하는 딥러닝) common/ 레이어와 수치적으로 같은지 확인한다.
사용법: python test_textbook_equiv.py [교재 저장소 경로]  (생략하면 이 저장소의 루트)

nn.py는 교재 레이어를 NHWC 배치로 옮기고 합성곱 계산만 빠르게 바꾼 것이다.
 - Conv2D        == common.layers.Convolution   (NCHW <-> NHWC 변환 후 비교)
 - MaxPool2      == common.layers.Pooling(2, 2, stride=2)
 - Affine, ReLU  == common.layers.Affine, Relu
 - BatchNorm(2D) == common.layers.BatchNormalization
 - BatchNorm(4D) == 교재 BatchNormalization을 (N*H*W, C)로 펼친 입력에 적용한 것 (채널별 BN)
 - Adam          == common.optimizer.Adam (같은 갱신식, eps만 다름)
"""
import sys, os
import numpy as np
sys.path.insert(0, os.path.abspath(sys.argv[1]) if len(sys.argv) > 1 else
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
from common.layers import Convolution, Pooling, Affine as TAffine, Relu, BatchNormalization
from common.optimizer import Adam as TAdam
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nn
nn.set_dtype(np.float64)
rng = np.random.default_rng(0)
ok = True

def check(name, a, b, tol=1e-9):
    global ok
    e = np.abs(a - b).max() / max(1e-12, np.abs(b).max())
    ok &= e < tol
    print(f'{name:38s} max rel diff = {e:.1e} {"OK" if e < tol else "FAIL"}')

to_nchw = lambda x: x.transpose(0, 3, 1, 2)
to_nhwc = lambda x: x.transpose(0, 2, 3, 1)

for (cin, cout, k, s, p, thr) in [(1, 8, 5, 1, 0, 16), (20, 16, 3, 1, 1, 16), (20, 16, 3, 1, 1, 1000), (16, 24, 3, 2, 1, 16), (16, 8, 1, 2, 0, 16)]:
    nn.SHIFT_MIN_CIN = thr
    x = rng.standard_normal((4, 12, 15, cin))
    mine = nn.Conv2D(cin, cout, k, rng, stride=s, pad=p)
    mine.params['b'][...] = rng.standard_normal(cout)
    tb = Convolution(mine.params['W'].transpose(3, 2, 0, 1).copy(), mine.params['b'].copy(), s, p)
    y = mine.forward(x); yt = tb.forward(to_nchw(x))
    dy = rng.standard_normal(y.shape)
    dx = mine.backward(dy); dxt = tb.backward(to_nchw(dy))
    tag = f'Conv {k}x{k} s{s} cin{cin} ({"shift" if mine.shift else "im2col"})'
    check(tag + ' fwd', y, to_nhwc(yt))
    check(tag + ' dx', dx, to_nhwc(dxt))
    check(tag + ' dW', mine.grads['W'], tb.dW.transpose(2, 3, 1, 0))
    check(tag + ' db', mine.grads['b'], tb.db)

x = rng.standard_normal((4, 12, 15, 6))
mp, tp = nn.MaxPool2(), Pooling(2, 2, stride=2)
y = mp.forward(x); dy = rng.standard_normal(y.shape)
check('MaxPool2 fwd', y, to_nhwc(tp.forward(to_nchw(x))))
check('MaxPool2 dx', mp.backward(dy), to_nhwc(tp.backward(to_nchw(dy))))

x = rng.standard_normal((8, 10))
af = nn.Affine(10, 5, rng); af.params['b'][...] = rng.standard_normal(5)
ta = TAffine(af.params['W'].copy(), af.params['b'].copy())
y = af.forward(x); dy = rng.standard_normal(y.shape)
check('Affine fwd', y, ta.forward(x)); check('Affine dx', af.backward(dy), ta.backward(dy))
check('Affine dW', af.grads['W'], ta.dW)
r, tr = nn.ReLU(), Relu()
check('ReLU fwd', r.forward(x), tr.forward(x.copy())); check('ReLU dx', r.backward(x), tr.backward(x.copy()))

for shape in [(16, 7), (4, 5, 6, 7)]:
    x = rng.standard_normal(shape) * 3 + 1
    bn = nn.BatchNorm(7, eps=10e-7)
    bn.params['gamma'][...] = rng.standard_normal(7); bn.params['beta'][...] = rng.standard_normal(7)
    tb = BatchNormalization(bn.params['gamma'].copy(), bn.params['beta'].copy())
    y = bn.forward(x); dy = rng.standard_normal(y.shape)
    yt = tb.forward(x.reshape(-1, 7)).reshape(shape)
    dx = bn.backward(dy); dxt = tb.backward(dy.reshape(-1, 7)).reshape(shape)
    tag = f'BatchNorm {len(shape)}D'
    check(tag + ' fwd', y, yt, 1e-7); check(tag + ' dx', dx, dxt, 1e-7)
    check(tag + ' dgamma', bn.grads['gamma'], tb.dgamma, 1e-7)

# Adam: 같은 기울기 열로 20번 갱신
af = nn.Affine(10, 5, rng)
p_t = {'W': af.params['W'].copy(), 'b': af.params['b'].copy()}
opt, topt = nn.Adam(af, lr=1e-2, eps=1e-7), TAdam(lr=1e-2)
for _ in range(20):
    g = {'W': rng.standard_normal((10, 5)), 'b': rng.standard_normal(5)}
    af.grads.update({k: v.copy() for k, v in g.items()})
    opt.step(); topt.update(p_t, g)
# 교재는 eps를 편향 보정 전 v에 더하므로 아주 작은 차이가 있다
check('Adam 20 steps W', af.params['W'], p_t['W'], 1e-4)
print('ALL OK' if ok else 'SOME FAILED')
