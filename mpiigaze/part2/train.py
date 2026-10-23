"""학습/평가 스크립트.

평가 프로토콜: 사람 단위 분할(person-independent). 15명을 3명씩 5개 fold로 나누고
fold k의 3명을 테스트, 나머지 12명을 학습에 쓴다. 테스트 세트는 학습 중 어떤 선택(조기 종료 등)에도
쓰지 않고, 마지막 에폭 모델로만 보고한다.

사용 예: python train.py --name resnet_f0 --model resnet --model-kw '{"stem_pool": true}' --fold 0
"""
import argparse, json, os, time, csv
import numpy as np
import cv2
import nn, models

HERE = os.path.dirname(os.path.abspath(__file__))


def load_data(prep):
    d = np.load(os.path.join(HERE, 'data', 'mpiigaze_eval.npz'))
    X = d['X']
    if prep == 'histeq':
        X = np.stack([cv2.equalizeHist(im) for im in X])
    elif prep == 'clahe':
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 5))
        X = np.stack([c.apply(im) for im in X])
    return X.astype(np.float32) / 255.0, d['gaze'], d['head'], d['pid']


def split(pid, fold, protocol='5fold'):
    """'5fold': 3명씩 5개 fold (빠른 비교용). 'lopo': leave-one-person-out, fold = 테스트할 사람 번호
    (Part 1과 원 논문의 표준 프로토콜)."""
    persons = np.arange(15)
    test_p = persons[fold:fold + 1] if protocol == 'lopo' else persons[fold * 3:(fold + 1) * 3]
    te = np.isin(pid, test_p)
    return np.where(~te)[0], np.where(te)[0], test_p


def normalize(x, mode, stats=None):
    """x: (N,H,W) in [0,1]. 'global': 학습셋 평균/표준편차, 'image': 이미지별 표준화."""
    if mode == 'image':
        m = x.mean(axis=(1, 2), keepdims=True)
        s = x.std(axis=(1, 2), keepdims=True) + 1e-3
        return (x - m) / s
    return (x - stats[0]) / stats[1]


def augment(x, g, h, rng, cfg):
    """x: (B,H,W) [0,1]. 이동, 밝기/대비, 감마, 노이즈, 좌우 반전(yaw 부호 반전)."""
    B, H, W = x.shape
    x = x.copy(); g = g.copy(); h = h.copy()
    s = cfg.get('shift', 0)
    if s:
        xp = np.pad(x, ((0, 0), (s, s), (s, s)), mode='edge')
        dy = rng.integers(0, 2 * s + 1, B); dx = rng.integers(0, 2 * s + 1, B)
        x = np.stack([xp[i, dy[i]:dy[i] + H, dx[i]:dx[i] + W] for i in range(B)])
    if cfg.get('gamma', 0):
        gm = np.exp(rng.uniform(-cfg['gamma'], cfg['gamma'], (B, 1, 1))).astype(np.float32)
        x = np.clip(x, 0, 1) ** gm
    if cfg.get('contrast', 0):
        c = rng.uniform(1 - cfg['contrast'], 1 + cfg['contrast'], (B, 1, 1)).astype(np.float32)
        m = x.mean(axis=(1, 2), keepdims=True)
        x = (x - m) * c + m
    if cfg.get('bright', 0):
        x = x + rng.uniform(-cfg['bright'], cfg['bright'], (B, 1, 1)).astype(np.float32)
    if cfg.get('noise', 0):
        x = x + rng.normal(0, cfg['noise'], x.shape).astype(np.float32)
    if cfg.get('flip', 0):
        f = rng.random(B) < cfg['flip']
        x[f] = x[f, :, ::-1]
        g[f, 1] *= -1; h[f, 1] *= -1
    return x, g, h


def predict(model, X, H, bs=256, tta_flip=False):
    nn.set_train(model, False)
    out = []
    for i in range(0, len(X), bs):
        xb = X[i:i + bs, :, :, None]
        p = model.forward(xb, H[i:i + bs])
        if tta_flip:
            pf = model.forward(xb[:, :, ::-1].copy(), H[i:i + bs] * [1, -1])
            p = (p + pf * [1, -1]) / 2
        out.append(p)
    nn.set_train(model, True)
    return np.concatenate(out).astype(np.float32)


def lr_at(step, total, cfg):
    warm = cfg['warmup_steps']
    if step < warm:
        return cfg['lr'] * (step + 1) / warm
    if cfg['sched'] == 'cosine':
        t = (step - warm) / max(1, total - warm)
        return cfg['lr_min'] + 0.5 * (cfg['lr'] - cfg['lr_min']) * (1 + np.cos(np.pi * t))
    if cfg['sched'] == 'step':
        e = step / cfg['steps_per_epoch']
        return cfg['lr'] * (0.1 ** sum(e >= m for m in cfg['milestones']))
    return cfg['lr']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--model', default='resnet')
    ap.add_argument('--model-kw', default='{}')
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--protocol', default='5fold', choices=['5fold', 'lopo'])
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--bs', type=int, default=128)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--lr-min', type=float, default=1e-5)
    ap.add_argument('--sched', default='cosine')
    ap.add_argument('--milestones', default='[]')
    ap.add_argument('--warmup', type=float, default=0.5, help='워밍업 에폭 수')
    ap.add_argument('--opt', default='adam')
    ap.add_argument('--wd', type=float, default=0.0)
    ap.add_argument('--loss', default='l1')
    ap.add_argument('--prep', default='none')
    ap.add_argument('--norm', default='global')
    ap.add_argument('--aug', default='{}')
    ap.add_argument('--tta', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--eval-every', type=int, default=1)
    ap.add_argument('--train-frac', type=float, default=1.0)
    ap.add_argument('--out', default=os.path.join(HERE, 'runs'))
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    X, G, Hp, P = load_data(a.prep)
    tr, te, test_p = split(P, a.fold, a.protocol)
    if a.train_frac < 1:
        tr = rng.choice(tr, int(len(tr) * a.train_frac), replace=False)
    stats = (X[tr].mean(), X[tr].std())
    aug = json.loads(a.aug)
    Xte = normalize(X[te], a.norm, stats)

    model = models.build(a.model, rng, **json.loads(a.model_kw))
    model.head.layers[-1].params['b'][...] = G[tr].mean(0)  # 출력 편향을 학습셋 평균 시선으로 초기화
    opt = nn.Adam(model, lr=a.lr, wd=a.wd) if a.opt == 'adam' else nn.SGD(model, lr=a.lr, wd=a.wd)
    loss_fn = {'mse': nn.MSELoss(), 'l1': nn.L1Loss(), 'smoothl1': nn.SmoothL1Loss(np.radians(2))}[a.loss]
    spe = len(tr) // a.bs
    sch = dict(lr=a.lr, lr_min=a.lr_min, sched=a.sched, warmup_steps=int(a.warmup * spe),
               steps_per_epoch=spe, milestones=json.loads(a.milestones))
    total = spe * a.epochs
    print(f'[{a.name}] params={nn.n_params(model):,} train={len(tr)} test={len(te)} persons={list(test_p)}', flush=True)

    hist, step, t0 = [], 0, time.time()
    for ep in range(a.epochs):
        perm = rng.permutation(tr)
        tl, terr = 0.0, []
        for b in range(spe):
            idx = perm[b * a.bs:(b + 1) * a.bs]
            xb, gb, hb = X[idx], G[idx], Hp[idx]
            if aug:
                xb, gb, hb = augment(xb, gb, hb, rng, aug)
            xb = normalize(xb, a.norm, stats)[:, :, :, None]
            opt.lr = lr_at(step, total, sch)
            pred = model.forward(xb, hb)
            tl += loss_fn(pred, gb)
            model.backward(loss_fn.backward())
            opt.step()
            step += 1
            terr.append(nn.angular_error_deg(pred, gb).mean())
        row = dict(epoch=ep + 1, loss=tl / spe, train_err=float(np.mean(terr)), lr=opt.lr, time=time.time() - t0)
        if (ep + 1) % a.eval_every == 0 or ep + 1 == a.epochs:
            pte = predict(model, Xte, Hp[te], tta_flip=False)
            row['test_err'] = float(nn.angular_error_deg(pte, G[te]).mean())
        hist.append(row)
        print(json.dumps(row), flush=True)

    pte = predict(model, Xte, Hp[te], tta_flip=a.tta)
    err = nn.angular_error_deg(pte, G[te])
    per_person = {int(p): float(err[P[te] == p].mean()) for p in test_p}
    np.save(os.path.join(a.out, f'{a.name}_pred.npy'), pte)
    np.savez(os.path.join(a.out, f'{a.name}_weights.npz'), **nn.state_dict(model))
    json.dump(dict(args=vars(a), hist=hist, per_person=per_person), open(os.path.join(a.out, f'{a.name}.json'), 'w'), indent=1)
    res = dict(name=a.name, model=a.model, model_kw=a.model_kw, protocol=a.protocol, fold=a.fold, epochs=a.epochs, lr=a.lr, wd=a.wd,
               loss=a.loss, prep=a.prep, norm=a.norm, aug=a.aug, tta=a.tta, seed=a.seed, params=nn.n_params(model),
               train_err=round(hist[-1]['train_err'], 3), test_err=round(float(err.mean()), 3),
               minutes=round((time.time() - t0) / 60, 1), per_person=json.dumps({k: round(v, 2) for k, v in per_person.items()}))
    path = os.path.join(HERE, 'results.csv')
    new = not os.path.exists(path)
    with open(path, 'a', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(res))
        if new:
            w.writeheader()
        w.writerow(res)
    print('RESULT', json.dumps(res), flush=True)


if __name__ == '__main__':
    main()
