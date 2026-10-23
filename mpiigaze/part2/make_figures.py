"""보고서용 그림을 figures/에 만든다. results.csv, runs/*.json, runs/*_pred.npy를 읽기만 한다."""
import os, csv, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nn
from train import load_data, split, HERE

OUT = os.path.join(HERE, 'figures')
os.makedirs(OUT, exist_ok=True)
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']   # 범주 색 (고정 순서)
INK, MUTED, GRID = '#1f1f1e', '#6b6a64', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
                     'savefig.dpi': 200, 'savefig.bbox': 'tight'})

PART1 = np.array([5.20, 7.36, 5.27, 7.18, 7.21, 7.11, 6.34, 8.06, 6.87, 8.57, 6.16, 5.51, 6.51, 6.87, 5.48])
rows = list(csv.DictReader(open(os.path.join(HERE, 'results.csv'))))


def lopo(prefix):
    d = {int(r['fold']): float(r['test_err']) for r in rows if r['name'].startswith(prefix + '_p')}
    return np.array([d[k] for k in range(15)])


# 1. 사람별 LOPO 오차
series = [('Part 1 GazeNet (LeNet)', PART1), ('ResNet + aug (flatten head)', lopo('L_aug')),
          ('ResNet + aug + GAP, full-res (final)', lopo('L_gapfull'))]
fig, ax = plt.subplots(figsize=(10, 3.6))
x = np.arange(15); w = 0.26
for i, (lab, v) in enumerate(series):
    ax.bar(x + (i - 1) * w, v, w - 0.03, color=C[i], label=f'{lab}: mean {v.mean():.2f}°')
ax.set_xticks(x, [f'p{k:02d}' for k in range(15)])
ax.set_ylabel('Mean angular error (°)')
ax.set_ylim(0, 9.5)
ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3, fontsize=8.5)
ax.set_title('Leave-one-person-out error per test person', loc='left', color=INK)
fig.savefig(os.path.join(OUT, 'lopo_per_person.png')); plt.close(fig)

# 2. 단계별 개선 (LOPO 평균)
steps = [('Part 1 GazeNet', PART1.mean()), ('ResNet + augmentation', lopo('L_aug').mean()),
         ('+ global average pooling head', lopo('L_gap').mean()), ('+ full-resolution input', lopo('L_gapfull').mean())]
fig, ax = plt.subplots(figsize=(6.5, 2.6))
y = np.arange(len(steps))[::-1]
ax.barh(y, [s[1] for s in steps], 0.6, color=[C[0]] * (len(steps) - 1) + [C[1]])
for yi, (lab, v) in zip(y, steps):
    ax.text(max(v, 6.0) + 0.05, yi, f'{v:.2f}°', va='center', color=INK)
ax.axvline(6.0, color=MUTED, lw=1, ls='--')
ax.text(6.0, y[0] + 0.45, ' 6°', color=MUTED, va='bottom')
ax.set_yticks(y, [s[0] for s in steps]); ax.set_xlim(0, 7.2)
ax.set_xlabel('LOPO mean angular error (°), single model'); ax.grid(axis='y', visible=False)
fig.savefig(os.path.join(OUT, 'lopo_steps.png')); plt.close(fig)

# 3. 학습 곡선 (LOPO 15 fold 평균)
fig, ax = plt.subplots(figsize=(5.5, 3.4))
for i, (pre, lab) in enumerate([('L_aug', 'ResNet + aug (flatten)'), ('L_gapfull', 'ResNet + aug + GAP, full-res')]):
    H = [json.load(open(os.path.join(HERE, 'runs', f'{pre}_p{k:02d}.json')))['hist'] for k in range(15)]
    tr = np.mean([[h['train_err'] for h in hh] for hh in H], 0)
    te = np.mean([[h['test_err'] for h in hh] for hh in H], 0)
    ep = np.arange(1, len(tr) + 1)
    ax.plot(ep, te, color=C[i], lw=2, marker='o', ms=4, label=f'{lab}: test')
    ax.plot(ep, tr, color=C[i], lw=2, ls='--', label=f'{lab}: train (augmented)')
ax.set_xlabel('Epoch'); ax.set_ylabel('Mean angular error (°)'); ax.set_ylim(2, 9)
ax.legend(frameon=False, fontsize=8, loc='upper center', bbox_to_anchor=(0.5, -0.18), ncol=1)
fig.savefig(os.path.join(OUT, 'learning_curves.png')); plt.close(fig)

# 4. 사람별 평균 잔차 (pitch, yaw) - 개인차(편향)가 오차의 큰 부분
_, G, _, P = load_data('none')
fig, ax = plt.subplots(figsize=(4.6, 4.2))
bias, tot = [], []
for k in range(15):
    _, te, _ = split(P, k, 'lopo')
    p = np.load(os.path.join(HERE, 'runs', f'L_gapfull_p{k:02d}_pred.npy'))
    r = np.degrees(p - G[te])
    b = r.mean(0); bias.append(b)
    ax.scatter(b[1], b[0], s=40, color=C[0], edgecolor='white', linewidth=1.5, zorder=3)
    ax.annotate(f'p{k:02d}', (b[1], b[0]), textcoords='offset points', xytext=(5, 3), fontsize=8, color=MUTED)
ax.axhline(0, color=MUTED, lw=0.8); ax.axvline(0, color=MUTED, lw=0.8)
lim = np.abs(np.array(bias)).max() + 1
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect('equal')
ax.set_xlabel('Mean yaw residual (°)'); ax.set_ylabel('Mean pitch residual (°)')
ax.set_title('Per-person systematic offset (final model)', loc='left', color=INK, fontsize=10)
fig.savefig(os.path.join(OUT, 'person_bias.png')); plt.close(fig)

# 편향을 사람별로 빼면 오차가 얼마나 줄어드는지 (분석용 수치, 테스트 라벨을 쓰므로 성능 주장이 아님)
e0, e1 = [], []
for k in range(15):
    _, te, _ = split(P, k, 'lopo')
    p = np.load(os.path.join(HERE, 'runs', f'L_gapfull_p{k:02d}_pred.npy'))
    e0.append(nn.angular_error_deg(p, G[te]).mean())
    e1.append(nn.angular_error_deg(p - (p - G[te]).mean(0), G[te]).mean())
print(f'per-person bias |(pitch,yaw)| mean = {np.linalg.norm(bias, axis=1).mean():.2f} deg')
print(f'error {np.mean(e0):.2f} -> {np.mean(e1):.2f} deg if each person\'s mean offset were removed (oracle)')
print('figures written to', OUT)
