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
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED,
                     'ytick.color': MUTED, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'axes.axisbelow': True,
                     'savefig.dpi': 300, 'savefig.bbox': 'tight'})

PART1 = np.array([5.20, 7.36, 5.27, 7.18, 7.21, 7.11, 6.34, 8.06, 6.87, 8.57, 6.16, 5.51, 6.51, 6.87, 5.48])
rows = list(csv.DictReader(open(os.path.join(HERE, 'results.csv'))))


def lopo(prefix):
    d = {int(r['fold']): float(r['test_err']) for r in rows if r['name'].startswith(prefix + '_p')}
    return np.array([d[k] for k in range(15)])


from scipy import stats
_, G, _, P = load_data('none')

# 1. 모델별 LOPO 오차: 사람별 값(회색 선으로 짝지음) + 평균
models_ = [('Part 1\nLeNet', PART1), ('ResNet\n+ aug', lopo('L_aug')),
           ('+ GAP\nhead', lopo('L_gap')), ('+ full-res\n(final)', lopo('L_gapfull'))]
V = np.array([m[1] for m in models_])           # (4, 15)
fig, ax = plt.subplots(figsize=(3.35, 2.6))
x = np.arange(len(models_))
better = V[-1] < V[0]
for k in range(15):
    ax.plot(x, V[:, k], color=GRID, lw=1.2, zorder=1)
for i in range(len(models_)):
    if i == len(models_) - 1:   # 최종 모델: Part 1보다 낮으면 파랑, 높으면 주황 (주황을 위에 그려 가려지지 않게)
        ax.scatter(np.full(better.sum(), x[i]), V[i][better], s=10, color=C[0], zorder=4, edgecolor='white', linewidth=0.6)
        ax.scatter(np.full((~better).sum(), x[i]), V[i][~better], s=10, color=C[1], zorder=5, edgecolor='white', linewidth=0.6)
    else:
        ax.scatter(np.full(15, x[i]), V[i], s=10, color=MUTED, zorder=2, edgecolor='white', linewidth=0.6)
    ax.plot([x[i] - 0.22, x[i] + 0.22], [V[i].mean()] * 2, color=INK, lw=2, zorder=3)
    ax.text(x[i] + 0.26, V[i].mean(), f'{V[i].mean():.2f}°', va='center', fontsize=7.5, color=INK, fontweight='bold')
ax.axhline(6.0, color=MUTED, lw=1, ls='--', zorder=0)
p = stats.wilcoxon(V[-1], V[0]).pvalue
ax.text(x[-1] + 0.35, V.max() + 0.1, f'final vs Part 1: {better.sum()}/15 lower\n(Wilcoxon p = {p:.3f})', ha='right', fontsize=7, color=INK)
ax.set_xticks(x, [m[0] for m in models_]); ax.set_xlim(-0.4, x[-1] + 0.75)
ax.set_ylabel('LOPO angular error (°)'); ax.grid(axis='x', visible=False)
ax.set_ylim(3.5, V.max() + 1.0)
fig.savefig(os.path.join(OUT, 'lopo_paired.png')); plt.close(fig)

# 2. 학습 곡선 (LOPO 15 fold 평균), 범례 대신 직접 라벨
fig, ax = plt.subplots(figsize=(3.35, 2.4))
for i, (pre, lab) in enumerate([('L_aug', 'flatten'), ('L_gapfull', 'GAP')]):
    H = [json.load(open(os.path.join(HERE, 'runs', f'{pre}_p{k:02d}.json')))['hist'] for k in range(15)]
    tr = np.mean([[h['train_err'] for h in hh] for hh in H], 0)
    te = np.mean([[h['test_err'] for h in hh] for hh in H], 0)
    ep = np.arange(1, len(tr) + 1)
    ax.plot(ep, te, color=C[i], lw=1.6, marker='o', ms=3)
    ax.plot(ep, tr, color=C[i], lw=1.6, ls='--')
    ax.text(ep[-1] + 0.2, te[-1], f'{lab} test {te[-1]:.2f}°', color=C[i], va='center', fontsize=7)
    ax.text(ep[-1] + 0.2, tr[-1], f'{lab} train {tr[-1]:.2f}°', color=C[i], va='center', fontsize=7)
    if pre == 'L_gapfull':
        ax.fill_between(ep[1:], tr[1:], te[1:], color=C[i], alpha=0.08)
        ax.text(5.6, (tr[5] + te[5]) / 2, f'gap {te[-1] - tr[-1]:.1f}°', color=C[i], fontsize=7)
ax.set_xlabel('Epoch'); ax.set_ylabel('Mean angular error (°)'); ax.set_ylim(2.5, 9)
ax.set_xlim(0.7, 12.0); ax.set_xticks(range(1, 9))
fig.savefig(os.path.join(OUT, 'learning_curves.png')); plt.close(fig)

# 3. 사람별 고정 편향: (a) 평균 잔차 (b) 편향 제거 전후 오차 (분석용, 테스트 라벨 사용)
bias, e0, e1 = [], [], []
for k in range(15):
    _, te, _ = split(P, k, 'lopo')
    p_ = np.load(os.path.join(HERE, 'runs', f'L_gapfull_p{k:02d}_pred.npy'))
    bias.append(np.degrees(p_ - G[te]).mean(0))
    e0.append(nn.angular_error_deg(p_, G[te]).mean())
    e1.append(nn.angular_error_deg(p_ - (p_ - G[te]).mean(0), G[te]).mean())
bias, e0, e1 = np.array(bias), np.array(e0), np.array(e1)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(6.9, 2.9), gridspec_kw={'width_ratios': [1, 1.15]})
a1.scatter(bias[:, 1], bias[:, 0], s=22, color=C[0], edgecolor='white', linewidth=1.2, zorder=3)
for k in range(15):
    a1.annotate(f'p{k:02d}', (bias[k, 1], bias[k, 0]), textcoords='offset points',
                xytext=(3, -8) if k == 0 else (3, 2), fontsize=6.5, color=MUTED)   # p00은 p12와 겹치지 않게 아래에
a1.axhline(0, color=MUTED, lw=0.8); a1.axvline(0, color=MUTED, lw=0.8)
lim = np.abs(bias).max() + 1
a1.set_xlim(-lim, lim); a1.set_ylim(-lim, lim); a1.set_aspect('equal')
a1.set_xlabel('Mean yaw residual (°)'); a1.set_ylabel('Mean pitch residual (°)')
a1.set_title(f'(a) per-person offset, mean |offset| = {np.linalg.norm(bias, axis=1).mean():.1f}°', loc='left', fontsize=8, color=INK)
o = np.argsort(e0)
y = np.arange(15)
a2.hlines(y, e1[o], e0[o], color=GRID, lw=2)
a2.scatter(e0[o], y, s=16, color=C[1], zorder=3, label=f'final model ({e0.mean():.2f}°)')
a2.scatter(e1[o], y, s=16, color=C[0], zorder=3, label=f'offset removed ({e1.mean():.2f}°)')
a2.set_yticks(y, [f'p{k:02d}' for k in o], fontsize=6.5)
a2.set_xlabel('LOPO angular error (°)'); a2.grid(axis='y', visible=False)
a2.legend(frameon=False, fontsize=7, loc='upper center', bbox_to_anchor=(0.5, -0.17), ncol=2)
a2.set_title('(b) error before / after removing each offset', loc='left', fontsize=8, color=INK)
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'person_bias.png')); plt.close(fig)

print(f'per-person bias |(pitch,yaw)| mean = {np.linalg.norm(bias, axis=1).mean():.2f} deg')
print(f'error {e0.mean():.2f} -> {e1.mean():.2f} deg if each person\'s mean offset were removed (oracle)')
print('figures written to', OUT)
