| 실험 | 프로토콜 | 모델 | 파라미터 | fold별 테스트 오차(°) | 평균(°) | 완료 fold |
|---|---|---|---|---|---|---|
| mlp | 5fold | mlp  | 1,240,834 | f0:5.84 / f1:7.34 / f2:7.02 / f3:8.01 / f4:6.38 | 6.92 | 5/5 |
| lenet | 5fold | lenet  | 1,827,076 | f0:5.65 | 5.65 | 1/5 |
| res_noaug | 5fold | resnet {"stem_pool":true} | 2,007,266 | f0:5.46 | 5.46 | 1/5 |
| plain_noaug | 5fold | resnet {"stem_pool":true,"skip":false} | 1,996,642 | f0:5.49 | 5.49 | 1/5 |
| r8_base | 5fold | resnet {"stem_pool":true} | 2,007,266 | f0:5.43 / f1:6.93 / f2:6.84 / f3:6.88 / f4:6.25 | 6.47 | 5/5 |
| r8_aug | 5fold | resnet {"stem_pool":true} | 2,007,266 | f0:5.21 / f1:6.49 / f2:6.84 / f3:6.48 / f4:6.34 | 6.27 | 5/5 |
| r8_augreg | 5fold | resnet {"stem_pool":true,"dropout":0.5} | 2,007,266 | f0:5.31 / f1:6.70 / f2:7.15 / f3:6.18 / f4:5.92 | 6.25 | 5/5 |
| L_aug | lopo | resnet {"stem_pool":true} | 2,007,266 | f0:4.31 / f1:5.20 / f2:6.12 / f3:6.50 / f4:7.04 / f5:6.31 / f6:6.18 / f7:8.05 / f8:6.35 / f9:7.09 / f10:5.81 / f11:5.88 / f12:5.54 / f13:7.15 / f14:6.04 | 6.24 | 15/15 |
| L_augreg | lopo | resnet {"stem_pool":true,"dropout":0.5} | 2,007,266 | f0:4.80 / f1:5.44 / f2:5.71 / f3:6.54 / f4:8.20 / f5:6.06 / f6:6.70 / f7:7.77 / f8:6.46 / f9:7.42 / f10:5.68 / f11:5.13 / f12:5.57 / f13:7.17 / f14:5.51 | 6.28 | 15/15 |
| L_flip | lopo | resnet {"stem_pool":true} | 2,007,266 | f0:4.60 / f1:6.78 / f2:5.83 / f3:6.79 / f4:6.95 / f5:6.06 / f6:5.87 / f7:7.24 / f8:6.49 / f9:7.69 / f10:5.51 / f11:5.68 / f12:5.68 / f13:7.40 / f14:5.46 | 6.27 | 15/15 |
| s_nopose | 5fold | resnet {"stem_pool":true,"use_pose":false} | 2,006,754 | f1:6.71 | 6.71 | 1/5 |
| s_flip | 5fold | resnet {"stem_pool":true} | 2,007,266 | f1:6.73 | 6.73 | 1/5 |
| s_imgnorm | 5fold | resnet {"stem_pool":true} | 2,007,266 | f1:6.67 | 6.67 | 1/5 |
| s_gap | 5fold | resnet {"stem_pool":true,"gap":true} | 729,314 | f1:6.18 | 6.18 | 1/5 |
| s_mse | 5fold | resnet {"stem_pool":true} | 2,007,266 | f1:6.82 | 6.82 | 1/5 |
| s_wide | 5fold | resnet {"stem_pool":true,"ch":[48,96,192]} | 3,529,170 | f1:6.67 | 6.67 | 1/5 |
| s_strongaug | 5fold | resnet {"stem_pool":true} | 2,007,266 | f1:7.02 | 7.02 | 1/5 |
| s_lr3e3 | 5fold | resnet {"stem_pool":true} | 2,007,266 | f1:6.53 | 6.53 | 1/5 |
| s_ep16 | 5fold | resnet {"stem_pool":true} | 2,007,266 | f1:6.51 | 6.51 | 1/5 |
| L_gap | lopo | resnet {"stem_pool":true,"gap":true} | 729,314 | f0:4.18 / f1:4.85 / f2:6.18 / f3:5.64 / f4:6.79 / f5:6.21 / f6:5.63 / f7:6.98 / f8:6.09 / f9:6.73 / f10:5.37 / f11:5.93 / f12:5.09 / f13:6.54 / f14:5.76 | 5.87 | 15/15 |
| t_gap_s1 | 5fold | resnet {"stem_pool":true,"gap":true} | 729,314 | f1:6.50 | 6.50 | 1/5 |
| t_gap_deep | 5fold | resnet {"stem_pool":true,"gap":true,"blocks":[3,3,3]} | 1,117,282 | f1:5.85 | 5.85 | 1/5 |
| t_gap_ep16 | 5fold | resnet {"stem_pool":true,"gap":true} | 729,314 | f1:6.09 | 6.09 | 1/5 |
| t_gap_flip | 5fold | resnet {"stem_pool":true,"gap":true} | 729,314 | f1:6.12 | 6.12 | 1/5 |
| t_gap_wide | 5fold | resnet {"stem_pool":true,"gap":true,"ch":[48,96,192]} | 1,612,242 | f1:6.28 | 6.28 | 1/5 |
| t_gap_fullres | 5fold | resnet {"stem_pool":false,"gap":true} | 729,314 | f1:5.74 | 5.74 | 1/5 |
| L_gapfull | lopo | resnet {"stem_pool":false,"gap":true} | 729,314 | f0:3.94 / f1:4.75 / f2:5.70 / f3:5.62 / f4:6.41 / f5:6.22 / f6:5.55 / f7:6.27 / f8:6.07 / f9:6.30 / f10:5.86 / f11:6.37 / f12:5.36 / f13:6.34 / f14:6.19 | 5.80 | 15/15 |

## LOPO 최종 (15-fold leave-one-person-out, 45,000장 평균)

| 설정 | 파라미터 | 평균 각도 오차(°) |
|---|---|---|
| Part 1 GazeNet (LeNet, 3에폭) | 1.8M | 6.65 |
| 증강 ResNet, flatten 헤드 (L_aug) | 2.0M | 6.24 |
| 증강 ResNet, GAP 헤드 (L_gap) | 0.73M | 5.87 |
| **증강 ResNet, GAP 헤드, 첫 풀링 제거 (L_gapfull)** | 0.73M | **5.80** |
| L_gapfull + 좌우 반전 TTA | 0.73M | 5.76 |
| L_gapfull + L_gap 앙상블 | 2 x 0.73M | 5.67 |
| L_aug + L_augreg + L_flip 앙상블 + TTA | 3 x 2.0M | 5.89 |

사람 단위 Wilcoxon 부호순위 검정 (15쌍): L_gapfull vs Part 1: 12/15 개선, p = 0.007 / vs L_aug: 12/15 개선, p = 0.005 / vs L_gap: 10/15 개선, p = 0.42 (유의하지 않음)
