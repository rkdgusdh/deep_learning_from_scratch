# coding: utf-8
"""Train GazeNet (leave-one-person-out)

Examples:
  python mpiigaze/train.py --test-person p00 --epochs 10
  python mpiigaze/train.py --all-folds --epochs 5
"""
import sys, os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir))
import argparse
import json
import numpy as np
from mpiigaze.dataset import load_mpiigaze, PERSONS
from mpiigaze.gaze_net import GazeNet
from mpiigaze.trainer import RegressionTrainer
from mpiigaze.layers_ext import angular_error

RESULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')


def run_fold(test_person, args):
    (x_tr, p_tr, t_tr), (x_te, p_te, t_te) = load_mpiigaze(test_person=test_person,
                                                           equalize=args.equalize)
    if args.train_limit:
        idx = np.random.choice(x_tr.shape[0], args.train_limit, replace=False)
        x_tr, p_tr, t_tr = x_tr[idx], p_tr[idx], t_tr[idx]

    # baseline: always predict the mean training gaze
    baseline = float(np.mean(angular_error(np.tile(t_tr.mean(axis=0), (len(t_te), 1)), t_te)))
    print("[%s] train %d / test %d, mean-gaze baseline: %.2f deg"
          % (test_person, len(x_tr), len(x_te), baseline))

    network = GazeNet(pose_size=0 if args.no_pose else 2)
    if args.no_pose:
        p_tr = p_te = None
    trainer = RegressionTrainer(network, x_tr, p_tr, t_tr, x_te, p_te, t_te,
                                epochs=args.epochs, mini_batch_size=args.batch_size,
                                optimizer=args.optimizer, optimizer_param={'lr': args.lr},
                                lr_decay=args.lr_decay, evaluate_sample_num_per_epoch=3000)
    test_err = trainer.train()

    os.makedirs(RESULT_DIR, exist_ok=True)
    network.save_params(os.path.join(RESULT_DIR, 'gazenet_%s.pkl' % test_person))
    return {'test_person': test_person, 'baseline_deg': baseline, 'test_err_deg': test_err,
            'train_loss': [float(v) for v in trainer.train_loss_list],
            'train_err': trainer.train_err_list, 'test_err': trainer.test_err_list}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-person', default='p00', choices=PERSONS)
    parser.add_argument('--all-folds', action='store_true', help='run leave-one-person-out over all 15 participants')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=100)
    parser.add_argument('--optimizer', default='Adam')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr-decay', type=float, default=0.7, help='multiply the learning rate by this after every epoch')
    parser.add_argument('--train-limit', type=int, default=0, help='cap the number of training samples (quick experiments)')
    parser.add_argument('--no-pose', action='store_true', help='drop the head pose input (ablation)')
    parser.add_argument('--equalize', action='store_true', help='re-apply histogram equalization (patches are already equalized)')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)

    persons = PERSONS if args.all_folds else [args.test_person]
    results = [run_fold(p, args) for p in persons]

    tag = 'all' if args.all_folds else args.test_person
    with open(os.path.join(RESULT_DIR, 'result_%s.json' % tag), 'w') as f:
        json.dump({'args': vars(args), 'folds': results}, f, indent=1)

    print("\n%-6s %10s %10s" % ('person', 'baseline', 'GazeNet'))
    for r in results:
        print("%-6s %9.2f deg %9.2f deg" % (r['test_person'], r['baseline_deg'], r['test_err_deg']))
    print("%-6s %9.2f deg %9.2f deg" % ('mean', np.mean([r['baseline_deg'] for r in results]),
                                  np.mean([r['test_err_deg'] for r in results])))


if __name__ == '__main__':
    main()
