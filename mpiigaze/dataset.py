# coding: utf-8
"""MPIIGaze loader (in the style of the textbook dataset/mnist.py)

Uses the *normalized* eye images (36x60 grayscale) in mpiigaze/data/MPIIGaze/Data/Normalized.
Two caches are built on first use:

  mpiigaze/data/mpiigaze_eval.npz : the standard evaluation subset (15 persons x 3000 eyes) with images
  mpiigaze/data/mpiigaze_meta.npz : every normalized sample (~430k eyes), labels + image statistics only

Conventions
  - gaze / head pose are stored as (pitch, yaw) in radians
  - right eyes are mirrored horizontally (and yaw negated) so all samples look like left eyes,
    as done in the original paper (only in the evaluation subset; the meta table keeps both sides raw)
"""
import os
import glob
import numpy as np
import scipy.io

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
ROOT = os.path.join(DATA_DIR, 'MPIIGaze')
NORM_DIR = os.path.join(ROOT, 'Data', 'Normalized')
EVAL_LIST_DIR = os.path.join(ROOT, 'Evaluation Subset', 'sample list for eye image')
EVAL_CACHE = os.path.join(DATA_DIR, 'mpiigaze_eval.npz')
META_CACHE = os.path.join(DATA_DIR, 'mpiigaze_meta.npz')

PERSONS = ['p%02d' % i for i in range(15)]
IMG_SHAPE = (36, 60)


# ---------------------------------------------------------------- angle conversion
def gaze_to_pitchyaw(gaze):
    """(N, 3) gaze vectors -> (N, 2) [pitch, yaw]."""
    g = gaze / np.linalg.norm(gaze, axis=1, keepdims=True)
    return np.stack([np.arcsin(-g[:, 1]), np.arctan2(-g[:, 0], -g[:, 2])], axis=1)


def rodrigues(rvec):
    """(N, 3) rotation vectors -> (N, 3, 3) rotation matrices (NumPy version of cv2.Rodrigues)."""
    theta = np.linalg.norm(rvec, axis=1)
    k = rvec / np.maximum(theta, 1e-12)[:, None]
    K = np.zeros((len(rvec), 3, 3))
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    s, c = np.sin(theta)[:, None, None], np.cos(theta)[:, None, None]
    return np.eye(3)[None] + s * K + (1 - c) * (K @ K)


def pose_to_pitchyaw(pose):
    """(N, 3) head rotation vectors -> (N, 2) [pitch, yaw] of the head's z axis."""
    z = rodrigues(pose)[:, :, 2]
    return np.stack([np.arcsin(z[:, 1]), np.arctan2(z[:, 0], z[:, 2])], axis=1)


# ---------------------------------------------------------------- raw .mat reading
def _read_day(path):
    """Read one dayXX.mat -> filenames, {'left': (img, gaze, pose), 'right': (...)}."""
    mat = scipy.io.loadmat(path, struct_as_record=False, squeeze_me=True)
    filenames = np.atleast_1d(mat['filenames'])
    n = len(filenames)
    sides = {}
    for side in ('left', 'right'):
        s = getattr(mat['data'], side)
        sides[side] = (s.image.reshape(n, *IMG_SHAPE), s.gaze.reshape(n, 3), s.pose.reshape(n, 3))
    return list(filenames), sides


def _day_files(person):
    return sorted(glob.glob(os.path.join(NORM_DIR, person, 'day*.mat')))


def _build_eval_cache():
    imgs, gazes, poses, persons, days, sides = [], [], [], [], [], []
    for pi, person in enumerate(PERSONS):
        print('  building evaluation subset: ' + person)
        with open(os.path.join(EVAL_LIST_DIR, person + '.txt')) as f:
            items = [line.split() for line in f if line.strip()]
        cache = {}
        for rel, side in items:
            day, fname = rel.split('/')
            if day not in cache:
                filenames, data = _read_day(os.path.join(NORM_DIR, person, day + '.mat'))
                cache[day] = ({fn: i for i, fn in enumerate(filenames)}, data)
            index, data = cache[day]
            i = index[fname]
            img, gaze, pose = (a[i] for a in data[side])
            gaze_py = gaze_to_pitchyaw(gaze[None])[0]
            pose_py = pose_to_pitchyaw(pose[None])[0]
            if side == 'right':  # mirror right eyes so they look like left eyes
                img = img[:, ::-1]
                gaze_py[1] *= -1
                pose_py[1] *= -1
            imgs.append(img)
            gazes.append(gaze_py)
            poses.append(pose_py)
            persons.append(pi)
            days.append(int(day[3:]))
            sides.append(0 if side == 'left' else 1)
    np.savez_compressed(EVAL_CACHE, image=np.array(imgs, dtype=np.uint8),
                        gaze=np.array(gazes, dtype=np.float32), pose=np.array(poses, dtype=np.float32),
                        person=np.array(persons, dtype=np.int8), day=np.array(days, dtype=np.int16),
                        side=np.array(sides, dtype=np.int8))


def _build_meta_cache():
    cols = {k: [] for k in ('person', 'day', 'side', 'gaze', 'pose', 'img_mean', 'img_std')}
    mean_img = np.zeros(IMG_SHAPE)
    for pi, person in enumerate(PERSONS):
        print('  building metadata table: ' + person)
        for path in _day_files(person):
            day = int(os.path.basename(path)[3:5])
            _, data = _read_day(path)
            for si, side in enumerate(('left', 'right')):
                img, gaze, pose = data[side]
                n = len(img)
                f = img.reshape(n, -1).astype(np.float32)
                cols['person'].append(np.full(n, pi, np.int8))
                cols['day'].append(np.full(n, day, np.int16))
                cols['side'].append(np.full(n, si, np.int8))
                cols['gaze'].append(gaze_to_pitchyaw(gaze).astype(np.float32))
                cols['pose'].append(pose_to_pitchyaw(pose).astype(np.float32))
                cols['img_mean'].append(f.mean(axis=1))
                cols['img_std'].append(f.std(axis=1))
                mean_img += (img[:, :, ::-1] if side == 'right' else img).sum(axis=0)
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out['mean_image'] = mean_img / len(out['person'])
    np.savez_compressed(META_CACHE, **out)


# ---------------------------------------------------------------- public API
def _check_raw():
    if not os.path.exists(NORM_DIR):
        raise FileNotFoundError('MPIIGaze not found at %s - run `python mpiigaze/download.py` first'
                                % os.path.abspath(ROOT))


def load_eval_subset():
    """Evaluation subset as a dict of arrays (image uint8 (N,36,60), gaze, pose, person, day, side)."""
    if not os.path.exists(EVAL_CACHE):
        _check_raw()
        _build_eval_cache()
    with np.load(EVAL_CACHE) as f:
        return {k: f[k] for k in f.files}


def load_meta():
    """Per-sample metadata of the full normalized set (no images) + the mean eye image."""
    if not os.path.exists(META_CACHE):
        _check_raw()
        _build_meta_cache()
    with np.load(META_CACHE) as f:
        return {k: f[k] for k in f.files}


def equalize_hist(images):
    """Per-image histogram equalization for a uint8 batch (N, H, W) (NumPy version of cv2.equalizeHist)."""
    out = np.empty_like(images)
    for i, img in enumerate(images):
        hist = np.bincount(img.ravel(), minlength=256)
        cdf = hist.cumsum()
        cdf_min = cdf[cdf > 0][0]
        lut = np.round((cdf - cdf_min) / max(cdf[-1] - cdf_min, 1) * 255).clip(0, 255).astype(np.uint8)
        out[i] = lut[img]
    return out


def load_mpiigaze(test_person='p00', equalize=False):
    """Leave-one-person-out split of the evaluation subset.

    Returns
    -------
    (x_train, pose_train, t_train), (x_test, pose_test, t_test)
      x : float32 (N, 1, 36, 60) in [0, 1] (the patches are already histogram-equalized by the authors)
      pose, t : float32 (N, 2) (pitch, yaw) in radians
    """
    d = load_eval_subset()
    images = equalize_hist(d['image']) if equalize else d['image']
    test = d['person'] == PERSONS.index(test_person)
    train = ~test
    # convert each split separately so that only one float32 copy of the images exists
    to_x = lambda mask: (images[mask].astype(np.float32) / np.float32(255.0))[:, None]
    return (to_x(train), d['pose'][train], d['gaze'][train]), (to_x(test), d['pose'][test], d['gaze'][test])


if __name__ == '__main__':
    ev = load_eval_subset()
    print('evaluation subset:', {k: v.shape for k, v in ev.items()})
    meta = load_meta()
    print('full normalized set: %d eye samples' % len(meta['person']))
