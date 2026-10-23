"""MPIIGaze 평가 서브셋(참가자당 3000장, 총 45,000장)을 하나의 npz로 캐시한다.
오른쪽 눈은 좌우 반전하고 yaw 부호를 뒤집는다 (원 논문 프로토콜)."""
import os, numpy as np, scipy.io as sio, cv2

# 데이터셋 위치: 환경변수 MPIIGAZE_ROOT, 없으면 Part 1 download.py가 받는 mpiigaze/data/MPIIGaze
ROOT = os.environ.get('MPIIGAZE_ROOT', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'MPIIGaze'))
NORM = os.path.join(ROOT, 'Data', 'Normalized')
EVAL = os.path.join(ROOT, 'Evaluation Subset', 'sample list for eye image')

def gaze_to_angle(v):  # (N,3) -> (N,2) [pitch, yaw]
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    return np.stack([np.arcsin(-y), np.arctan2(-x, -z)], 1)

def pose_to_angle(v):
    out = []
    for r in v:
        z = cv2.Rodrigues(r)[0][:, 2]
        out.append([np.arcsin(z[1]), np.arctan2(z[0], z[2])])
    return np.array(out)

X, G, H, P = [], [], [], []
for pid in range(15):
    p = f'p{pid:02d}'
    cache = {}
    for line in open(os.path.join(EVAL, p + '.txt')):
        path, side = line.split()
        day, fname = path.split('/')
        if day not in cache:
            m = sio.loadmat(os.path.join(NORM, p, day + '.mat'))
            names = [f[0][0] for f in m['filenames']]
            cache[day] = (m['data'], {n: i for i, n in enumerate(names)})
        data, idx = cache[day]
        i = idx[fname]
        e = data[side][0, 0]
        img = e['image'][0, 0][i]
        g = gaze_to_angle(e['gaze'][0, 0][i:i+1])[0]
        h = pose_to_angle(e['pose'][0, 0][i:i+1])[0]
        if side == 'right':
            img = img[:, ::-1]; g = g * [1, -1]; h = h * [1, -1]
        X.append(img); G.append(g); H.append(h); P.append(pid)
    print(p, len(X))
os.makedirs(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data'), exist_ok=True)
np.savez_compressed(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'mpiigaze_eval.npz'),
                    X=np.array(X, np.uint8), gaze=np.array(G, np.float32),
                    head=np.array(H, np.float32), pid=np.array(P, np.int8))
