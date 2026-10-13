# coding: utf-8
"""Download and extract the original MPIIGaze dataset (~2.1GB)

  python mpiigaze/download.py   ->  creates mpiigaze/data/MPIIGaze/
"""
import os
import tarfile
import urllib.request

URL = 'https://datasets.d2.mpi-inf.mpg.de/MPIIGaze/MPIIGaze.tar.gz'
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def _progress(block_num, block_size, total_size):
    done = block_num * block_size
    print('\r  %.1f / %.1f MB' % (done / 2**20, total_size / 2**20), end='', flush=True)


def download_mpiigaze():
    os.makedirs(DATA_DIR, exist_ok=True)
    tar_path = os.path.join(DATA_DIR, 'MPIIGaze.tar.gz')
    if not os.path.exists(os.path.join(DATA_DIR, 'MPIIGaze')):
        if not os.path.exists(tar_path):
            print('Downloading ' + URL)
            urllib.request.urlretrieve(URL, tar_path, _progress)
            print()
        print('Extracting ...')
        with tarfile.open(tar_path) as tar:
            tar.extractall(DATA_DIR)
    print('Done: ' + os.path.abspath(os.path.join(DATA_DIR, 'MPIIGaze')))


if __name__ == '__main__':
    download_mpiigaze()
