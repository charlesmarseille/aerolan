import numpy as np
import matplotlib.pyplot as plt
from glob import glob

%matplotlib

fnames = glob('illum_santa_results/*.bin')

files = np.array([np.fromfile(fname) for fname in fnames])

sums = np.sum(files, axis=0)

plt.plot(sums)