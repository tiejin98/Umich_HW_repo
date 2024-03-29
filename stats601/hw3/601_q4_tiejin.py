import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS


dist_matrix = np.array([[0,587,1212,701,1936,604,748,2139,2182,543],
          [587,0,920,940,1745,1188,713,1858,1737,597],
          [1212,920,0,879,831,1726,1631,949,1021,1494],
          [701,940,879,0,1374,968,1420,1645,1891,1220],
          [1936,1745,831,1374,0,2339,2451,347,959,2300],
          [604,1188,1726,968,2339,0,1092,2594,2734,923],
          [748,713,1631,1420,2451,1092,0,2571,2408,205],
          [2139,1858,949,1645,347,2594,2571,0,678,2442],
          [2182,1737,1021,1891,959,2734,2408,678,0,2329],
          [543,597,1494,1220,2300,923,205,2442,2329,0]])

evals, evecs = np.linalg.eigh(dist_matrix)
res = evecs[:,-2:] * np.sqrt(abs(evals[-2:]))
place = ["ATLANTA","CHICAGO","DENVER","HOUSTON","LOS ANGELES","MIAMI","NEW YORK",
         "SAN FRANCISCO","SEATTLE","WASHINGTON D.C"]

model = MDS(2,dissimilarity="precomputed")
res = model.fit_transform(dist_matrix)


plt.scatter(res[:,0],res[:,1])
for i in range(res.shape[0]):
    plt.annotate(place[i],xy=(res[i,0],res[i,1]),xytext =(res[i,0]+0.1,res[i,1]+0.1))
plt.show()
