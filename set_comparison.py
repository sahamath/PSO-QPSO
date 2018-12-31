import json
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

with open("qpso_trappist_crs_1.json", "r") as f:
    d1 = json.loads(f.read())

with open("qpso_trappist_modified_crs_2.json","r") as f:
    d2 = json.loads(f.read())

planets = list(set(d1.keys()).intersection(set(d2.keys())))

distances = []

for planet in planets:
    x = []
    y = []
    for front in d1[planet]["pareto_front"]:
        x.append([front["objectives"]["interior_score"],front["objectives"]["surface_score"]])

    for front in d2[planet]["pareto_front"]:
        y.append([front["objectives"]["interior_score"],front["objectives"]["surface_score"]])
    dist = cdist(np.array(x),np.array(y))
    distances.append(np.min(dist))

df = pd.DataFrame({"planet" : planets, "Hausdorff distance": distances})
sns.barplot(x = "planet", y = "Hausdorff distance", data = df)
plt.rcParams.update({'font.size': 25})
plt.show()

df.to_csv("trappist_set_distances_2.csv",index = False)



