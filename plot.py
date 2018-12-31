import numpy as np
import json
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

with open("qpso_earth_crs.json", "r") as f:
	data = json.loads(f.read())


trappist = filter(lambda x: "TRAPPIST" in x, data.keys())

for planet in list(data.keys()):
	interior_scores = []
	surface_scores = []
	for front in data[planet]["pareto_front"]:
		interior_scores.append(front["objectives"]["interior_score"])
		surface_scores.append(front["objectives"]["surface_score"])

	print(planet)

	plt.scatter(interior_scores , surface_scores , marker = "x")
	plt.xlabel("interior score")
	plt.ylabel("surface score")
	plt.title("Pareto front for {}".format(planet))
	plt.savefig("{}.png".format(planet.replace(" ","_")))
	plt.close()
