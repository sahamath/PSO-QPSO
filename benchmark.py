
from jmetal.util import FrontPlot
from jmetal.algorithm import SMPSO, MOQPSO, NSGAII
from jmetal.component import CrowdingDistanceArchive,RankingAndCrowdingDistanceComparator
from jmetal.problem import Kursawe,Fonseca,Schaffer, ZDT1,ZDT2, ZDT3, ZDT4,Viennet2
from jmetal.operator import Polynomial,SBX, BinaryTournamentSelection, Uniform
from objective_functions import get_crs_objective, get_drs_objective, interior_score, surface_score
from jmetal.component.quality_indicator import HyperVolume
from matplotlib import pyplot as plt 
import pandas as pd
import numpy as np

problem_names = ["kurswae", "fonseca", "zdt1" , "zdt2","zdt3","zdt4"]
problems = [Viennet2()]
sm_hv = []
q_hv = []

for i,problem in enumerate(problems):
    print("solving for {}".format(problem_names[i]))
    hv_comp = HyperVolume([5]*problem.number_of_objectives)
    smpso = SMPSO(
            problem=problem,
            swarm_size=100,
            max_evaluations=100000,
            mutation=Polynomial(probability=0.3, distribution_index=10),
            leaders=CrowdingDistanceArchive(100),
            reference_point=[5] * problem.number_of_objectives
        )
    smpso.run()
    print("SMPSO HV {}".format(hv_comp.compute(smpso.get_result())))
    print("SMPSO ITERATIONS {}".format(smpso.evaluations))
    moqpso = MOQPSO(
        problem=problem,
        swarm_size=100,
        max_evaluations=100000,
        mutation=Polynomial(probability=0.3, distribution_index=10),
        leaders=CrowdingDistanceArchive(500),
        reference_point=  [5] * problem.number_of_objectives
    )
    moqpso.run()
    print("MOQPSO hv {}".format(hv_comp.compute(moqpso.get_result())))
    print("MOQPSO ITERATIONS {}".format(moqpso.evaluations))
    sm_hv.append(smpso.current_hv)
    q_hv.append(moqpso.current_hv)
    nsga = NSGAII(
            problem=problem,
            population_size=100,
            max_evaluations=1000,
            mutation=Uniform(0.01),
            crossover=SBX(probability=1.0, distribution_index=20),
            selection=BinaryTournamentSelection(comparator=RankingAndCrowdingDistanceComparator())
        )
    nsga.run()
    print("NSGAII hv {}".format(hv_comp.compute(nsga.get_result())))
  



