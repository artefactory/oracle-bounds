import numpy as np
import copy

def analyze_simulations(simulations, test_length):
    simulations_analysis = {}
    for i, simulation in simulations.items():
        simulation_analysis = {}
        for test_size in range(1, test_length+1):
            best_empirical_risk = np.inf
            best_theo_risk = None
            for model_name, model_result in simulation.items():
                model_loss = copy.deepcopy(model_result.loss_array[:test_size])
                empirical_risk = model_loss.mean()
                if empirical_risk < best_empirical_risk:
                    best_empirical_risk = empirical_risk
                    best_theo_risk = model_result.theo_risk
            simulation_analysis[test_size] = best_theo_risk
        best_theo_risk = min(simulation_analysis.values())
        simulation_analysis = {test_size: theo_risk - best_theo_risk for test_size, theo_risk in simulation_analysis.items()}
        simulations_analysis[i] = simulation_analysis
    return simulations_analysis


def get_best_model_distribution(simulations_analysis, test_length):
    best_models = {}
    for test_size in range(1, test_length+1):
        best_models[test_size] = []
        for i, simulation in simulations_analysis.items():
            best_models[test_size].append(simulation[test_size])
        best_models[test_size].sort()
    return best_models
