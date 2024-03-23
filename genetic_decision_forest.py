import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import resample
from deap import base, creator, tools, algorithms
from scipy.stats import entropy

# Genetic Decision Forest classifier
class GeneticDecisionForestClassifier(BaseEstimator, ClassifierMixin):
    class Individual:
        def __init__(self, feature_mask, max_depth, fitness):
            self.feature_mask = feature_mask
            self.max_depth = max_depth
            self.fitness = fitness

    def __init__(self, population_size=50, max_generations=100, max_depth=None,
                 crossover_rate=0.8, mutation_rate=0.1, tournament_size=5,
                 feature_importance_threshold=0.1, min_samples_split=2,
                 min_samples_leaf=1, max_features='auto', bootstrap=True,
                 diversity_weight=0.5, niches=3, adaptive_diversity=True,
                 random_state=None):
        self.population_size = population_size
        self.max_generations = max_generations
        self.max_depth = max_depth
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.feature_importance_threshold = feature_importance_threshold
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.diversity_weight = diversity_weight
        self.niches = niches
        self.adaptive_diversity = adaptive_diversity
        self.random_state = random_state
        self.ensemble_ = []
        self.ensemble_preds_ = []

    def _create_individual(self):
        feature_mask = np.random.choice([True, False], size=self.n_features_)
        max_depth = int(np.random.randint(1, self.max_depth))
        fitness = creator.FitnessMax()
        individual = self.Individual(feature_mask=feature_mask, max_depth=max_depth, fitness=fitness)
        return individual
        
    def _evaluate_individual(self, individual, X, y):
        tree = DecisionTreeClassifier(max_depth=individual.max_depth,
                                      min_samples_split=self.min_samples_split,
                                      min_samples_leaf=self.min_samples_leaf,
                                      max_features=self.max_features,
                                      random_state=self.random_state)
        X_selected = X[:, individual.feature_mask]
        X_selected = np.squeeze(X_selected)
        if self.bootstrap:
            indices = resample(range(X_selected.shape[0]), replace=True, n_samples=X_selected.shape[0], random_state=self.random_state)
            X_bootstrap, y_bootstrap = X_selected[indices, :], y[indices]
            X_bootstrap = np.squeeze(X_bootstrap)
            tree.fit(X_bootstrap, y_bootstrap)
        else:
            tree.fit(X_selected, y)
        y_pred = tree.predict(X_selected)
        
        accuracy = accuracy_score(y, y_pred)
        if self.ensemble_preds_:
            diversity = 1 - np.mean([accuracy_score(y_pred, pred) for pred in self.ensemble_preds_])
        else:
            diversity = 0
        
        fitness = accuracy + self.diversity_weight * diversity
        return (fitness,)

    def _create_toolbox(self):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))

        toolbox = base.Toolbox()
        toolbox.register("individual", self._create_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)

        toolbox.register("evaluate", self._evaluate_individual, X=self.X_, y=self.y_)
        toolbox.register("mate", self._cx_uniform, indpb=0.5)
        toolbox.register("mutate", self._mut_flip_bit, indpb=0.1)
        toolbox.register("select", tools.selTournament, tournsize=self.tournament_size)

        return toolbox

    def _cx_uniform(self, ind1, ind2, indpb):
        for i in range(len(ind1.feature_mask)):
            if np.random.random() < indpb:
                ind1.feature_mask[i], ind2.feature_mask[i] = ind2.feature_mask[i], ind1.feature_mask[i]
        return ind1, ind2

    def _mut_flip_bit(self, individual, indpb):
        for i in range(len(individual.feature_mask)):
            if np.random.random() < indpb:
                individual.feature_mask[i] = not individual.feature_mask[i]
        return individual,

    def _evolve_population(self, population, toolbox):
        halloffame = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        self.generation_fitness_ = []

        if self.adaptive_diversity:
            self.diversity_weight = 1.0

        for generation in range(self.max_generations):
            offspring = algorithms.varAnd(population, toolbox, cxpb=self.crossover_rate, mutpb=self.mutation_rate)
            
            for ind in offspring:
                if len(ind.feature_mask) != self.n_features_:
                    ind.feature_mask = np.random.choice([True, False], size=self.n_features_)
            
            fits = toolbox.map(toolbox.evaluate, offspring)
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = fit

            population = toolbox.select(offspring, k=len(population))
            halloffame.update(population)

            if self.adaptive_diversity:
                self.diversity_weight = 1.0 - generation / self.max_generations

            record = stats.compile(population)
            self.generation_fitness_.append(record["avg"])

        return population, halloffame

    def _niche_population(self, population):
        niches = []
        for _ in range(self.niches):
            niche_population = tools.selBest(population, k=len(population) // self.niches)
            niches.append(niche_population)
        return niches

    def _prune_ensemble(self, niches):
        pruned_ensemble = []
        pruned_individuals = []
        for niche in niches:
            feature_importances = np.zeros(self.n_features_)
            for individual in niche:
                feature_importances[individual.feature_mask] += 1

            selected_features = feature_importances > self.feature_importance_threshold * len(niche)
            pruned_niche = []
            pruned_niche_individuals = []
            for individual in niche:
                if np.any(individual.feature_mask[selected_features]):
                    tree = DecisionTreeClassifier(max_depth=individual.max_depth,
                                                min_samples_split=self.min_samples_split,
                                                min_samples_leaf=self.min_samples_leaf,
                                                max_features=self.max_features,
                                                random_state=self.random_state)
                    pruned_niche.append(tree)
                    pruned_niche_individuals.append(individual)
            pruned_ensemble.extend(pruned_niche)
            pruned_individuals.extend(pruned_niche_individuals)

        return pruned_ensemble, pruned_individuals

    def fit(self, X, y):
        self.X_ = X
        self.y_ = y
        self.n_features_ = X.shape[1]

        toolbox = self._create_toolbox()
        population = toolbox.population(n=self.population_size)

        population, _ = self._evolve_population(population, toolbox)

        self.ensemble_preds_ = []
        for individual in population:
            tree = DecisionTreeClassifier(max_depth=individual.max_depth,
                                        min_samples_split=self.min_samples_split,
                                        min_samples_leaf=self.min_samples_leaf,
                                        max_features=self.max_features,
                                        random_state=self.random_state)
            X_selected = self.X_[:, individual.feature_mask]
            X_selected = np.squeeze(X_selected)
            tree.fit(X_selected, self.y_)
            y_pred = tree.predict(X_selected)
            self.ensemble_preds_.append(y_pred)

        niches = self._niche_population(population)
        self.ensemble_, pruned_individuals = self._prune_ensemble(niches)
        self.final_population_ = pruned_individuals

        for tree, individual in zip(self.ensemble_, pruned_individuals):
            X_selected = self.X_[:, individual.feature_mask]
            X_selected = np.squeeze(X_selected)
            tree.fit(X_selected, self.y_)

        self.n_trees_ = len(self.ensemble_)

        return self

    def predict(self, X):
        y_preds = []
        for tree in self.ensemble_:
            individual_mask = None
            for individual in self.final_population_:
                if individual.max_depth == tree.max_depth:
                    individual_mask = individual.feature_mask
                    break
            if individual_mask is not None:
                X_selected = X[:, individual_mask]
                X_selected = np.squeeze(X_selected)
                y_pred = tree.predict(X_selected)
                y_preds.append(y_pred)

        if len(y_preds) > 0:
            y_preds = np.array(y_preds).T
            y_pred = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=y_preds)
        else:
            y_pred = np.zeros(X.shape[0], dtype=int)

        return y_pred

# Implementation
if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score

    # Load the breast cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Create an instance of the Genetic Decision Forest classifier
    gdf = GeneticDecisionForestClassifier(population_size=50, max_generations=100, max_depth=200,
                                          crossover_rate=0.8, mutation_rate=0.1, tournament_size=5,
                                          feature_importance_threshold=0.1, min_samples_split=2,
                                          min_samples_leaf=1, max_features='sqrt', bootstrap=True,
                                          diversity_weight=0.9, niches=5, adaptive_diversity=True,
                                          random_state=42)

    # Train the classifier
    gdf.fit(X_train, y_train)

    # Print the mean fitness for each generation
    for generation, fitness in enumerate(gdf.generation_fitness_):
        print(f"Generation {generation}: Mean Fitness = {fitness}")

    # Print the number of trees in the ensemble
    print("Number of trees in the ensemble:", gdf.n_trees_)

    # Make predictions on the test set
    y_pred = gdf.predict(X_test)

    # Calculate the accuracy and F1 score
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)