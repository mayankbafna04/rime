import numpy as np
import random
from tqdm import trange

class RimeOptimizer:
    def __init__(self, objective_func, search_space, population_size=10, max_iterations=20):
        self.objective_func = objective_func
        self.search_space = search_space
        self.pop_size = population_size
        self.max_iter = max_iterations
        self.dims = len(search_space)
        self.lb = np.array([s[0] for s in search_space])
        self.ub = np.array([s[1] for s in search_space])
        self.w = 5.0
        self.beta = 0.1

        print("Initializing RIME population...")
        self.agents = self.lb + np.random.rand(self.pop_size, self.dims) * (self.ub - self.lb)
        self.agents = np.round(self.agents).astype(int)
        
        self.fitness = np.array([self.objective_func(agent) for agent in self.agents])
        self.best_agent = self.agents[np.argmin(self.fitness)]
        self.best_fitness = np.min(self.fitness)

    def optimize(self):
        print("Starting RIME optimization...")
        iterator = trange(self.max_iter, desc=f"RIME Optimization | Best BER: {self.best_fitness:.4f}")
        for t in iterator:
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                self.best_fitness = self.fitness[best_idx]
                self.best_agent = self.agents[best_idx].copy()

            E = np.exp(-t / self.max_iter)
            rime_factor = self.w * np.cos(np.pi * t / (2 * self.max_iter)) * (1 - E)

            for i in range(self.pop_size):
                r1 = np.random.rand()
                if r1 < rime_factor: # Soft-rime search
                    self.agents[i] = self.best_agent + rime_factor * (np.random.rand(self.dims) * 2 - 1)
                
                r2 = np.random.rand()
                if r2 < self.beta: # Hard-rime puncture
                    j = random.randint(0, self.pop_size - 1)
                    if self.fitness[i] > self.fitness[j]:
                         self.agents[i] = self.best_agent - self.agents[j] + np.random.rand() * (self.best_agent - self.agents[i])
                    else:
                         self.agents[i] = self.best_agent - self.agents[i] + np.random.rand() * (self.best_agent - self.agents[j])
                
                self.agents[i] = np.clip(self.agents[i], self.lb, self.ub)
                self.agents[i] = np.round(self.agents[i]).astype(int)
                self.fitness[i] = self.objective_func(self.agents[i])
            
            iterator.set_description(f"RIME Optimization | Best BER: {self.best_fitness:.4f}")

        print(f"\nRIME optimization finished. Final best fitness (BER): {self.best_fitness:.4f}")
        return self.best_agent, self.best_fitness