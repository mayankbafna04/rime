import numpy as np
import random
from tqdm import trange


class EnhancedRimeOptimizer:
    """
    Enhanced RIME optimizer with adaptive parameters and improved exploration/exploitation.
    """
    
    def __init__(self, objective_func, search_space, population_size=30, max_iterations=50):
        self.objective_func = objective_func
        self.search_space = search_space
        self.pop_size = population_size
        self.max_iter = max_iterations
        self.dims = len(search_space)
        self.lb = np.array([s[0] for s in search_space])
        self.ub = np.array([s[1] for s in search_space])
        
        # Enhanced RIME parameters
        self.w_initial = 5.0
        self.w_final = 0.5
        self.beta_initial = 0.1
        self.beta_final = 0.3
        
        # Adaptive parameters
        self.stagnation_counter = 0
        self.best_history = []
        self.diversity_threshold = 0.1
        
        print("Initializing Enhanced RIME population...")
        
        # Initialize population with better diversity
        self.agents = self.initialize_population()
        
        # Evaluate initial population
        print("Evaluating initial population...")
        self.fitness = np.array([self.objective_func(agent) for agent in trange(
            len(self.agents), desc="Initial evaluation")])
        
        # Track best solution
        self.best_agent = self.agents[np.argmin(self.fitness)].copy()
        self.best_fitness = np.min(self.fitness)
        
        # Elite population for maintaining good solutions
        self.elite_size = max(2, int(0.1 * self.pop_size))
        self.update_elite()
    
    def initialize_population(self):
        """
        Initialize population with better diversity using multiple strategies.
        """
        agents = []
        
        # Random initialization (40% of population)
        n_random = int(0.4 * self.pop_size)
        for _ in range(n_random):
            agent = self.lb + np.random.rand(self.dims) * (self.ub - self.lb)
            agents.append(np.round(agent).astype(int))
        
        # Latin hypercube sampling for better coverage (30% of population)
        n_lhs = int(0.3 * self.pop_size)
        for i in range(n_lhs):
            agent = np.zeros(self.dims)
            for d in range(self.dims):
                segment = (self.ub[d] - self.lb[d]) / n_lhs
                agent[d] = self.lb[d] + (i + np.random.rand()) * segment
            agents.append(np.round(agent).astype(int))
        
        # Grid-based initialization for systematic coverage (30% of population)
        n_grid = self.pop_size - n_random - n_lhs
        for _ in range(n_grid):
            agent = np.zeros(self.dims)
            for d in range(self.dims):
                # Create grid points
                n_points = min(5, int(self.ub[d] - self.lb[d] + 1))
                grid_points = np.linspace(self.lb[d], self.ub[d], n_points)
                agent[d] = np.random.choice(grid_points)
            agents.append(np.round(agent).astype(int))
        
        return np.array(agents)
    
    def update_elite(self):
        """
        Maintain elite solutions for preserving good genes.
        """
        sorted_indices = np.argsort(self.fitness)
        self.elite_agents = self.agents[sorted_indices[:self.elite_size]].copy()
        self.elite_fitness = self.fitness[sorted_indices[:self.elite_size]].copy()
    
    def calculate_diversity(self):
        """
        Calculate population diversity to detect convergence.
        """
        mean_agent = np.mean(self.agents, axis=0)
        diversity = np.mean([np.linalg.norm(agent - mean_agent) for agent in self.agents])
        return diversity / (np.linalg.norm(self.ub - self.lb) + 1e-10)
    
    def adaptive_parameters(self, t):
        """
        Adapt RIME parameters based on optimization progress.
        """
        # Linear adaptation
        progress = t / self.max_iter
        
        # Adaptive w (exploration factor)
        self.w = self.w_initial * (1 - progress) + self.w_final * progress
        
        # Adaptive beta (exploitation factor)
        self.beta = self.beta_initial * (1 - progress) + self.beta_final * progress
        
        # Boost exploration if stagnating
        if self.stagnation_counter > 5:
            self.w *= 1.5
            self.beta *= 0.8
            self.stagnation_counter = 0
    
    def levy_flight(self, dim):
        """
        Levy flight for enhanced exploration.
        """
        beta = 1.5
        sigma = (np.math.gamma(1 + beta) * np.sin(np.pi * beta / 2) / 
                (np.math.gamma((1 + beta) / 2) * beta * 2**((beta - 1) / 2)))**(1 / beta)
        
        u = np.random.randn(dim) * sigma
        v = np.random.randn(dim)
        step = u / (np.abs(v)**(1 / beta))
        
        return step
    
    def crossover(self, parent1, parent2):
        """
        Crossover operation for genetic diversity.
        """
        child = np.zeros_like(parent1)
        for i in range(len(parent1)):
            if np.random.rand() < 0.5:
                child[i] = parent1[i]
            else:
                child[i] = parent2[i]
        return child
    
    def mutation(self, agent, mutation_rate=0.1):
        """
        Adaptive mutation for local search.
        """
        mutated = agent.copy()
        for i in range(len(agent)):
            if np.random.rand() < mutation_rate:
                # Gaussian mutation
                mutated[i] += np.random.randn() * (self.ub[i] - self.lb[i]) * 0.1
        return np.clip(np.round(mutated).astype(int), self.lb, self.ub)
    
    def optimize(self):
        """
        Main optimization loop with enhanced RIME algorithm.
        """
        print("Starting Enhanced RIME optimization...")
        iterator = trange(self.max_iter, desc=f"RIME | Best: {self.best_fitness:.4f}")
        
        for t in iterator:
            # Update adaptive parameters
            self.adaptive_parameters(t)
            
            # Calculate diversity
            diversity = self.calculate_diversity()
            
            # Update best solution
            best_idx = np.argmin(self.fitness)
            if self.fitness[best_idx] < self.best_fitness:
                improvement = self.best_fitness - self.fitness[best_idx]
                self.best_fitness = self.fitness[best_idx]
                self.best_agent = self.agents[best_idx].copy()
                self.stagnation_counter = 0
            else:
                self.stagnation_counter += 1
            
            # Store best fitness history
            self.best_history.append(self.best_fitness)
            
            # Environmental factor
            E = np.exp(-t / self.max_iter)
            rime_factor = self.w * np.cos(np.pi * t / (2 * self.max_iter)) * (1 - E)
            
            new_agents = []
            new_fitness = []
            
            for i in range(self.pop_size):
                # Enhanced RIME operations
                r1 = np.random.rand()
                
                if r1 < rime_factor:  # Soft-rime search
                    # Use Levy flight for better exploration
                    levy_step = self.levy_flight(self.dims)
                    new_agent = self.best_agent + rime_factor * levy_step
                    
                elif r1 < rime_factor + self.beta:  # Hard-rime puncture
                    # Select random agents for interaction
                    j, k = np.random.choice(self.pop_size, 2, replace=False)
                    
                    if self.fitness[i] > self.fitness[j]:
                        # Move towards better solution
                        new_agent = self.agents[i] + np.random.rand() * (self.agents[j] - self.agents[i])
                        
                        # Add crossover with best agent
                        if np.random.rand() < 0.3:
                            new_agent = self.crossover(new_agent, self.best_agent)
                    else:
                        # Exploration move
                        new_agent = self.agents[i] + np.random.randn(self.dims) * (self.agents[k] - self.agents[j])
                
                else:  # Elite-guided search
                    # Use elite solutions for guidance
                    if len(self.elite_agents) > 0:
                        elite_idx = np.random.randint(len(self.elite_agents))
                        elite = self.elite_agents[elite_idx]
                        
                        # Combine with current agent
                        new_agent = 0.7 * self.agents[i] + 0.3 * elite
                        
                        # Apply mutation for diversity
                        if diversity < self.diversity_threshold:
                            new_agent = self.mutation(new_agent, mutation_rate=0.2)
                    else:
                        new_agent = self.agents[i]
                
                # Boundary handling
                new_agent = np.clip(np.round(new_agent).astype(int), self.lb, self.ub)
                
                # Evaluate new agent
                new_fit = self.objective_func(new_agent)
                
                # Greedy selection
                if new_fit < self.fitness[i]:
                    new_agents.append(new_agent)
                    new_fitness.append(new_fit)
                else:
                    # Probabilistic acceptance for escaping local optima
                    if np.random.rand() < np.exp(-(new_fit - self.fitness[i]) / (E + 1e-10)):
                        new_agents.append(new_agent)
                        new_fitness.append(new_fit)
                    else:
                        new_agents.append(self.agents[i])
                        new_fitness.append(self.fitness[i])
            
            # Update population
            self.agents = np.array(new_agents)
            self.fitness = np.array(new_fitness)
            
            # Update elite pool
            self.update_elite()
            
            # Restart mechanism if stuck
            if self.stagnation_counter > 10:
                print(f"\n  Applying restart mechanism at iteration {t}")
                # Replace worst 30% with new random solutions
                worst_indices = np.argsort(self.fitness)[-int(0.3 * self.pop_size):]
                for idx in worst_indices:
                    self.agents[idx] = self.lb + np.random.rand(self.dims) * (self.ub - self.lb)
                    self.agents[idx] = np.round(self.agents[idx]).astype(int)
                    self.fitness[idx] = self.objective_func(self.agents[idx])
                self.stagnation_counter = 0
            
            # Update progress bar
            iterator.set_description(
                f"RIME | Best: {self.best_fitness:.4f} | Div: {diversity:.3f} | Stag: {self.stagnation_counter}"
            )
        
        print(f"\nOptimization complete. Final best fitness: {self.best_fitness:.4f}")
        return self.best_agent, self.best_fitness