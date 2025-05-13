import pulp

import numpy as np

class ProductionWarehouseOptimizer:
    def __init__(self, num_items, num_locations, num_periods):
        """
        Initialize the optimizer with problem dimensions
        
        Args:
            num_items (int): Number of items (I)
            num_locations (int): Number of storage locations (L)
            num_periods (int): Number of time periods (T)
        """
        self.I = num_items
        self.L = num_locations
        self.T = num_periods
        
        # Initialize all parameters with default values (to be set later)
        self.R = np.zeros(num_locations)  # Space cost per location
        self.O = np.zeros(num_locations)  # Retrieval cost per location
        self.P = np.zeros(num_locations)  # Moving cost to location
        self.h = np.zeros((num_items, num_periods))  # Holding cost
        self.u = np.zeros((num_items, num_periods))  # Setup cost
        self.c = np.zeros((num_items, num_periods))  # Production cost
        self.d = np.zeros((num_items, num_periods))  # Demand
        self.F = np.zeros(num_periods)  # Production capacity
        self.v = np.zeros((num_items, num_periods))  # Resource usage
        self.M = 1e6  # Large number for big-M constraints
        
        # Initial inventory (n_0^il)
        self.initial_inventory = np.zeros((num_items, num_locations))
        
        # Create the optimization model
        self.model = pulp.LpProblem("Integrated_Production_Warehouse_Optimization", pulp.LpMinimize)
        
        # Decision variables
        self.x = None  # Production quantity
        self.y = None  # Production setup
        self.w = None  # Movement to storage
        self.q = None  # Retrieval from storage
        self.n = None  # Inventory at location
        self.z = None  # Location usage
        
        self._create_variables()
        self._add_constraints()
    
    def _create_variables(self):
        """Create all decision variables"""
        # Production quantity (integer)
        self.x = pulp.LpVariable.dicts("x", 
                                     ((i, t) for i in range(self.I) for t in range(self.T)),
                                     lowBound=0, 
                                     cat='Integer')
        
        # Production setup (binary)
        self.y = pulp.LpVariable.dicts("y", 
                                     ((i, t) for i in range(self.I) for t in range(self.T)),
                                     cat='Binary')
        
        # Movement to storage (binary)
        self.w = pulp.LpVariable.dicts("w", 
                                      ((i, l, t) for i in range(self.I) for l in range(self.L) for t in range(self.T)),
                                      cat='Binary')
        
        # Retrieval from storage (binary)
        self.q = pulp.LpVariable.dicts("q", 
                                      ((i, l, t) for i in range(self.I) for l in range(self.L) for t in range(self.T)),
                                      cat='Binary')
        
        # Inventory at location (binary)
        self.n = pulp.LpVariable.dicts("n", 
                                      ((i, l, t) for i in range(self.I) for l in range(self.L) for t in range(self.T+1)),  # T+1 to include initial
                                      cat='Binary')
        
        # Location usage (binary)
        self.z = pulp.LpVariable.dicts("z", 
                                     ((l, t) for l in range(self.L) for t in range(self.T)),
                                     cat='Binary')
    
    def _add_constraints(self):
        """Add all constraints to the model"""
        # Constraint (2): Item can only be moved to location if it's available
        for l in range(self.L):
            for t in range(self.T):
                for i in range(self.I):
                    self.model += self.w[i,l,t] + self.n[i,l,t] - 1 <= self.z[l,t]
        
        # Constraint (3): Quantity moved equals quantity produced
        for i in range(self.I):
            for t in range(self.T):
                self.model += pulp.lpSum(self.w[i,l,t] for l in range(self.L)) == self.x[i,t]
        
        # Constraint (4): Retrieval equals demand
        for i in range(self.I):
            for t in range(self.T):
                self.model += pulp.lpSum(self.q[i,l,t] for l in range(self.L)) == self.d[i,t]
        
        # Constraint (5): Flow balance
        for i in range(self.I):
            for l in range(self.L):
                for t in range(1, self.T+1):  # Starting from t=1
                    if t == 1:
                        # For first period, use initial inventory
                        self.model += self.n[i,l,t] == self.initial_inventory[i,l] + self.w[i,l,t-1] - self.q[i,l,t-1]
                    else:
                        self.model += self.n[i,l,t] == self.n[i,l,t-1] + self.w[i,l,t-1] - self.q[i,l,t-1]
        
        # Constraint (6): Production setup
        for i in range(self.I):
            for t in range(self.T):
                self.model += self.x[i,t] <= self.M * self.y[i,t]
        
        # Constraint (7): Production capacity
        for t in range(self.T):
            self.model += pulp.lpSum(self.v[i,t] * self.x[i,t] for i in range(self.I)) <= self.F[t]
        
        # Although constraints (10)-(16) are redundant, we can add them to potentially help the solver
        # Constraint (10): At most one item moved to location per period
        for l in range(self.L):
            for t in range(self.T):
                self.model += pulp.lpSum(self.w[i,l,t] for i in range(self.I)) <= 1
        
        # Constraint (11): At most one item inventoried per location
        for l in range(self.L):
            for t in range(1, self.T+1):
                self.model += pulp.lpSum(self.n[i,l,t] for i in range(self.I)) <= 1
        
        # Constraint (12): At most one item retrieved per location per period
        for l in range(self.L):
            for t in range(self.T):
                self.model += pulp.lpSum(self.q[i,l,t] for i in range(self.I)) <= 1
        
        # Constraint (13): Can only retrieve if previously inventoried or moved
        for i in range(self.I):
            for l in range(self.L):
                for t in range(self.T):
                    if t == 0:
                        self.model += self.q[i,l,t] <= self.w[i,l,t] + self.initial_inventory[i,l]
                    else:
                        self.model += self.q[i,l,t] <= self.w[i,l,t] + self.n[i,l,t-1]
        
        # Constraints (14)-(16): Operations only in available locations
        for i in range(self.I):
            for l in range(self.L):
                for t in range(self.T):
                    self.model += self.w[i,l,t] <= self.z[l,t]
                    self.model += self.n[i,l,t] <= self.z[l,t]
                    self.model += self.q[i,l,t] <= self.z[l,t]
    
    def set_objective(self):
        """Set the objective function"""
        # Space cost
        space_cost = pulp.lpSum(self.R[l] * self.z[l,t] for l in range(self.L) for t in range(self.T))
        
        # Movement, retrieval, and holding costs
        operation_cost = pulp.lpSum(
            self.P[l] * self.w[i,l,t] + self.O[l] * self.q[i,l,t] + self.h[i,t] * self.n[i,l,t] 
            for i in range(self.I) for l in range(self.L) for t in range(self.T)
        )
        
        # Production and setup costs
        production_cost = pulp.lpSum(
            self.c[i,t] * self.x[i,t] + self.u[i,t] * self.y[i,t] 
            for i in range(self.I) for t in range(self.T)
        )
        
        # Total objective
        self.model += space_cost + operation_cost + production_cost
    
    def solve(self, solver=None, time_limit=None):
        """
        Solve the optimization problem
        
        Args:
            solver: Optional solver to use (e.g., pulp.GUROBI, pulp.CPLEX)
            time_limit: Optional time limit in seconds
            
        Returns:
            dict: Solution status and values
        """
        if solver is None:
            solver = pulp.PULP_CBC_CMD(msg=True, timeLimit=time_limit)
        
        self.set_objective()
        status = self.model.solve(solver)
        
        solution = {
            'status': pulp.LpStatus[status],
            'objective_value': pulp.value(self.model.objective),
            'production': {},
            'storage': {},
            'retrieval': {},
            'inventory': {}
        }
        
        # Collect production decisions
        for i in range(self.I):
            for t in range(self.T):
                solution['production'][(i,t)] = {
                    'x': pulp.value(self.x[i,t]),
                    'y': pulp.value(self.y[i,t])
                }
        
        # Collect storage decisions
        for i in range(self.I):
            for l in range(self.L):
                for t in range(self.T):
                    solution['storage'][(i,l,t)] = pulp.value(self.w[i,l,t])
                    solution['retrieval'][(i,l,t)] = pulp.value(self.q[i,l,t])
                    solution['inventory'][(i,l,t)] = pulp.value(self.n[i,l,t])
        
        return solution

# Example usage
if __name__ == "__main__":
    # Problem dimensions
    num_items = 3
    num_locations = 5
    num_periods = 4
    
    # Create optimizer instance
    optimizer = ProductionWarehouseOptimizer(num_items, num_locations, num_periods)
    
    # Set parameters (in a real scenario, these would come from your data)
    optimizer.R = np.random.rand(num_locations) * 10  # Space costs
    optimizer.O = np.random.rand(num_locations) * 5   # Retrieval costs
    optimizer.P = np.random.rand(num_locations) * 7    # Moving costs
    optimizer.h = np.random.rand(num_items, num_periods) * 2  # Holding costs
    optimizer.u = np.random.rand(num_items, num_periods) * 50  # Setup costs
    optimizer.c = np.random.rand(num_items, num_periods) * 20  # Production costs
    optimizer.d = np.random.randint(1, 10, size=(num_items, num_periods))  # Demand
    optimizer.F = np.ones(num_periods) * 100  # Production capacity
    optimizer.v = np.random.rand(num_items, num_periods) * 2  # Resource usage
    
    # Solve the problem
    solution = optimizer.solve(time_limit=60)
    
    print(f"Solution status: {solution['status']}")
    print(f"Objective value: {solution['objective_value']}")
    
    # Print some solution details
    print("\nProduction plan:")
    for (i,t), vals in solution['production'].items():
        if vals['x'] > 0:
            print(f"Item {i}, Period {t}: Produce {vals['x']} units (setup: {vals['y']})")
    
    print("\nStorage assignments:")
    for (i,l,t), val in solution['storage'].items():
        if val > 0.5:  # Binary variable threshold
            print(f"Item {i} stored in location {l} in period {t}")
    
    print("\nRetrieval operations:")
    for (i,l,t), val in solution['retrieval'].items():
        if val > 0.5:
            print(f"Item {i} retrieved from location {l} in period {t}")