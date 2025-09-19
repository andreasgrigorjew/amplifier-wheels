import argparse
import os
import networkx as nx
from datetime import datetime

REPEATS_DEFAULT = 150

class amplifier:
    V: list[int]
    E: list[tuple[int, int]]
    contacts: list[int]
    checkers: list[int]
    num_contacts: int
    num_checkers: int
    N: int

    def __init__(self, num_contacts: int, num_checkers: int, block_sizes: list[int] | None = None):
        """
        Initialize the amplifier with vertices, edges, contacts, and checkers.

        If block_sizes is provided, it must be a list of length num_contacts giving the
        number of checkers in each block (non-uniform). Otherwise, a uniform num_checkers
        is used for every block.
        """
        self.num_contacts = num_contacts
        self.num_checkers = num_checkers  # kept for backward compatibility (uniform case)

        # Determine per-block checker counts
        if block_sizes is not None:
            assert len(block_sizes) == num_contacts, "block_sizes length must equal num_contacts"
            assert all(isinstance(b, int) and b >= 0 for b in block_sizes), "block_sizes must be non-negative ints"
            self.block_checkers = list(block_sizes)
        else:
            self.block_checkers = [num_checkers] * num_contacts

        # Compute block starts (contact indices) and N
        block_sizes_with_contact = [1 + b for b in self.block_checkers]
        self.block_starts = []
        acc = 0
        for sz in block_sizes_with_contact:
            self.block_starts.append(acc)  # contact index for this block
            acc += sz
        self.N = acc

        self.V = list(range(self.N))

        # Build bidirected ring edges
        self.E = list()
        for i in range(len(self.V)):
            self.E.append((self.V[i], self.V[(i + 1) % len(self.V)]))
            self.E.append((self.V[(i + 1) % len(self.V)], self.V[i]))

        # Contacts: first vertex of each block
        self.contacts = self.block_starts[:]

        # Checkers: remaining vertices in each block
        self.checkers = []
        for g, start in enumerate(self.block_starts):
            for offset in range(1, 1 + self.block_checkers[g]):
                self.checkers.append(start + offset)

    @classmethod
    def from_average_block_size(cls, num_contacts: int, avg_checkers_per_block: float):
        """
        Build an amplifier whose per-block checker counts average to the given value.
        Uses a largest-remainder-style distribution between floor(avg) and ceil(avg).
        """
        from fractions import Fraction

        frac = Fraction(str(avg_checkers_per_block))
        base = frac.numerator // frac.denominator  # floor
        rem_num = frac.numerator - base * frac.denominator
        rem_den = frac.denominator

        # Number of blocks that should have base+1 checkers
        k = round(num_contacts * rem_num / rem_den)
        k = max(0, min(num_contacts, k))

        block_sizes = [base + 1] * k + [base] * (num_contacts - k)
        # Optional: shuffle to spread larger blocks
        import random
        random.shuffle(block_sizes)

        return cls(num_contacts, base, block_sizes=block_sizes)

    def group_of(self, v: int) -> int:
        """
        Return the block index for vertex v.
        """
        # block_starts is sorted; find the rightmost start <= v
        import bisect
        i = bisect.bisect_right(self.block_starts, v) - 1
        return max(0, i)

    def checkers_group(self, c: int):
        """
        Return the group of a checker vertex c.
        """
        return self.group_of(c)

    def random_matching(self):
        """
        Generate a random matching of the amplifier's checkers.
        The function returns a list of tuples representing the matching.
        """
        import random

        # Generate random pairs
        matching = []
        random_order = self.checkers.copy()
        random.shuffle(random_order)
        current_group = []
        for i in range(len(random_order)):
            if len(current_group) == 0 or self.checkers_group(current_group[-1]) == self.checkers_group(random_order[i]):
                current_group.append(random_order[i])
            else:
                matching.append((current_group[-1], random_order[i]))
                current_group.pop()
        
        return matching
    
    def generate_random_amplifier(self, degree: int):
        """
        Generate a random regular amplifier of degree `degree`+2.
        The function adjusts the amplifier in the class and adds edges to E.
        """
        for _ in range(degree):
            matching = self.random_matching()
            for u, v in matching:
                self.E.append((u, v))
                self.E.append((v, u))
    
    def trivially_check_expansion(self):
        """
        Check if the amplifier has an expansion of 1.
        For every subset S of V with |S| <= |V|/2, check if |E(S, V\\S)| >= |S|.
        Returns True if the property holds for all such S, False otherwise.
        Note: This checks all non-empty subsets of V of size at most |V|/2.
        This implementation is not efficient. Use the ILP formulation for larger amplifiers.
        """
        from itertools import combinations

        n = len(self.V)
        max_subset_size = n // 2
        # For all subset sizes from 1 to n//2
        for r in range(1, max_subset_size + 1):
            for S in combinations(self.V, r):
                S_set = set(S)
                S_comp = set(self.V) - S_set
                contacts_in_S_set = set(self.contacts) & S_set
                # Count edges from S to V\S (counting multiplicity)
                edge_count = 0
                for u in S_set:
                    for v in S_comp:
                        edge_count += self.E.count((u, v))
                if edge_count < len(contacts_in_S_set):
                    return False
        return True
    
    def ilp_check_expansion(self):
        """
        Check expansion using an ILP. Uses Gurobi solver.
        Returns (is_expanding, min_value, S) where:
            - is_expanding: True if the minimum is >= 0, False otherwise
            - min_value: the optimal value of sum(y_{i,j}) - sum(x_i)
            - S: the subset of vertices (as a list) corresponding to x_i = 1
        """
        try:
            import gurobipy as gp
            from gurobipy import GRB
        except ImportError:
            raise ImportError("Gurobi not installed. Install gurobipy or use the PuLP version.")

        n = len(self.V)
        
        # Create a new model
        model = gp.Model("ExpansionCheck")
        
        # Suppress output
        model.setParam('OutputFlag', 0)
        
        # Variables: x_i for each vertex, y_{i,j} for each edge (counting multiplicity)
        x = [model.addVar(vtype=GRB.BINARY, name=f"x_{i}") for i in self.V]
        y = [model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{j}_{k}") for k, (i, j) in enumerate(self.E)]

        # Update to register variables
        model.update()

        # y_{i,j} = x_i xor x_j, linearized:
        for k, (i, j) in enumerate(self.E):
            model.addConstr(y[k] >= x[i] - x[j], name=f"xor1_{k}")
            model.addConstr(y[k] >= x[j] - x[i], name=f"xor2_{k}")
            model.addConstr(y[k] <= x[i] + x[j], name=f"xor3_{k}")
            model.addConstr(y[k] <= 2 - x[i] - x[j], name=f"xor4_{k}")

        # Subset size constraint: 1 <= sum x_i <= n//2
        model.addConstr(gp.quicksum(x) >= 1, name="min_size")
        model.addConstr(gp.quicksum(x) <= n // 2, name="max_size")

        # Objective: minimize sum(y) - sum(x_i for i in self.contacts)*2
        obj_expr = gp.quicksum(y) - 2 * gp.quicksum(x[i] for i in self.contacts)
        model.setObjective(obj_expr, GRB.MINIMIZE)

        # Solve the model
        model.optimize()

        # Check if the problem was solved
        if model.status == GRB.OPTIMAL:
            min_value = model.objVal
            assert min_value % 2 == 0, "Objective value should be even"
            S = [i for i in self.V if x[i].X > 0.5]
            is_expanding = min_value >= 0
            return is_expanding, min_value/2, S
        else:
            print(f"Gurobi error: {model.status}")
            return False, float("inf"), []
    
    def check_ILP_result_correctness(self, S):
        """
        Check if the ILP result is correct.
        This function checks if the set S satisfies the expansion property.
        """
        n = len(self.V)
        S_set = set(S)
        S_comp = set(self.V) - S_set
        contacts_in_S_set = set(self.contacts) & S_set
        edge_count = 0
        for u in S_set:
            for v in S_comp:
                edge_count += self.E.count((u, v))
        return edge_count >= len(contacts_in_S_set) and len(S_set) <= n // 2
        
    @classmethod
    def create_biwheel_from_amplifier(cls, amp, num_contacts_per_wheel: int):
        """
        Convert an existing amplifier to a bi-wheel structure with two separate cycles.
        
        Parameters:
            amp: Existing amplifier instance
            num_contacts_per_wheel: Number of contacts in each wheel
        """
        # Clear existing edges
        amp.E = []
        
        # Separate vertices into left and right wheels
        left_wheel_vertices = set()
        right_wheel_vertices = set()
        
        # First half of contacts go to left wheel, second half to right wheel
        left_contacts = amp.contacts[:num_contacts_per_wheel]
        right_contacts = amp.contacts[num_contacts_per_wheel:]
        
        left_wheel_vertices.update(left_contacts)
        right_wheel_vertices.update(right_contacts)
        
        # Assign checkers to the appropriate wheel based on their group
        for checker in amp.checkers:
            group = amp.group_of(checker)
            if group < num_contacts_per_wheel:
                left_wheel_vertices.add(checker)
            else:
                right_wheel_vertices.add(checker)
        
        # Create cycle edges for left wheel
        left_wheel_list = sorted(left_wheel_vertices)
        for i in range(len(left_wheel_list)):
            u = left_wheel_list[i]
            v = left_wheel_list[(i + 1) % len(left_wheel_list)]
            amp.E.append((u, v))
            amp.E.append((v, u))
        
        # Create cycle edges for right wheel
        right_wheel_list = sorted(right_wheel_vertices)
        for i in range(len(right_wheel_list)):
            u = right_wheel_list[i]
            v = right_wheel_list[(i + 1) % len(right_wheel_list)]
            amp.E.append((u, v))
            amp.E.append((v, u))
        
        # Create matching between checkers from different wheels
        left_wheel_checkers = [c for c in amp.checkers if c in left_wheel_vertices]
        right_wheel_checkers = [c for c in amp.checkers if c in right_wheel_vertices]
        
        import random
        random.shuffle(left_wheel_checkers)
        random.shuffle(right_wheel_checkers)
        
        # Match checkers between wheels
        for i in range(min(len(left_wheel_checkers), len(right_wheel_checkers))):
            u = left_wheel_checkers[i]
            v = right_wheel_checkers[i]
            amp.E.append((u, v))
            amp.E.append((v, u))
        
        return amp

    @classmethod
    def create_biwheel(cls, num_contacts_per_wheel: int, num_checkers_per_contact: int):
        """
        Creates a bi-wheel amplifier with two separate cycles connected by checker matchings.
        
        Parameters:
            num_contacts_per_wheel: Number of contacts in each wheel
            num_checkers_per_contact: Number of checkers per contact
        """
        # Create amplifier with specified structure
        total_contacts = num_contacts_per_wheel * 2
        amp = cls(total_contacts, num_checkers_per_contact)
        
        # Convert to bi-wheel
        return cls.create_biwheel_from_amplifier(amp, num_contacts_per_wheel)

def find_amplifier(num_contacts: int, repeats: int = REPEATS_DEFAULT):
    """
    Find an amplifier with a given number of contacts using as few checkers as possible.
    It generates a random amplifier repeats many times.
    There is no guarantee that the amplifier will be found, but there always exists an amplifier with 6 checkers.
    """
    print(f"Searching for an amplifier with {num_contacts} contacts")
    for ell in range(1, 7):
        print(f"Using {ell} x {num_contacts} checkers")
        for _ in range(repeats):
            #print("Run number:", _ + 1, "using", ell, "checkers")
            amp = amplifier(num_contacts, ell)
            amp.generate_random_amplifier(1)
            is_expanding, min_value, violating_set = amp.ilp_check_expansion()
            if is_expanding:
                #print(f"Found an amplifier with {num_contacts} contacts and {ell} x {num_contacts} checkers")
                return amp
    return None

def find_amplifier_with_average(num_contacts: int, avg: float, repeats: int = REPEATS_DEFAULT):
    """
    Find an amplifier using an average (possibly fractional) number of checkers per block.
    Tries `repeats` random matchings for the given average construction.
    """
    print(f"Searching for an amplifier with {num_contacts} contacts and average {avg} checkers")
    for _ in range(repeats):
        amp = amplifier.from_average_block_size(num_contacts, avg)
        amp.generate_random_amplifier(1)
        is_expanding, min_value, violating_set = amp.ilp_check_expansion()
        if is_expanding:
            return amp
    return None

def find_biwheel_amplifier(num_contacts_per_wheel: int, repeats: int = REPEATS_DEFAULT):
    """
    Find a bi-wheel amplifier with minimal number of checkers per contact.
    
    Parameters:
        num_contacts_per_wheel: Number of contacts in each wheel
        repeats: Number of random trials per checker count
    """
    print(f"Searching for a bi-wheel amplifier with {num_contacts_per_wheel} contacts per wheel")
    for ell in range(1, 7):  # Try 1-6 checkers per contact
        print(f"Using {ell} checkers per contact ({ell*num_contacts_per_wheel} checkers per wheel)")
        for _ in range(repeats):
            amp = amplifier.create_biwheel(num_contacts_per_wheel, ell)
            is_expanding, min_value, violating_set = amp.ilp_check_expansion()
            if is_expanding:
                return amp
    return None

def find_biwheel_amplifier_with_average(num_contacts_per_wheel: int, avg: float, repeats: int = REPEATS_DEFAULT):
    """
    Find a bi-wheel amplifier with fractional average number of checkers.
    
    Parameters:
        num_contacts_per_wheel: Number of contacts in each wheel
        avg: Average number of checkers per contact
        repeats: Number of random trials
    """
    print(f"Searching for a bi-wheel amplifier with {num_contacts_per_wheel} contacts per wheel and average {avg} checkers per contact")
    for _ in range(repeats):
        # Create a standard amplifier with the given average
        total_contacts = num_contacts_per_wheel * 2
        amp = amplifier.from_average_block_size(total_contacts, avg)
        
        # Convert to bi-wheel
        amp = amplifier.create_biwheel_from_amplifier(amp, num_contacts_per_wheel)
        
        # Check expansion
        is_expanding, min_value, violating_set = amp.ilp_check_expansion()
        if is_expanding:
            return amp
    
    return None

def save_graph(amp, filename_prefix, output_dir='output_graphs'):
    """
    Save an amplifier graph using NetworkX in multiple formats.
    
    Parameters:
        amp: The amplifier object
        filename_prefix: Prefix for the output filenames
        output_dir: Directory to save the graphs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a NetworkX graph
    G = nx.MultiDiGraph()  # Use MultiDiGraph to handle parallel edges
    
    # Add nodes with attributes
    for v in amp.V:
        is_contact = v in amp.contacts
        G.add_node(v, is_contact=is_contact, 
                  type="contact" if is_contact else "checker",
                  group=amp.group_of(v))
    
    # Add edges
    for u, v in amp.E:
        G.add_edge(u, v)
    
    # Save in multiple formats
    base_path = os.path.join(output_dir, filename_prefix)
    
    # GraphML format - preserves node/edge attributes
    nx.write_graphml(G, f"{base_path}.graphml")
    
    # GEXF format - good for Gephi visualization
    nx.write_gexf(G, f"{base_path}.gexf")
    
    # Edge list - simple text format
    nx.write_edgelist(G, f"{base_path}.edgelist")
    
    return f"{base_path}"

def main():
    """
    Main function to demonstrate the amplifier class.
    """
    # Add command line argument parsing
    parser = argparse.ArgumentParser(description="Find and save amplifier graphs")
    parser.add_argument("--save-graphs", action="store_true", help="Save found graphs to files")
    parser.add_argument("--output-dir", default="output_graphs", help="Directory to save graphs")
    parser.add_argument("--max-contacts", type=int, default=40, help="Maximum number of contacts to try")
    parser.add_argument("--max-biwheel-contacts", type=int, default=20, help="Maximum contacts per wheel to try for bi-wheels")
    args = parser.parse_args()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Rest of your existing main function - just add graph saving where needed
    for num_contacts in range(1, args.max_contacts):
        amp = find_amplifier(num_contacts)
        if amp is not None:
            print(f"Amplifier found with {num_contacts} contacts and {amp.num_checkers} checkers.", flush=True)
            
            # Save the graph if requested
            if args.save_graphs:
                filename = f"wheel_{num_contacts}c_{amp.num_checkers}ch_{timestamp}"
                path = save_graph(amp, filename, args.output_dir)
                print(f"Graph saved to {path}.*")

            # Try to find an amplifier with a rational number of average checkers
            avg = amp.num_checkers - 0.5
            amp_avg = find_amplifier_with_average(num_contacts, avg)
            if amp_avg is not None:
                print(f"Amplifier found with {num_contacts} contacts and average {avg} checkers.", flush=True)
                
                # Save the fractional graph if requested
                if args.save_graphs:
                    filename = f"wheel_{num_contacts}c_avg{avg}ch_{timestamp}"
                    path = save_graph(amp_avg, filename, args.output_dir)
                    print(f"Graph saved to {path}.*")
                
                print("")
            else:
                print(f"No amplifier found with {num_contacts} contacts and average {avg} checkers.", flush=True)
                print("")

        if amp is None:
            print(f"No amplifier found with {num_contacts} contacts.")
            print(f"This is unexpected, as there should always be an amplifier with 6 checkers.", flush=True)
            break

    print("\n=== Bi-Wheel Amplifiers ===")
    for num_contacts_per_wheel in range(1, args.max_biwheel_contacts):
        amp = find_biwheel_amplifier(num_contacts_per_wheel)
        if amp is not None:
            print(f"Bi-wheel amplifier found with {num_contacts_per_wheel} contacts per wheel and {amp.num_checkers} checkers per contact.", flush=True)
            
            # Save the graph if requested
            if args.save_graphs:
                filename = f"biwheel_{num_contacts_per_wheel}c_{amp.num_checkers}ch_{timestamp}"
                path = save_graph(amp, filename, args.output_dir)
                print(f"Graph saved to {path}.*")

            # Try to find a bi-wheel with a fractional number of checkers
            avg = amp.num_checkers - 0.5
            amp_avg = find_biwheel_amplifier_with_average(num_contacts_per_wheel, avg)
            if amp_avg is not None:
                print(f"Bi-wheel amplifier found with {num_contacts_per_wheel} contacts per wheel and average {avg} checkers per contact.", flush=True)
                
                # Save the fractional graph if requested
                if args.save_graphs:
                    filename = f"biwheel_{num_contacts_per_wheel}c_avg{avg}ch_{timestamp}"
                    path = save_graph(amp_avg, filename, args.output_dir)
                    print(f"Graph saved to {path}.*")
            
            print("")
        
        if amp is None:
            print(f"No bi-wheel amplifier found with {num_contacts_per_wheel} contacts per wheel.")
            break

if __name__ == "__main__":
    main()
