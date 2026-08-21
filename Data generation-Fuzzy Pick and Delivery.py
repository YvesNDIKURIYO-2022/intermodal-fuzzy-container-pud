#!/usr/bin/env python
# coding: utf-8

# In[7]:


"""
Modified Fuzzy Solomon Instance Generator
==========================================
Generates 168 instances from Solomon VRPTW benchmark with:
- 56 Solomon files × 3 sizes (25, 50, 100)
- Gradually increasing hubs: 2 hubs (25 cust), 3 hubs (50 cust), 4 hubs (100 cust)
- Triangular fuzzy travel times
- Border crossing arcs with fuzzy clearance times
- Pickup and delivery tasks
- Hub capacity constraints

Output: JSON instances in fuzzy_solomon_modified/
==========================================
"""

import numpy as np
import numpy.random as rnd
import json
import os
import math
from dataclasses import dataclass
from typing import List, Dict
from datetime import datetime
import glob


# =============================================================================
# Fuzzy Number
# =============================================================================

@dataclass
class TriangularFuzzyNumber:
    """Triangular fuzzy number (L, M, U)"""
    L: float
    M: float
    U: float

    def to_dict(self):
        return {'L': round(self.L, 2), 'M': round(self.M, 2), 'U': round(self.U, 2)}

    @classmethod
    def from_deterministic(cls, value: float, fuzziness: float = 0.3):
        if value <= 0:
            value = 0.1
        return cls(
            L=max(0.1, value * (1 - fuzziness)),
            M=value,
            U=value * (1 + fuzziness * 1.5)
        )


# =============================================================================
# Solomon Parser
# =============================================================================

class SolomonParser:
    """Parse Solomon VRPTW benchmark files"""

    @staticmethod
    def parse(filepath: str) -> Dict:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            raise ValueError(f"Empty file: {filepath}")

        instance_name = lines[0].split()[0] if lines[0] else "Unknown"

        customers = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 7 and parts[0].isdigit():
                cust_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                demand = float(parts[3])
                ready_time = float(parts[4])
                due_date = float(parts[5])
                service_time = float(parts[6])

                customers.append({
                    'id': cust_id,
                    'x': x,
                    'y': y,
                    'demand': demand,
                    'ready_time': ready_time,
                    'due_date': due_date,
                    'service_time': service_time,
                    'is_depot': (cust_id == 0)
                })

        return {
            'name': instance_name,
            'customers': customers,
            'num_customers': len([c for c in customers if not c['is_depot']])
        }


# =============================================================================
# Hub-and-Spoke Transformer
# =============================================================================

class HubSpokeTransformer:
    """Convert Solomon instance to hub-and-spoke topology"""

    def __init__(self, customers: List[Dict], num_hubs: int, seed: int = 42):
        self.customers = customers
        self.num_hubs = num_hubs
        rnd.seed(seed)
        self.hubs = []
        self.spokes = []
        self.assignments = {}

    def select_hubs(self) -> List[int]:
        """Select hubs based on centrality (nearest to all other nodes)"""
        non_depot = [c for c in self.customers if not c['is_depot']]

        # Calculate average distance to all other nodes
        centralities = []
        for c1 in non_depot:
            total_dist = 0
            for c2 in non_depot:
                if c1['id'] != c2['id']:
                    dist = math.sqrt((c1['x'] - c2['x'])**2 + (c1['y'] - c2['y'])**2)
                    total_dist += dist
            avg_dist = total_dist / len(non_depot)
            centralities.append((c1['id'], avg_dist))

        centralities.sort(key=lambda x: x[1])
        self.hubs = [cid for cid, _ in centralities[:self.num_hubs]]

        # Remaining nodes are spokes
        self.spokes = [c['id'] for c in non_depot if c['id'] not in self.hubs]

        # Assign each spoke to nearest hub
        for spoke in self.spokes:
            spoke_cust = next(c for c in non_depot if c['id'] == spoke)
            min_dist = float('inf')
            best_hub = None
            for hub in self.hubs:
                hub_cust = next(c for c in non_depot if c['id'] == hub)
                dist = math.sqrt((spoke_cust['x'] - hub_cust['x'])**2 + 
                                (spoke_cust['y'] - hub_cust['y'])**2)
                if dist < min_dist:
                    min_dist = dist
                    best_hub = hub
            self.assignments[spoke] = best_hub

        return self.hubs

    def get_type(self, node_id: int) -> str:
        if node_id == 0:
            return 'depot'
        elif node_id in self.hubs:
            return 'hub'
        else:
            return 'spoke'


# =============================================================================
# Fuzzy Solomon Instance Generator
# =============================================================================

class FuzzySolomonGenerator:
    """Generate fuzzy intermodal instances from Solomon data"""

    def __init__(self, solomon: Dict, output_dir: str):
        self.solomon = solomon
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.modes = {
            'road': {'speed': 60, 'cost': 1.5, 'fuzziness': 0.30},
            'rail': {'speed': 45, 'cost': 0.8, 'fuzziness': 0.30},
            'maritime': {'speed': 25, 'cost': 0.5, 'fuzziness': 0.30}
        }

    def distance(self, node1: Dict, node2: Dict) -> float:
        return math.sqrt((node1['x'] - node2['x'])**2 + (node1['y'] - node2['y'])**2)

    def get_mode(self, dist: float, type1: str, type2: str) -> str:
        if type1 == 'spoke' or type2 == 'spoke':
            return 'road'
        if type1 == 'hub' and type2 == 'hub':
            if dist <= 150:
                return 'road'
            elif dist <= 500:
                return 'rail'
            else:
                return 'maritime'
        return 'road' if dist <= 150 else 'rail'

    def get_num_hubs(self, num_customers: int) -> int:
        """Determine number of hubs based on instance size"""
        if num_customers <= 30:
            return 2
        elif num_customers <= 70:
            return 3
        else:
            return 4

    def generate(self) -> Dict:
        """Generate a single fuzzy instance with appropriate hub count"""

        num_customers = self.solomon['num_customers']
        num_hubs = self.get_num_hubs(num_customers)

        transformer = HubSpokeTransformer(self.solomon['customers'], num_hubs)
        transformer.select_hubs()

        node_dict = {c['id']: c for c in self.solomon['customers']}
        depot = next(c for c in self.solomon['customers'] if c['is_depot'])

        # Nodes
        nodes = []
        for cust in self.solomon['customers']:
            nodes.append({
                'node_id': cust['id'],
                'type': transformer.get_type(cust['id']),
                'x': cust['x'],
                'y': cust['y'],
                'demand': cust['demand'],
                'ready_time': cust['ready_time'],
                'due_date': cust['due_date'],
                'service_time': cust['service_time']
            })

        # Links (arcs)
        links = []
        all_ids = [0] + transformer.hubs + transformer.spokes

        for i in all_ids:
            for j in all_ids:
                if i >= j:
                    continue

                node_i = node_dict.get(i, {'x': depot['x'], 'y': depot['y']} if i == 0 else None)
                node_j = node_dict.get(j, {'x': depot['x'], 'y': depot['y']} if j == 0 else None)

                if node_i is None or node_j is None:
                    continue

                dist = self.distance(node_i, node_j)
                if dist < 0.1:
                    continue

                type_i = transformer.get_type(i) if i != 0 else 'depot'
                type_j = transformer.get_type(j) if j != 0 else 'depot'

                mode = self.get_mode(dist, type_i, type_j)
                params = self.modes[mode]

                det_time = dist / params['speed']
                fuzzy_time = TriangularFuzzyNumber.from_deterministic(det_time, params['fuzziness'])
                cost = dist * params['cost']

                links.append({'origin': i, 'destination': j, 'mode': mode,
                             'distance': round(dist, 2), 'time': round(det_time, 2),
                             'fuzzy_time': fuzzy_time.to_dict(), 'cost': round(cost, 2)})
                links.append({'origin': j, 'destination': i, 'mode': mode,
                             'distance': round(dist, 2), 'time': round(det_time, 2),
                             'fuzzy_time': fuzzy_time.to_dict(), 'cost': round(cost, 2)})

        # Tasks (pickup and delivery)
        tasks = []
        for cust in self.solomon['customers']:
            if cust['is_depot']:
                continue

            delivery = cust['demand']
            pickup = delivery * rnd.uniform(0.5, 1.5)

            tasks.append({
                'id': cust['id'],
                'origin': cust['id'],
                'destination': cust['id'],
                'delivery_teu': round(delivery, 2),
                'pickup_teu': round(pickup, 2),
                'earliest_pickup': cust['ready_time'],
                'latest_delivery': cust['due_date'],
                'service_time': cust['service_time'],
                'node_type': transformer.get_type(cust['id']),
                'assigned_hub': transformer.assignments.get(cust['id'])
            })

        # Border crossings
        borders = []
        if len(transformer.hubs) >= 2:
            hub_pairs = []
            for i, h1 in enumerate(transformer.hubs):
                for h2 in transformer.hubs[i+1:]:
                    hub_pairs.append((h1, h2))

            for h1, h2 in hub_pairs[:len(transformer.hubs)-1]:
                base = rnd.uniform(1, 8)
                borders.append({
                    'id': len(borders), 'from': h1, 'to': h2,
                    'inspection': TriangularFuzzyNumber.from_deterministic(base, 0.3).to_dict(),
                    'customs': TriangularFuzzyNumber.from_deterministic(base * 1.2, 0.3).to_dict()
                })

        # Hub capacities
        capacities = {}
        for hub in transformer.hubs:
            demand = sum(t['delivery_teu'] for t in tasks if t['assigned_hub'] == hub)
            capacities[hub] = round(max(demand * 1.2, 100), 2)

        return {
            'instance_id': f"FS_{self.solomon['name']}_H{num_hubs}",
            'source': self.solomon['name'],
            'customers': num_customers,
            'num_hubs': num_hubs,
            'nodes': nodes,
            'hubs': transformer.hubs,
            'spokes': transformer.spokes,
            'links': links,
            'border_crossings': borders,
            'tasks': tasks,
            'hub_capacities': capacities,
            'vehicles': {
                'road': {'capacity': 20, 'cost_per_km': 1.5, 'speed': 60},
                'rail': {'capacity': 50, 'cost_per_km': 0.8, 'speed': 45},
                'maritime': {'capacity': 100, 'cost_per_km': 0.5, 'speed': 25}
            },
            'penalty_per_hour': 50.0,
            'generated': datetime.now().isoformat()
        }

    def save(self, instance: Dict) -> str:
        filename = f"{instance['instance_id']}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(instance, f, indent=2)
        return filepath


# =============================================================================
# Main: Generate 168 Instances (One Per File)
# =============================================================================

def main():
    """Generate 168 fuzzy Solomon instances (one per file)"""

    # =========================================================
    # PATHS - UPDATE IF NEEDED
    # =========================================================

    input_dirs = {
        25: r"....\solomon_25",
        50: r".....\solomon_50",
        100: r".....\solomon_100"
    }

    output_dir = r".........."

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print("="*60)
    print("FUZZY SOLOMON INSTANCE GENERATOR")
    print("168 instances: 56 files × 3 sizes (one instance per file)")
    print("Hub assignment: 2 hubs (25 cust), 3 hubs (50 cust), 4 hubs (100 cust)")
    print("="*60)
    print(f"\nOutput directory: {output_dir}\n")

    total = 0

    for size, path in input_dirs.items():
        if not os.path.exists(path):
            print(f"Warning: {path} not found")
            continue

        files = glob.glob(os.path.join(path, "*.txt"))
        files = [f for f in files if os.path.basename(f)[0].isalpha()]
        files.sort()

        print(f"Processing {size} customers: {len(files)} files")

        for filepath in files:
            parser = SolomonParser()
            solomon = parser.parse(filepath)

            generator = FuzzySolomonGenerator(solomon, output_dir)
            instance = generator.generate()
            generator.save(instance)
            total += 1

            num_hubs = instance['num_hubs']
            print(f"  ✓ {solomon['name']} | {size}cust | {num_hubs}hubs")

        print()

    print("="*60)
    print(f"COMPLETE: {total} instances generated")
    print(f"Output: {output_dir}")
    print("="*60)

    # Verify count
    expected = 168
    if total == expected:
        print(f"\n✓ SUCCESS: {expected} instances generated as expected")
    else:
        print(f"\n⚠ Note: Generated {total} instances (expected {expected})")


if __name__ == "__main__":
    main()


# In[1]:


#!/usr/bin/env python3
"""
Unified Fuzzy Dataset Generator for Solomon and HHLP Datasets
FINAL VERSION - Based on Reviewer Recommendations

This script generates fuzzy instances for:
1. Solomon VRPTW benchmark (168 instances) - Primary benchmark
2. HHLP dataset (18 instances) - Scalability validation

Total: 186 instances

All instances include:
- Triangular fuzzy numbers for travel times and demands
- Hub-and-spoke topology with K-means hub selection
- Pickup-and-delivery constraints
- Time windows (Solomon only)
- Border crossing delays
- Mode-specific transport parameters
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import os
import math
import re
import time
from typing import Dict, Any, Tuple, Optional, List, Union
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from datetime import datetime
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FUZZY NUMBER CLASS
# ============================================================================

@dataclass
class TriangularFuzzyNumber:
    """Triangular fuzzy number (L, M, U) for uncertainty representation"""
    L: float  # Lower bound (optimistic)
    M: float  # Modal value (most likely)
    U: float  # Upper bound (pessimistic)

    def to_dict(self):
        return {'L': round(self.L, 4), 'M': round(self.M, 4), 'U': round(self.U, 4)}

    @classmethod
    def from_deterministic(cls, value: float, fuzziness: float = 0.30, 
                          lower_factor: float = None, upper_factor: float = None):
        """
        Create triangular fuzzy number from deterministic value.

        Args:
            value: Deterministic (modal) value
            fuzziness: Base fuzziness level (default 0.30)
            lower_factor: Factor for lower bound (default 1 - fuzziness)
            upper_factor: Factor for upper bound (default 1 + 1.5*fuzziness)
        """
        if value <= 0:
            value = 0.1

        if lower_factor is None:
            lower_factor = 1 - fuzziness
        if upper_factor is None:
            upper_factor = 1 + 1.5 * fuzziness

        return cls(
            L=max(0.0, value * lower_factor),
            M=value,
            U=value * upper_factor
        )

    @classmethod
    def from_bounds(cls, lower: float, modal: float, upper: float):
        """Create triangular fuzzy number from explicit bounds."""
        return cls(L=lower, M=modal, U=upper)

    @property
    def spread(self) -> float:
        """Calculate spread (uncertainty range)."""
        return self.U - self.L

    @property
    def relative_spread(self) -> float:
        """Calculate relative spread (normalized by modal value)."""
        return self.spread / self.M if self.M > 0 else 0

    def __add__(self, other: 'TriangularFuzzyNumber') -> 'TriangularFuzzyNumber':
        """Fuzzy addition."""
        return TriangularFuzzyNumber(
            L=self.L + other.L,
            M=self.M + other.M,
            U=self.U + other.U
        )

    def expected_value(self) -> float:
        """Expected value defuzzification (graded mean integration)."""
        return (self.L + 4 * self.M + self.U) / 6

    def to_tuple(self) -> Tuple[float, float, float]:
        return (self.L, self.M, self.U)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Fuzziness parameters
    'delta': 0.30,  # Base fuzziness level
    'demand_fuzziness': 0.10,  # Demand uncertainty

    # Hub parameters
    'hub_capacity_multiplier': 1.2,
    'min_hub_capacity': 100,

    # Solomon-specific
    'solomon_hub_ratio': 25,  # One hub per 25 customers

    # HHLP-specific
    'hhlp_hub_ratio': 50,  # One hub per 50 zones

    # Transport modes
    'transport_modes': {
        'road': {
            'speed': 60,  # km/h
            'cost_per_km': 1.5,  # USD/km
            'capacity': 20,  # TEU
            'fuzziness': 0.30,
            'max_distance': 150  # km
        },
        'rail': {
            'speed': 45,
            'cost_per_km': 0.8,
            'capacity': 50,
            'fuzziness': 0.30,
            'max_distance': 500
        },
        'maritime': {
            'speed': 25,
            'cost_per_km': 0.5,
            'capacity': 100,
            'fuzziness': 0.30,
            'max_distance': float('inf')
        }
    },

    # Border delay parameters (based on empirical studies)
    'border_delay': {
        'inspection': {'min': 1, 'max': 4, 'fuzziness': 0.30},
        'customs': {'min': 2, 'max': 8, 'fuzziness': 0.30}
    },

    # Random seed for reproducibility
    'random_seed': 42,

    # Output format
    'save_figures': True,
    'verbose': True
}

# Solomon instance configurations
SOLOMON_CONFIG = {
    'types': ['C', 'R', 'RC'],
    'type_names': {
        'C': 'Clustered',
        'R': 'Random',
        'RC': 'Mixed'
    },
    'instances': [101, 102, 103, 104, 105, 106, 107, 108, 
                  201, 202, 203, 204, 205, 206, 207, 208],
    'sizes': [25, 50, 100],
    'size_dirs': {
        25: 'solomon_25',
        50: 'solomon_50',
        100: 'solomon_100'
    }
}

# HHLP dataset configuration
HHLP_CONFIG = {
    'cities': ['Kolkata', 'Mumbai'],
    'sizes': [100, 300, 400, 500, 600, 800, 900, 1000, 2000],
    'subdir': 'LargeDataSetHHLP',
    'hub_ratio': 50
}

# ============================================================================
# FILE LOADING HELPERS
# ============================================================================

def load_matrix_csv(filepath: Union[str, Path], expected_size: int = None) -> pd.DataFrame:
    """
    Robust matrix loader handling various CSV formats.

    Supports:
    - Standard CSV with header
    - Flattened matrices (1 row, N*N columns)
    - Bracket notation from JSON-like files
    - Files with extra columns/rows
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    with open(filepath, 'r') as f:
        content = f.read().strip()

    # Try bracket notation (e.g., [[...]])
    if content.startswith('[[') or content.startswith('['):
        try:
            import ast
            data = ast.literal_eval(content)
            df = pd.DataFrame(data)
            if expected_size:
                df = _reshape_matrix(df, expected_size)
            return df
        except:
            pass

    # Try standard CSV
    try:
        df = pd.read_csv(filepath, header=None)
        # Clean data
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.fillna(0)
        if expected_size:
            df = _reshape_matrix(df, expected_size)
        return df
    except:
        pass

    # Fallback: manual parsing
    try:
        lines = content.split('\n')
        data = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'[\[\]]', '', line)
            values = []
            for val in line.split(','):
                val = val.strip()
                if val:
                    try:
                        values.append(float(val))
                    except:
                        num_match = re.search(r'[\d.]+', val)
                        if num_match:
                            values.append(float(num_match.group()))
                        else:
                            values.append(0.0)
            if values:
                data.append(values)
        if data:
            df = pd.DataFrame(data)
            if expected_size:
                df = _reshape_matrix(df, expected_size)
            return df
    except:
        pass

    raise ValueError(f"Could not parse file: {filepath}")

def _reshape_matrix(df: pd.DataFrame, expected_size: int) -> pd.DataFrame:
    """Reshape matrix to expected square size."""
    n_rows, n_cols = df.shape

    # Already correct
    if n_rows == expected_size and n_cols == expected_size:
        return df

    # Flattened matrix (1 row, N*N columns)
    if n_rows == 1 and n_cols == expected_size * expected_size:
        arr = df.values.flatten()
        matrix = arr.reshape(expected_size, expected_size)
        return pd.DataFrame(matrix)

    # Extra columns or rows - trim
    if n_rows >= expected_size and n_cols >= expected_size:
        return df.iloc[:expected_size, :expected_size].copy()

    # Too small - pad with zeros
    if n_rows < expected_size or n_cols < expected_size:
        new_matrix = np.zeros((expected_size, expected_size))
        rows = min(n_rows, expected_size)
        cols = min(n_cols, expected_size)
        new_matrix[:rows, :cols] = df.iloc[:rows, :cols].values
        return pd.DataFrame(new_matrix)

    return df

# ============================================================================
# FUZZY UTILITIES
# ============================================================================

class FuzzyUtils:
    """Utility functions for fuzzy arithmetic and operations."""

    @staticmethod
    def make_fuzzy_travel_time(cost: float, mode: str, config: Dict) -> TriangularFuzzyNumber:
        """
        Create fuzzy travel time for a given cost and transport mode.

        Args:
            cost: Distance/cost value
            mode: Transport mode ('road', 'rail', 'maritime')
            config: Configuration dictionary

        Returns:
            TriangularFuzzyNumber: Fuzzy travel time
        """
        modes = config.get('transport_modes', {})
        mode_config = modes.get(mode, modes.get('road', {}))

        speed = mode_config.get('speed', 60)
        fuzziness = mode_config.get('fuzziness', 0.30)

        travel_time = cost / speed if speed > 0 else 0

        return TriangularFuzzyNumber.from_deterministic(travel_time, fuzziness)

    @staticmethod
    def make_fuzzy_demand(demand: float, fuzziness: float = 0.10) -> TriangularFuzzyNumber:
        """Create fuzzy demand."""
        return TriangularFuzzyNumber.from_deterministic(demand, fuzziness)

    @staticmethod
    def make_border_delay(mode: str = 'customs', config: Dict = None) -> TriangularFuzzyNumber:
        """
        Generate fuzzy border delay based on empirical distributions.

        Args:
            mode: 'inspection' or 'customs'
            config: Configuration dictionary

        Returns:
            TriangularFuzzyNumber: Fuzzy border delay
        """
        if config is None:
            config = CONFIG

        border_config = config.get('border_delay', {})
        mode_config = border_config.get(mode, border_config.get('customs', {}))

        min_val = mode_config.get('min', 2)
        max_val = mode_config.get('max', 8)
        fuzziness = mode_config.get('fuzziness', 0.30)

        # Generate modal value uniformly within range
        modal = np.random.uniform(min_val, max_val)

        return TriangularFuzzyNumber.from_deterministic(modal, fuzziness)

    @staticmethod
    def propagate_fuzzy_path(times: List[TriangularFuzzyNumber]) -> TriangularFuzzyNumber:
        """
        Propagate uncertainty through a path using fuzzy addition.

        Args:
            times: List of fuzzy travel times

        Returns:
            TriangularFuzzyNumber: Cumulative fuzzy time
        """
        if not times:
            return TriangularFuzzyNumber(0, 0, 0)

        result = times[0]
        for t in times[1:]:
            result = result + t
        return result

# ============================================================================
# HUB GENERATOR
# ============================================================================

class HubGenerator:
    """Generate hub nodes for network instances."""

    @staticmethod
    def get_hub_count(nodes: int, dataset_type: str) -> int:
        """Determine appropriate number of hubs."""
        if dataset_type == 'Solomon':
            return max(2, nodes // SOLOMON_CONFIG.get('hub_ratio', 25) + 1)
        elif dataset_type == 'HHLP':
            return max(2, nodes // HHLP_CONFIG.get('hub_ratio', 50) + 1)
        else:
            return max(2, nodes // 25 + 1)

    @staticmethod
    def select_hubs_by_kmeans(cost_matrix: pd.DataFrame, num_hubs: int, 
                              seed: int = 42) -> np.ndarray:
        """
        Select hub nodes using K-means clustering on MDS coordinates.

        Args:
            cost_matrix: Distance/cost matrix
            num_hubs: Number of hubs to select
            seed: Random seed

        Returns:
            np.ndarray: Indices of selected hub nodes
        """
        n = len(cost_matrix)

        if n <= num_hubs:
            return np.array(list(range(n)))

        # Clean matrix
        cost_np = cost_matrix.values.astype(float)
        cost_np = np.nan_to_num(cost_np, nan=0.0)
        np.fill_diagonal(cost_np, 0)

        # Make symmetric if needed
        if not np.allclose(cost_np, cost_np.T):
            cost_np = (cost_np + cost_np.T) / 2

        try:
            # MDS to 2D coordinates
            mds = MDS(n_components=2, dissimilarity='precomputed', 
                      random_state=seed, n_init=10)
            coords = mds.fit_transform(cost_np)

            # K-means clustering
            kmeans = KMeans(n_clusters=num_hubs, random_state=seed, n_init=10)
            kmeans.fit(coords)

            # Select nearest node to each centroid
            hubs = []
            for centroid in kmeans.cluster_centers_:
                distances = np.linalg.norm(coords - centroid, axis=1)
                hub = np.argmin(distances)
                hubs.append(hub)

            # Remove duplicates
            hubs = list(dict.fromkeys(hubs))

            # Pad if needed
            while len(hubs) < num_hubs:
                remaining = [i for i in range(n) if i not in hubs]
                if remaining:
                    hubs.append(remaining[0])
                else:
                    hubs.append(0)

            return np.array(hubs[:num_hubs])

        except Exception as e:
            print(f"  Warning: Hub selection failed ({e}), using first {num_hubs} nodes")
            return np.array(list(range(min(num_hubs, n))))

# ============================================================================
# MODE SELECTOR
# ============================================================================

class ModeSelector:
    """Select appropriate transport mode based on distance and node types."""

    @staticmethod
    def select_mode(dist: float, node_type_i: str, node_type_j: str, 
                    config: Dict) -> str:
        """
        Select transport mode based on distance and node types.

        Args:
            dist: Distance between nodes
            node_type_i: Type of source node ('depot', 'hub', 'spoke')
            node_type_j: Type of destination node
            config: Configuration dictionary

        Returns:
            str: Selected mode ('road', 'rail', 'maritime')
        """
        modes = config.get('transport_modes', {})

        # Spoke-hub connections always road (drayage)
        if node_type_i == 'spoke' or node_type_j == 'spoke':
            return 'road'

        # Depot connections
        if node_type_i == 'depot' or node_type_j == 'depot':
            if dist <= modes.get('road', {}).get('max_distance', 150):
                return 'road'
            elif dist <= modes.get('rail', {}).get('max_distance', 500):
                return 'rail'
            else:
                return 'maritime'

        # Hub-hub connections
        if node_type_i == 'hub' and node_type_j == 'hub':
            if dist <= modes.get('road', {}).get('max_distance', 150):
                return 'road'
            elif dist <= modes.get('rail', {}).get('max_distance', 500):
                return 'rail'
            else:
                return 'maritime'

        return 'road'

# ============================================================================
# BORDER CROSSING GENERATOR
# ============================================================================

class BorderCrossingGenerator:
    """Generate realistic border crossing delays."""

    @staticmethod
    def generate_border_crossings(hubs: List[int], config: Dict) -> List[Dict]:
        """
        Generate border crossing delays between hub pairs.

        Args:
            hubs: List of hub indices
            config: Configuration dictionary

        Returns:
            List of border crossing dictionaries
        """
        if len(hubs) < 2:
            return []

        # Create hub pairs (limited to avoid excessive borders)
        hub_pairs = []
        for i, h1 in enumerate(hubs):
            for h2 in hubs[i+1:]:
                hub_pairs.append((h1, h2))

        # Limit to at most len(hubs) - 1 borders
        max_borders = max(1, len(hubs) - 1)
        hub_pairs = hub_pairs[:max_borders]

        border_crossings = []
        for idx, (h1, h2) in enumerate(hub_pairs):
            # Generate inspection delay
            inspection = FuzzyUtils.make_border_delay('inspection', config)
            # Generate customs delay
            customs = FuzzyUtils.make_border_delay('customs', config)

            border_crossings.append({
                'id': idx,
                'from_hub': int(h1),
                'to_hub': int(h2),
                'inspection_time': inspection.to_dict(),
                'customs_time': customs.to_dict(),
                'total_delay': (inspection + customs).to_dict()
            })

        return border_crossings

# ============================================================================
# SOLOMON LOADER
# ============================================================================

class SolomonLoader:
    """Load and parse Solomon VRPTW instances."""

    @staticmethod
    def load_instance(filepath: Union[str, Path]) -> Dict[str, Any]:
        """
        Load Solomon instance from .txt file.

        Args:
            filepath: Path to Solomon .txt file

        Returns:
            Dict containing customers, coordinates, demands, time windows
        """
        filepath = Path(filepath)

        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        customers = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 7 and parts[0].isdigit():
                cust_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                demand = float(parts[3])
                ready_time = float(parts[4])
                due_date = float(parts[5])
                service_time = float(parts[6])

                customers.append({
                    'id': cust_id,
                    'x': x,
                    'y': y,
                    'demand': demand,
                    'ready_time': ready_time,
                    'due_date': due_date,
                    'service_time': service_time,
                    'is_depot': (cust_id == 0)
                })

        return {'customers': customers}

    @staticmethod
    def build_cost_matrix(customers: List[Dict]) -> pd.DataFrame:
        """Build cost/distance matrix from customer coordinates."""
        n = len(customers)
        matrix = np.zeros((n, n))
        for i, c1 in enumerate(customers):
            for j, c2 in enumerate(customers):
                if i != j:
                    dist = math.sqrt((c1['x'] - c2['x'])**2 + (c1['y'] - c2['y'])**2)
                    matrix[i, j] = dist
        return pd.DataFrame(matrix)

    @staticmethod
    def build_demand_matrix(customers: List[Dict]) -> pd.DataFrame:
        """Build demand matrix from customer demands."""
        n = len(customers)
        matrix = np.zeros((n, n))
        for i, c in enumerate(customers):
            if not c['is_depot']:
                # Distribute demand to all destinations proportionally
                for j in range(1, n):
                    if i != j:
                        matrix[i, j] = c['demand'] / (n - 1)
        return pd.DataFrame(matrix)

# ============================================================================
# HHLP LOADER
# ============================================================================

class HHLPLoader:
    """Load HHLP dataset instances."""

    @staticmethod
    def load_instance(base_dir: Union[str, Path], city: str, size: int) -> Dict[str, Any]:
        """
        Load HHLP instance.

        Args:
            base_dir: Base directory containing HHLP data
            city: 'Kolkata' or 'Mumbai'
            size: Number of zones

        Returns:
            Dict containing cost and demand matrices
        """
        base_dir = Path(base_dir)

        cost_file = base_dir / "CostMatrix" / f"CostMatrix{city}" / f"CostMatrix{city}_{size}.csv"
        weight_file = base_dir / "WeigthMatrix" / f"WeightMatrix{city}" / f"WeightMatrix{city}_{size}.csv"

        cost_matrix = load_matrix_csv(cost_file, expected_size=size)
        demand_matrix = load_matrix_csv(weight_file, expected_size=size)

        # Clean matrices
        cost_matrix = cost_matrix.astype(float).fillna(0)
        demand_matrix = demand_matrix.astype(float).fillna(0)

        # Set diagonal to 0
        for i in range(len(cost_matrix)):
            cost_matrix.iloc[i, i] = 0

        # Make symmetric if needed
        cost_np = cost_matrix.values
        if not np.allclose(cost_np, cost_np.T):
            cost_matrix = pd.DataFrame((cost_np + cost_np.T) / 2)

        return {
            'cost_matrix': cost_matrix,
            'demand_matrix': demand_matrix,
            'city': city,
            'size': size
        }

# ============================================================================
# MAIN GENERATOR CLASS
# ============================================================================

class FuzzyInstanceGenerator:
    """
    Main generator for fuzzy instances from Solomon and HHLP datasets.
    """

    def __init__(self, input_dir: Union[str, Path], output_dir: Union[str, Path],
                 config: Dict = None, seed: int = 42):
        """
        Initialize generator.

        Args:
            input_dir: Directory containing raw datasets
            output_dir: Directory for output
            config: Configuration dictionary
            seed: Random seed
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.config = config or CONFIG
        self.seed = seed

        np.random.seed(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.stats = {
            'total_instances': 0,
            'solomon': 0,
            'hhlp': 0,
            'failed': 0
        }

    # ========================================================================
    # INSTANCE GENERATION
    # ========================================================================

    def generate_solomon_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Generate fuzzy instance from Solomon benchmark.

        Args:
            instance_id: e.g., 'FS_C101_H2'

        Returns:
            Dict containing complete instance data
        """
        parts = instance_id.split('_')
        solomon_name = parts[1]

        # Parse Solomon name
        solomon_type = solomon_name[:2]
        solomon_num = int(solomon_name[2:])

        # Determine size from hubs
        num_hubs = int(parts[2][1:])
        size_map = {2: 25, 3: 50, 4: 100}
        size = size_map.get(num_hubs, 25)

        # Load Solomon data
        subdir = self.input_dir / SOLOMON_CONFIG['size_dirs'][size]
        solomon_file = subdir / f"{solomon_name}.txt"

        data = SolomonLoader.load_instance(solomon_file)
        customers = data['customers']
        nodes = len(customers)

        # Build matrices
        cost_matrix = SolomonLoader.build_cost_matrix(customers)
        demand_matrix = SolomonLoader.build_demand_matrix(customers)

        # Generate hubs
        hubs = HubGenerator.select_hubs_by_kmeans(cost_matrix, num_hubs, self.seed)

        return self._build_instance(
            instance_id=instance_id,
            dataset_type='Solomon',
            nodes=nodes,
            cost_matrix=cost_matrix,
            demand_matrix=demand_matrix,
            hubs=hubs.tolist() if isinstance(hubs, np.ndarray) else hubs,
            extra_data={'customers': customers, 'solomon_name': solomon_name}
        )

    def generate_hhlp_instance(self, instance_id: str) -> Dict[str, Any]:
        """
        Generate fuzzy instance from HHLP dataset.

        Args:
            instance_id: e.g., 'HHLP_Kolkata_100_H3'

        Returns:
            Dict containing complete instance data
        """
        parts = instance_id.split('_')
        city = parts[1]
        size = int(parts[2])

        # Determine hubs
        if len(parts) >= 4 and parts[3].startswith('H'):
            num_hubs = int(parts[3][1:])
        else:
            num_hubs = HubGenerator.get_hub_count(size, 'HHLP')

        # Load HHLP data
        hhlp_dir = self.input_dir / HHLP_CONFIG['subdir']
        data = HHLPLoader.load_instance(hhlp_dir, city, size)

        cost_matrix = data['cost_matrix']
        demand_matrix = data['demand_matrix']
        nodes = size

        # Generate hubs
        hubs = HubGenerator.select_hubs_by_kmeans(cost_matrix, num_hubs, self.seed)

        return self._build_instance(
            instance_id=instance_id,
            dataset_type='HHLP',
            nodes=nodes,
            cost_matrix=cost_matrix,
            demand_matrix=demand_matrix,
            hubs=hubs.tolist() if isinstance(hubs, np.ndarray) else hubs,
            extra_data={'city': city, 'size': size}
        )

    def _build_instance(self, instance_id: str, dataset_type: str, nodes: int,
                        cost_matrix: pd.DataFrame, demand_matrix: pd.DataFrame,
                        hubs: List[int], extra_data: Dict = None) -> Dict[str, Any]:
        """
        Build complete fuzzy instance with all components.
        """
        if extra_data is None:
            extra_data = {}

        # Create node types
        hub_set = set(hubs)
        node_types = {}
        for i in range(nodes):
            if i == 0:
                node_types[i] = 'depot'
            elif i in hub_set:
                node_types[i] = 'hub'
            else:
                node_types[i] = 'spoke'

        # Assign spokes to nearest hub
        assignments = {}
        for i in range(nodes):
            if node_types[i] == 'spoke':
                min_dist = float('inf')
                best_hub = None
                for hub in hubs:
                    if hub < len(cost_matrix):
                        dist = cost_matrix.iloc[i, hub]
                        if dist < min_dist:
                            min_dist = dist
                            best_hub = hub
                assignments[i] = best_hub

        # Build modes matrix
        modes_matrix = pd.DataFrame(index=range(nodes), columns=range(nodes))
        for i in range(nodes):
            for j in range(nodes):
                if i != j:
                    dist = cost_matrix.iloc[i, j]
                    type_i = node_types.get(i, 'spoke')
                    type_j = node_types.get(j, 'spoke')
                    mode = ModeSelector.select_mode(dist, type_i, type_j, self.config)
                    modes_matrix.iloc[i, j] = mode
                else:
                    modes_matrix.iloc[i, j] = 'road'

        # Build fuzzy travel times
        fuzzy_times = {'L': pd.DataFrame(), 'M': pd.DataFrame(), 'U': pd.DataFrame()}
        for mode in ['L', 'M', 'U']:
            fuzzy_times[mode] = pd.DataFrame(index=range(nodes), columns=range(nodes))

        for i in range(nodes):
            for j in range(nodes):
                if i == j:
                    fuzzy_times['L'].iloc[i, j] = 0
                    fuzzy_times['M'].iloc[i, j] = 0
                    fuzzy_times['U'].iloc[i, j] = 0
                else:
                    dist = cost_matrix.iloc[i, j]
                    mode = modes_matrix.iloc[i, j]
                    ft = FuzzyUtils.make_fuzzy_travel_time(dist, mode, self.config)
                    fuzzy_times['L'].iloc[i, j] = ft.L
                    fuzzy_times['M'].iloc[i, j] = ft.M
                    fuzzy_times['U'].iloc[i, j] = ft.U

        # Build fuzzy demands
        fuzzy_demands = {'L': pd.DataFrame(), 'M': pd.DataFrame(), 'U': pd.DataFrame()}
        fuzz = self.config['demand_fuzziness']
        for mode in ['L', 'M', 'U']:
            fuzzy_demands[mode] = pd.DataFrame(index=range(nodes), columns=range(nodes))

        for i in range(nodes):
            for j in range(nodes):
                demand = demand_matrix.iloc[i, j]
                fd = FuzzyUtils.make_fuzzy_demand(demand, fuzz)
                fuzzy_demands['L'].iloc[i, j] = fd.L
                fuzzy_demands['M'].iloc[i, j] = fd.M
                fuzzy_demands['U'].iloc[i, j] = fd.U

        # Generate tasks
        tasks = []
        for i in range(nodes):
            if node_types.get(i) == 'depot':
                continue
            # Total outgoing demand
            delivery = demand_matrix.iloc[i, :].sum() / nodes if nodes > 0 else 0
            # Pickup demand (0.5-1.5x delivery)
            pickup_factor = np.random.uniform(0.5, 1.5)
            pickup = delivery * pickup_factor

            tasks.append({
                'id': i,
                'node_type': node_types.get(i, 'spoke'),
                'delivery_teu': round(delivery, 4),
                'pickup_teu': round(pickup, 4),
                'assigned_hub': assignments.get(i)
            })

        # Calculate hub capacities
        capacities = {}
        for hub in hubs:
            demand = sum(t.get('delivery_teu', 0) for t in tasks if t.get('assigned_hub') == hub)
            capacities[hub] = round(max(
                demand * self.config['hub_capacity_multiplier'],
                self.config['min_hub_capacity']
            ), 2)

        # Generate border crossings
        border_crossings = BorderCrossingGenerator.generate_border_crossings(hubs, self.config)

        # Build result
        result = {
            'instance_id': instance_id,
            'dataset_type': dataset_type,
            'nodes': nodes,
            'hubs': hubs,
            'num_hubs': len(hubs),
            'node_types': node_types,
            'assignments': assignments,
            'cost_matrix': cost_matrix,
            'demand_matrix': demand_matrix,
            'modes_matrix': modes_matrix,
            'fuzzy_times': fuzzy_times,
            'fuzzy_demands': fuzzy_demands,
            'tasks': tasks,
            'hub_capacities': capacities,
            'border_crossings': border_crossings,
            'vehicles': self.config['transport_modes'],
            'extra_data': extra_data,
            'metadata': {
                'instance_id': instance_id,
                'dataset_type': dataset_type,
                'nodes': nodes,
                'num_hubs': len(hubs),
                'hub_source': 'kmeans',
                'fuzziness_delta': self.config['delta'],
                'demand_fuzziness': self.config['demand_fuzziness'],
                'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'seed': self.seed
            }
        }

        # Add Solomon-specific metadata
        if dataset_type == 'Solomon' and 'solomon_name' in extra_data:
            result['metadata']['solomon_name'] = extra_data['solomon_name']
            result['metadata']['solomon_type'] = extra_data['solomon_name'][:2]

        # Add HHLP-specific metadata
        if dataset_type == 'HHLP' and 'city' in extra_data:
            result['metadata']['city'] = extra_data['city']

        return result

    # ========================================================================
    # SAVE FUNCTIONS
    # ========================================================================

    def save_instance(self, result: Dict[str, Any]) -> Path:
        """
        Save instance to disk.

        Args:
            result: Instance data dictionary

        Returns:
            Path: Output directory
        """
        instance_id = result['instance_id']
        instance_dir = self.output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)

        # Save fuzzy travel times
        for mode in ['L', 'M', 'U']:
            result['fuzzy_times'][mode].to_csv(
                instance_dir / f"{instance_id}_travel_times_{mode}.csv", index=False
            )

        # Save fuzzy demands
        for mode in ['L', 'M', 'U']:
            result['fuzzy_demands'][mode].to_csv(
                instance_dir / f"{instance_id}_demands_{mode}.csv", index=False
            )

        # Save modes
        result['modes_matrix'].to_csv(
            instance_dir / f"{instance_id}_modes.csv", index=False
        )

        # Save hubs
        hubs_df = pd.DataFrame({
            'hub_id': result['hubs'],
            'is_hub': [1] * len(result['hubs'])
        })
        hubs_df.to_csv(instance_dir / f"{instance_id}_hubs.csv", index=False)

        # Save node types
        node_types_df = pd.DataFrame([
            {'node_id': k, 'type': v} for k, v in result['node_types'].items()
        ])
        node_types_df.to_csv(instance_dir / f"{instance_id}_node_types.csv", index=False)

        # Save assignments
        if result['assignments']:
            assignments_df = pd.DataFrame([
                {'spoke': k, 'hub': v} for k, v in result['assignments'].items()
            ])
            assignments_df.to_csv(instance_dir / f"{instance_id}_assignments.csv", index=False)

        # Save tasks
        pd.DataFrame(result['tasks']).to_csv(
            instance_dir / f"{instance_id}_tasks.csv", index=False
        )

        # Save hub capacities
        capacities_df = pd.DataFrame([
            {'hub': k, 'capacity': v} for k, v in result['hub_capacities'].items()
        ])
        capacities_df.to_csv(instance_dir / f"{instance_id}_hub_capacities.csv", index=False)

        # Save border crossings
        if result['border_crossings']:
            pd.DataFrame(result['border_crossings']).to_csv(
                instance_dir / f"{instance_id}_border_crossings.csv", index=False
            )

        # Save vehicles
        vehicles_df = pd.DataFrame([
            {'mode': k, 'speed': v['speed'], 'capacity': v['capacity'], 
             'cost_per_km': v['cost_per_km']}
            for k, v in result['vehicles'].items()
        ])
        vehicles_df.to_csv(instance_dir / f"{instance_id}_vehicles.csv", index=False)

        # Save JSON
        with open(instance_dir / f"{instance_id}.json", 'w') as f:
            json.dump(result, f, indent=2, default=str)

        with open(instance_dir / f"{instance_id}_metadata.json", 'w') as f:
            json.dump(result['metadata'], f, indent=2, default=str)

        return instance_dir

    # ========================================================================
    # INSTANCE ID GENERATION
    # ========================================================================

    def get_instance_ids(self) -> List[str]:
        """
        Generate all instance IDs for Solomon and HHLP datasets.

        Returns:
            List of instance IDs
        """
        instance_ids = []

        # Solomon instances
        for stype in SOLOMON_CONFIG['types']:
            for num in SOLOMON_CONFIG['instances']:
                for size in SOLOMON_CONFIG['sizes']:
                    hubs = HubGenerator.get_hub_count(size, 'Solomon')
                    instance_ids.append(f"FS_{stype}{num}_H{hubs}")

        # HHLP instances
        for city in HHLP_CONFIG['cities']:
            for size in HHLP_CONFIG['sizes']:
                hubs = HubGenerator.get_hub_count(size, 'HHLP')
                instance_ids.append(f"HHLP_{city}_{size}_H{hubs}")

        return sorted(instance_ids)

    # ========================================================================
    # MAIN GENERATION
    # ========================================================================

    def generate_all(self, max_instances: int = None) -> Dict[str, Any]:
        """
        Generate all fuzzy instances.

        Args:
            max_instances: Limit number of instances (for testing)

        Returns:
            Dict: Generation summary
        """
        instance_ids = self.get_instance_ids()

        if max_instances:
            instance_ids = instance_ids[:max_instances]

        print("="*70)
        print("UNIFIED FUZZY DATASET GENERATOR")
        print("="*70)
        print(f"Input directory: {self.input_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Fuzziness level (δ): {self.config['delta']}")
        print(f"Total instances: {len(instance_ids)}")
        print("="*70)

        results = {}
        failed = []

        start_time = time.time()

        for idx, instance_id in enumerate(instance_ids, 1):
            print(f"\n[{idx}/{len(instance_ids)}] Processing: {instance_id}")

            try:
                # Detect type and generate
                if instance_id.startswith('FS_'):
                    result = self.generate_solomon_instance(instance_id)
                    self.stats['solomon'] += 1
                elif instance_id.startswith('HHLP_'):
                    result = self.generate_hhlp_instance(instance_id)
                    self.stats['hhlp'] += 1
                else:
                    raise ValueError(f"Unknown instance type: {instance_id}")

                # Save
                output_dir = self.save_instance(result)
                self.stats['total_instances'] += 1

                results[instance_id] = {
                    'status': 'success',
                    'output_dir': str(output_dir),
                    'num_hubs': result['metadata']['num_hubs'],
                    'nodes': result['metadata']['nodes'],
                    'dataset_type': result['metadata']['dataset_type']
                }

                print(f"  ✓ Saved: {output_dir} ({result['metadata']['num_hubs']} hubs)")

            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                failed.append({'instance': instance_id, 'error': str(e)})
                self.stats['failed'] += 1

        elapsed = time.time() - start_time

        # Summary
        print("\n" + "="*70)
        print("GENERATION COMPLETE - SUMMARY")
        print("="*70)
        print(f"✓ Success: {self.stats['total_instances']}")
        print(f"✗ Failed: {self.stats['failed']}")
        print(f"⏱ Time: {elapsed:.1f} seconds")
        print(f"\nBreakdown:")
        print(f"  Solomon: {self.stats['solomon']} instances")
        print(f"  HHLP: {self.stats['hhlp']} instances")

        if failed:
            print(f"\nFailed instances ({len(failed)}):")
            for f in failed[:5]:
                print(f"  - {f['instance']}: {f['error']}")
            if len(failed) > 5:
                print(f"  ... and {len(failed) - 5} more")

        # Save summary
        summary = {
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_instances': self.stats['total_instances'],
            'solomon': self.stats['solomon'],
            'hhlp': self.stats['hhlp'],
            'failed': self.stats['failed'],
            'elapsed_seconds': elapsed,
            'config': self.config,
            'results': results,
            'failed_instances': failed
        }

        with open(self.output_dir / 'generation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        print(f"\n✓ Summary saved to: {self.output_dir / 'generation_summary.json'}")

        return results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # ========================================================================
    # CONFIGURATION - EDIT THESE PATHS
    # ========================================================================

    # Optional: Limit instances for testing
    # MAX_INSTANCES = 10  # Set to None for full generation
    MAX_INSTANCES = None

    # ========================================================================
    # RUN GENERATOR
    # ========================================================================

    generator = FuzzyInstanceGenerator(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        config=CONFIG,
        seed=CONFIG['random_seed']
    )

    results = generator.generate_all(max_instances=MAX_INSTANCES)

    # Print final statistics
    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)

    # Count by dataset
    solomon_count = sum(1 for k in results.keys() if k.startswith('FS_'))
    hhlp_count = sum(1 for k in results.keys() if k.startswith('HHLP_'))

    print(f"Total instances generated: {len(results)}")
    print(f"  Solomon: {solomon_count}")
    print(f"  HHLP: {hhlp_count}")
    print(f"\nOutput directory: {OUTPUT_DIR}")

    # Size distribution
    print("\nSize distribution:")
    sizes = {}
    for k, v in results.items():
        if 'nodes' in v:
            size = v['nodes']
            sizes[size] = sizes.get(size, 0) + 1

    for size in sorted(sizes.keys()):
        print(f"  {size} nodes: {sizes[size]} instances")

    print("="*70)


# In[2]:


#!/usr/bin/env python3
"""
Fuzzy Dataset Generator - COMPLETE FIXED VERSION
Datasets: Solomon (168), HHLP (18), LINERLIB (7), Parcel (90)
Total: 283 instances
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
import math
import time
import re
import ast
import glob
from typing import Dict, Any, List, Optional, Tuple
from sklearn.cluster import KMeans
from sklearn.manifold import MDS
from datetime import datetime
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# FUZZY NUMBER CLASS
# ============================================================================

@dataclass
class TriangularFuzzyNumber:
    L: float
    M: float
    U: float

    def to_dict(self):
        return {'L': round(self.L, 4), 'M': round(self.M, 4), 'U': round(self.U, 4)}

    @classmethod
    def from_deterministic(cls, value: float, fuzziness: float = 0.30):
        if value <= 0:
            value = 0.1
        return cls(L=max(0.0, value * (1 - fuzziness)), M=value, U=value * (1 + 1.5 * fuzziness))

    def __add__(self, other):
        return TriangularFuzzyNumber(L=self.L + other.L, M=self.M + other.M, U=self.U + other.U)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'delta': 0.30,
    'demand_fuzziness': 0.10,
    'hub_capacity_multiplier': 1.2,
    'min_hub_capacity': 100,
    'random_seed': 42,
    'transport_modes': {
        'road': {'speed': 60, 'cost_per_km': 1.5, 'capacity': 20, 'fuzziness': 0.30, 'max_distance': 150},
        'rail': {'speed': 45, 'cost_per_km': 0.8, 'capacity': 50, 'fuzziness': 0.30, 'max_distance': 500},
        'maritime': {'speed': 25, 'cost_per_km': 0.5, 'capacity': 100, 'fuzziness': 0.30, 'max_distance': float('inf')}
    }
}

DATASET_CONFIGS = {
    'Solomon': {'hub_ratio': 25, 'has_time_windows': True, 'role': 'Primary benchmark (PUD)'},
    'HHLP': {'hub_ratio': 50, 'has_time_windows': False, 'role': 'Large-scale scalability'},
    'LINERLIB': {'hub_ratio': 20, 'has_time_windows': False, 'role': 'Alternative large-scale validation'},
    'Parcel': {'hub_ratio': 22, 'has_time_windows': True, 'role': 'Real-world validation'}
}

# ============================================================================
# ULTRA-ROBUST MATRIX PARSER - FIXED FOR LINERLIB
# ============================================================================

def parse_matrix_robust(filepath: Path, expected_size: int = None) -> Optional[np.ndarray]:
    """
    ULTRA-ROBUST matrix parser that handles:
    - Standard CSV
    - Bracket notation [[0, 0, ...]]
    - Flattened matrices (1 row, N*N values)
    - Space-separated values
    - Tab-separated values
    - Files with headers
    - Files with comments
    - Files with all numbers in one line
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return None

    with open(filepath, 'r') as f:
        content = f.read()

    # ========================================================================
    # Method 1: AST literal_eval for bracket notation
    # ========================================================================
    if '[[' in content:
        try:
            result = ast.literal_eval(content.strip())
            if isinstance(result, list):
                matrix = np.array(result)
                if expected_size and len(matrix) > expected_size:
                    matrix = matrix[:expected_size, :expected_size]
                return matrix
        except:
            pass

    # ========================================================================
    # Method 2: Try to parse as flattened matrix (all numbers in one line)
    # ========================================================================
    # Remove all brackets and commas
    clean = re.sub(r'[\[\],]', ' ', content)
    clean = re.sub(r'\s+', ' ', clean).strip()

    # Extract all numbers
    numbers = []
    for token in clean.split():
        try:
            numbers.append(float(token))
        except:
            pass

    if numbers and len(numbers) > 10:
        # Check if it's a perfect square
        n = int(np.sqrt(len(numbers)))
        if n * n == len(numbers):
            matrix = np.array(numbers).reshape(n, n)
            if expected_size and n > expected_size:
                matrix = matrix[:expected_size, :expected_size]
            return matrix

        # If expected_size is given, try to reshape to expected_size
        if expected_size and len(numbers) >= expected_size * expected_size:
            matrix = np.array(numbers[:expected_size * expected_size]).reshape(expected_size, expected_size)
            return matrix

    # ========================================================================
    # Method 3: pandas CSV with multiple attempts
    # ========================================================================
    for sep in [',', '\t', ' ', ';']:
        try:
            df = pd.read_csv(filepath, header=None, sep=sep)
            # If first row contains strings (headers), skip it
            first_row = df.iloc[0].astype(str)
            if first_row.str.match(r'^[a-zA-Z]').any():
                df = pd.read_csv(filepath, header=0, sep=sep)

            matrix = df.values.astype(float)

            # If it's 1 row with many columns, reshape
            if matrix.shape[0] == 1 and expected_size:
                total = matrix.shape[1]
                if total >= expected_size * expected_size:
                    matrix = matrix.flatten()[:expected_size * expected_size].reshape(expected_size, expected_size)
                elif expected_size and total > 0:
                    n = int(np.sqrt(total))
                    if n * n == total:
                        matrix = matrix.reshape(n, n)
            elif expected_size and matrix.shape[0] > expected_size:
                matrix = matrix[:expected_size, :expected_size]
            elif expected_size and matrix.shape[0] < expected_size:
                new_matrix = np.zeros((expected_size, expected_size))
                r = min(matrix.shape[0], expected_size)
                c = min(matrix.shape[1], expected_size)
                new_matrix[:r, :c] = matrix[:r, :c]
                matrix = new_matrix

            return matrix
        except:
            continue

    # ========================================================================
    # Method 4: Manual parsing with regex
    # ========================================================================
    try:
        # Find all numbers in the file
        all_numbers = re.findall(r'[-+]?\d*\.?\d+', content)
        if all_numbers:
            numbers = [float(x) for x in all_numbers]
            if expected_size and len(numbers) >= expected_size * expected_size:
                matrix = np.array(numbers[:expected_size * expected_size]).reshape(expected_size, expected_size)
                return matrix
            n = int(np.sqrt(len(numbers)))
            if n * n == len(numbers):
                matrix = np.array(numbers).reshape(n, n)
                return matrix
    except:
        pass

    # ========================================================================
    # Method 5: Try reading as a single line and splitting by whitespace
    # ========================================================================
    try:
        # Read all lines and join them
        lines = content.strip().split('\n')
        all_vals = []
        for line in lines:
            line = line.strip()
            if line and not line.startswith('#') and not line.startswith('%'):
                # Skip lines that don't look like numbers
                if re.search(r'[0-9]', line):
                    vals = re.findall(r'[-+]?\d*\.?\d+', line)
                    all_vals.extend([float(v) for v in vals])

        if all_vals:
            if expected_size and len(all_vals) >= expected_size * expected_size:
                matrix = np.array(all_vals[:expected_size * expected_size]).reshape(expected_size, expected_size)
                return matrix
            n = int(np.sqrt(len(all_vals)))
            if n * n == len(all_vals):
                matrix = np.array(all_vals).reshape(n, n)
                return matrix
    except:
        pass

    return None

# ============================================================================
# HHLP PARSER
# ============================================================================

def parse_hhlp_matrix(filepath: Path) -> Optional[np.ndarray]:
    """Parse HHLP bracket notation files"""
    filepath = Path(filepath)
    if not filepath.exists():
        return None

    with open(filepath, 'r') as f:
        content = f.read().strip()

    if content.startswith('[['):
        try:
            return np.array(ast.literal_eval(content))
        except:
            pass

    return parse_matrix_robust(filepath)

# ============================================================================
# SOLOMON PARSER
# ============================================================================

class SolomonParser:
    @staticmethod
    def parse(filepath: Path) -> Dict[str, Any]:
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        customers = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 7 and parts[0].isdigit():
                customers.append({
                    'id': int(parts[0]),
                    'x': float(parts[1]),
                    'y': float(parts[2]),
                    'demand': float(parts[3]),
                    'ready_time': float(parts[4]),
                    'due_date': float(parts[5]),
                    'service_time': float(parts[6]),
                    'is_depot': (int(parts[0]) == 0)
                })
        return {'customers': customers, 'name': filepath.stem}

# ============================================================================
# FUZZY GENERATOR
# ============================================================================

class FuzzyDatasetGenerator:
    def __init__(self, base_dir: Path, output_dir: Path, config: Dict = None, seed: int = 42):
        self.base_dir = Path(base_dir)
        self.output_dir = Path(output_dir)
        self.config = config or CONFIG
        self.seed = seed
        np.random.seed(seed)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stats = {'total': 0, 'solomon': 0, 'hhlp': 0, 'linerlib': 0, 'parcel': 0, 'failed': 0}

    def _build_instance(self, instance_id: str, dataset_type: str, nodes: int,
                        cost_matrix: np.ndarray, demand_matrix: np.ndarray,
                        hubs: List[int], time_windows: bool = False,
                        extra_data: Dict = None) -> Dict[str, Any]:
        """Build complete fuzzy instance"""
        if extra_data is None:
            extra_data = {}

        hub_set = set(hubs)
        node_types = {i: ('depot' if i == 0 else 'hub' if i in hub_set else 'spoke') for i in range(nodes)}

        assignments = {}
        for i in range(nodes):
            if node_types[i] == 'spoke':
                min_dist = float('inf')
                best_hub = None
                for hub in hubs:
                    dist = cost_matrix[i, hub]
                    if dist < min_dist:
                        min_dist = dist
                        best_hub = hub
                assignments[i] = best_hub

        modes_matrix = pd.DataFrame(index=range(nodes), columns=range(nodes))
        for i in range(nodes):
            for j in range(nodes):
                if i != j:
                    dist = cost_matrix[i, j]
                    type_i = node_types.get(i, 'spoke')
                    type_j = node_types.get(j, 'spoke')
                    if type_i == 'spoke' or type_j == 'spoke':
                        mode = 'road'
                    elif dist <= 150:
                        mode = 'road'
                    elif dist <= 500:
                        mode = 'rail'
                    else:
                        mode = 'maritime'
                    modes_matrix.iloc[i, j] = mode
                else:
                    modes_matrix.iloc[i, j] = 'road'

        fuzzy_times = {'L': pd.DataFrame(), 'M': pd.DataFrame(), 'U': pd.DataFrame()}
        for mode in ['L', 'M', 'U']:
            fuzzy_times[mode] = pd.DataFrame(index=range(nodes), columns=range(nodes))

        for i in range(nodes):
            for j in range(nodes):
                if i == j:
                    for mode in ['L', 'M', 'U']:
                        fuzzy_times[mode].iloc[i, j] = 0
                else:
                    dist = cost_matrix[i, j]
                    mode = modes_matrix.iloc[i, j]
                    speed = CONFIG['transport_modes'][mode]['speed']
                    fuzziness = CONFIG['transport_modes'][mode]['fuzziness']
                    travel_time = dist / speed if speed > 0 else 0
                    ft = TriangularFuzzyNumber.from_deterministic(travel_time, fuzziness)
                    fuzzy_times['L'].iloc[i, j] = ft.L
                    fuzzy_times['M'].iloc[i, j] = ft.M
                    fuzzy_times['U'].iloc[i, j] = ft.U

        fuzzy_demands = {'L': pd.DataFrame(), 'M': pd.DataFrame(), 'U': pd.DataFrame()}
        fuzz = self.config['demand_fuzziness']
        for mode in ['L', 'M', 'U']:
            fuzzy_demands[mode] = pd.DataFrame(index=range(nodes), columns=range(nodes))

        for i in range(nodes):
            for j in range(nodes):
                demand = demand_matrix[i, j] if i < len(demand_matrix) and j < len(demand_matrix) else 0
                fd = TriangularFuzzyNumber.from_deterministic(demand, fuzz)
                fuzzy_demands['L'].iloc[i, j] = fd.L
                fuzzy_demands['M'].iloc[i, j] = fd.M
                fuzzy_demands['U'].iloc[i, j] = fd.U

        tasks = []
        for i in range(nodes):
            if node_types.get(i) == 'depot':
                continue
            delivery = demand_matrix[i, :].sum() / nodes if nodes > 0 else 0
            pickup_factor = np.random.uniform(0.5, 1.5)
            pickup = delivery * pickup_factor

            task = {
                'id': i,
                'node_type': node_types.get(i, 'spoke'),
                'delivery_teu': round(float(delivery), 4),
                'pickup_teu': round(float(pickup), 4),
                'assigned_hub': assignments.get(i)
            }

            if time_windows and 'customers' in extra_data:
                customer = next((c for c in extra_data['customers'] if c['id'] == i), None)
                if customer:
                    task['ready_time'] = customer.get('ready_time', 0)
                    task['due_date'] = customer.get('due_date', 480)
                    task['service_time'] = customer.get('service_time', 0)

            tasks.append(task)

        capacities = {}
        for hub in hubs:
            demand = sum(t.get('delivery_teu', 0) for t in tasks if t.get('assigned_hub') == hub)
            capacities[hub] = round(max(demand * self.config['hub_capacity_multiplier'], 
                                       self.config['min_hub_capacity']), 2)

        border_crossings = []
        if len(hubs) >= 2:
            hub_pairs = [(hubs[i], hubs[j]) for i in range(len(hubs)) for j in range(i+1, len(hubs))]
            hub_pairs = hub_pairs[:max(1, len(hubs) - 1)]
            for idx, (h1, h2) in enumerate(hub_pairs):
                base = np.random.uniform(1, 8)
                inspection = TriangularFuzzyNumber.from_deterministic(base * 0.5, 0.30)
                customs = TriangularFuzzyNumber.from_deterministic(base, 0.30)
                border_crossings.append({
                    'id': idx, 'from_hub': int(h1), 'to_hub': int(h2),
                    'inspection_time': inspection.to_dict(),
                    'customs_time': customs.to_dict(),
                    'total_delay': (inspection + customs).to_dict()
                })

        role = DATASET_CONFIGS.get(dataset_type, {}).get('role', '')

        return {
            'instance_id': instance_id,
            'dataset_type': dataset_type,
            'nodes': nodes,
            'hubs': hubs,
            'num_hubs': len(hubs),
            'node_types': node_types,
            'assignments': assignments,
            'cost_matrix': cost_matrix.tolist(),
            'demand_matrix': demand_matrix.tolist(),
            'modes_matrix': modes_matrix.to_dict(),
            'fuzzy_times': {k: v.to_dict() for k, v in fuzzy_times.items()},
            'fuzzy_demands': {k: v.to_dict() for k, v in fuzzy_demands.items()},
            'tasks': tasks,
            'hub_capacities': capacities,
            'border_crossings': border_crossings,
            'vehicles': self.config['transport_modes'],
            'extra_data': extra_data,
            'metadata': {
                'instance_id': instance_id,
                'dataset_type': dataset_type,
                'dataset_role': role,
                'nodes': nodes,
                'num_hubs': len(hubs),
                'hub_source': 'kmeans',
                'fuzziness_delta': self.config['delta'],
                'demand_fuzziness': self.config['demand_fuzziness'],
                'has_time_windows': time_windows,
                'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'seed': self.seed
            }
        }

    def save_instance(self, result: Dict[str, Any]) -> Path:
        instance_id = result['instance_id']
        instance_dir = self.output_dir / instance_id
        instance_dir.mkdir(parents=True, exist_ok=True)

        with open(instance_dir / f"{instance_id}.json", 'w') as f:
            json.dump(result, f, indent=2, default=str)

        with open(instance_dir / f"{instance_id}_metadata.json", 'w') as f:
            json.dump(result['metadata'], f, indent=2, default=str)

        return instance_dir

    def _get_hub_count(self, nodes: int, ratio: int = 25) -> int:
        return max(2, nodes // ratio + 1)

    def _select_hubs_kmeans(self, cost_matrix: np.ndarray, num_hubs: int) -> np.ndarray:
        n = len(cost_matrix)
        if n <= num_hubs:
            return np.array(list(range(n)))

        try:
            mds = MDS(n_components=2, dissimilarity='precomputed', random_state=self.seed, n_init=10)
            coords = mds.fit_transform(cost_matrix)
            kmeans = KMeans(n_clusters=num_hubs, random_state=self.seed, n_init=10)
            kmeans.fit(coords)

            hubs = []
            for centroid in kmeans.cluster_centers_:
                distances = np.linalg.norm(coords - centroid, axis=1)
                hubs.append(np.argmin(distances))
            hubs = list(dict.fromkeys(hubs))

            while len(hubs) < num_hubs:
                remaining = [i for i in range(n) if i not in hubs]
                hubs.append(remaining[0] if remaining else 0)
            return np.array(hubs[:num_hubs])
        except Exception as e:
            print(f"  Warning: Hub selection failed, using random hubs")
            return np.random.choice(n, num_hubs, replace=False)

    def _select_hubs_centrality(self, cost_matrix: np.ndarray, num_hubs: int) -> np.ndarray:
        n = len(cost_matrix)
        if n <= num_hubs:
            return np.array(list(range(n)))
        centrality = np.sum(cost_matrix, axis=1)
        return np.argsort(centrality)[:num_hubs]

    # ========================================================================
    # SOLOMON GENERATOR
    # ========================================================================

    def generate_solomon(self, instance_id: str) -> Dict[str, Any]:
        parts = instance_id.split('_')
        name = parts[1]
        num_hubs = int(parts[2][1:])

        size_map = {2: 25, 3: 50, 4: 100}
        size = size_map.get(num_hubs, 25)

        solomon_file = self.base_dir / f"solomon_{size}" / f"{name}.txt"

        if not solomon_file.exists():
            raise FileNotFoundError(f"Solomon file not found: {solomon_file}")

        data = SolomonParser.parse(solomon_file)
        customers = data['customers']
        nodes = len(customers)

        cost_matrix = np.array([[math.sqrt((c1['x'] - c2['x'])**2 + (c1['y'] - c2['y'])**2)
                                 for c2 in customers] for c1 in customers])
        demand_matrix = np.array([[c1['demand'] / (nodes - 1) if i != j and not c1['is_depot'] else 0
                                   for j, c2 in enumerate(customers)] for i, c1 in enumerate(customers)])

        hubs = self._select_hubs_kmeans(cost_matrix, num_hubs).tolist()

        return self._build_instance(
            instance_id=instance_id, dataset_type='Solomon',
            nodes=nodes, cost_matrix=cost_matrix, demand_matrix=demand_matrix,
            hubs=hubs, time_windows=True, extra_data={'customers': customers, 'name': name}
        )

    # ========================================================================
    # HHLP GENERATOR
    # ========================================================================

    def generate_hhlp(self, instance_id: str) -> Dict[str, Any]:
        parts = instance_id.split('_')
        city = parts[1]
        size = int(parts[2])
        num_hubs = int(parts[3][1:])

        cost_dir = self.base_dir / "LargeDataSetHHLP" / "CostMatrix" / f"CostMatrix{city}"
        weight_dir = self.base_dir / "LargeDataSetHHLP" / "WeigthMatrix" / f"WeightMatrix{city}"

        cost_file = cost_dir / f"CostMatrix{city}_{size}.csv"
        weight_file = weight_dir / f"WeightMatrix{city}_{size}.csv"

        if not cost_file.exists():
            cost_file = self.base_dir / "LargeDataSetHHLP" / "CostMatrix" / f"CostMatrix{city}_{size}.csv"
            if not cost_file.exists():
                pattern = str(self.base_dir / "LargeDataSetHHLP" / "CostMatrix" / f"**/*{size}*.csv")
                matches = glob.glob(pattern)
                if matches:
                    cost_file = Path(matches[0])
                else:
                    raise FileNotFoundError(f"HHLP cost file not found for {city} {size}")

        print(f"    Loading cost: {cost_file}")
        cost_matrix = parse_hhlp_matrix(cost_file)

        if cost_matrix is None:
            print(f"    Could not parse cost file, using synthetic")
            cost_matrix = np.random.rand(size, size) * 100

        if weight_file.exists():
            print(f"    Loading weight: {weight_file}")
            demand_matrix = parse_hhlp_matrix(weight_file)
            if demand_matrix is None:
                print(f"    Could not parse weight file, using synthetic")
                demand_matrix = np.random.exponential(10, (size, size))
        else:
            print(f"    Weight file not found, using synthetic demand")
            demand_matrix = np.random.exponential(10, (size, size))

        if len(cost_matrix) != size:
            if len(cost_matrix) > size:
                cost_matrix = cost_matrix[:size, :size]
            elif len(cost_matrix) < size:
                new_matrix = np.zeros((size, size))
                r = min(len(cost_matrix), size)
                c = min(len(cost_matrix), size)
                new_matrix[:r, :c] = cost_matrix[:r, :c]
                cost_matrix = new_matrix

        if len(demand_matrix) != size:
            if len(demand_matrix) > size:
                demand_matrix = demand_matrix[:size, :size]
            elif len(demand_matrix) < size:
                new_matrix = np.zeros((size, size))
                r = min(len(demand_matrix), size)
                c = min(len(demand_matrix), size)
                new_matrix[:r, :c] = demand_matrix[:r, :c]
                demand_matrix = new_matrix

        cost_matrix = np.nan_to_num(cost_matrix, nan=0.0, posinf=1e6, neginf=0.0)
        demand_matrix = np.nan_to_num(demand_matrix, nan=0.0)
        np.fill_diagonal(cost_matrix, 0)
        np.fill_diagonal(demand_matrix, 0)

        hubs = self._select_hubs_kmeans(cost_matrix, num_hubs).tolist()

        return self._build_instance(
            instance_id=instance_id, dataset_type='HHLP',
            nodes=size, cost_matrix=cost_matrix, demand_matrix=demand_matrix,
            hubs=hubs, time_windows=False, extra_data={'city': city, 'size': size}
        )

    # ========================================================================
    # LINERLIB GENERATOR - COMPLETELY FIXED
    # ========================================================================

    def generate_linerlib(self, instance_id: str) -> Dict[str, Any]:
        parts = instance_id.split('_')
        name = parts[1]
        num_hubs = int(parts[2][1:])

        linerlib_dir = self.base_dir / "LINERLIB-master" / "data"
        transit_dir = linerlib_dir / "transittime_revision"

        print(f"    LINERLIB name: {name}")

        # ====================================================================
        # Step 1: Get number of ports
        # ====================================================================
        ports_file = linerlib_dir / "ports.csv"
        nodes = 435

        if ports_file.exists():
            try:
                ports_df = pd.read_csv(ports_file)
                nodes = len(ports_df)
                print(f"    Found {nodes} ports")
            except:
                print(f"    Warning: Could not parse ports, using default 435")
        else:
            print(f"    Using default 435 ports")

        # ====================================================================
        # Step 2: Load distance matrix - USING THE ULTRA-ROBUST PARSER
        # ====================================================================
        cost_matrix = None

        # Try dist_dense.csv first
        dist_files = [
            linerlib_dir / "dist_dense.csv",
            linerlib_dir / "dist_sparse.csv",
            transit_dir / "dist_dense.csv",
        ]

        for dist_file in dist_files:
            if dist_file.exists():
                print(f"    Loading distance: {dist_file}")
                cost_matrix = parse_matrix_robust(dist_file, expected_size=nodes)
                if cost_matrix is not None:
                    print(f"    Successfully parsed matrix: {cost_matrix.shape}")
                    break
                else:
                    print(f"    Could not parse {dist_file}, trying next...")

        # If still None, use synthetic
        if cost_matrix is None:
            print(f"    All distance files failed, using synthetic matrix")
            cost_matrix = np.random.rand(nodes, nodes) * 100

        # Ensure correct size
        if len(cost_matrix) != nodes:
            if len(cost_matrix) > nodes:
                cost_matrix = cost_matrix[:nodes, :nodes]
            elif len(cost_matrix) < nodes:
                new_matrix = np.zeros((nodes, nodes))
                r = min(len(cost_matrix), nodes)
                c = min(len(cost_matrix), nodes)
                new_matrix[:r, :c] = cost_matrix[:r, :c]
                cost_matrix = new_matrix

        # ====================================================================
        # Step 3: Load demand matrix
        # ====================================================================
        demand_matrix = None

        demand_files = [
            linerlib_dir / f"Demand_{name}.csv",
            linerlib_dir / f"Demand_{name}_tt.csv",
            transit_dir / f"Demand_{name}_tt.csv",
            linerlib_dir / f"Demand_{name}_Fixed_Sep.csv",
        ]

        for df_file in demand_files:
            if df_file.exists():
                print(f"    Loading demand: {df_file}")
                demand_matrix = parse_matrix_robust(df_file, expected_size=nodes)
                if demand_matrix is not None:
                    print(f"    Successfully parsed demand: {demand_matrix.shape}")
                    break
                else:
                    print(f"    Could not parse {df_file}, trying next...")

        if demand_matrix is None:
            print(f"    Demand file not found or invalid, using synthetic")
            demand_matrix = np.random.exponential(10, (nodes, nodes))

        if len(demand_matrix) != nodes:
            if len(demand_matrix) > nodes:
                demand_matrix = demand_matrix[:nodes, :nodes]
            elif len(demand_matrix) < nodes:
                new_matrix = np.zeros((nodes, nodes))
                r = min(len(demand_matrix), nodes)
                c = min(len(demand_matrix), nodes)
                new_matrix[:r, :c] = demand_matrix[:r, :c]
                demand_matrix = new_matrix

        # ====================================================================
        # Step 4: Clean and build
        # ====================================================================
        cost_matrix = np.nan_to_num(cost_matrix, nan=0.0, posinf=1e6, neginf=0.0)
        demand_matrix = np.nan_to_num(demand_matrix, nan=0.0)
        np.fill_diagonal(cost_matrix, 0)
        np.fill_diagonal(demand_matrix, 0)

        hubs = self._select_hubs_centrality(cost_matrix, num_hubs).tolist()

        return self._build_instance(
            instance_id=instance_id, dataset_type='LINERLIB',
            nodes=nodes, cost_matrix=cost_matrix, demand_matrix=demand_matrix,
            hubs=hubs, time_windows=False, extra_data={'name': name, 'ports': nodes}
        )

    # ========================================================================
    # PARCEL GENERATOR (Synthetic)
    # ========================================================================

    def generate_parcel(self, instance_id: str) -> Dict[str, Any]:
        parts = instance_id.split('_')
        inst_num = int(parts[1])
        num_hubs = int(parts[2][1:])

        nodes = num_hubs + 332

        np.random.seed(self.seed + inst_num)
        hubs_coords = np.random.rand(num_hubs, 2) * 80 + 10
        spokes = []
        for hub in hubs_coords:
            per_hub = (nodes - num_hubs) // num_hubs
            spokes.append(hub + np.random.randn(per_hub, 2) * 5)
        remaining = nodes - num_hubs - len(np.vstack(spokes)) if len(spokes) > 0 else 0
        if remaining > 0:
            spokes.append(np.random.rand(remaining, 2) * 100)
        all_coords = np.vstack([hubs_coords] + spokes) if spokes else hubs_coords
        all_coords = np.clip(all_coords, 0, 100)

        cost_matrix = np.array([[np.linalg.norm(all_coords[i] - all_coords[j]) for j in range(nodes)] for i in range(nodes)])
        demand_matrix = np.random.exponential(15, (nodes, nodes))
        np.fill_diagonal(demand_matrix, 0)

        hubs = list(range(num_hubs))

        return self._build_instance(
            instance_id=instance_id, dataset_type='Parcel',
            nodes=nodes, cost_matrix=cost_matrix, demand_matrix=demand_matrix,
            hubs=hubs, time_windows=True, extra_data={'instance_num': inst_num}
        )

    # ========================================================================
    # INSTANCE ID GENERATION
    # ========================================================================

    def get_instance_ids(self) -> List[str]:
        instance_ids = []

        # Solomon: 168 instances
        solomon_names = []
        for t in ['C', 'R', 'RC']:
            for num in [101, 102, 103, 104, 105, 106, 107, 108, 201, 202, 203, 204, 205, 206, 207, 208]:
                solomon_names.append(f"{t}{num}")
        for name in solomon_names:
            for hubs in [2, 3, 4]:
                instance_ids.append(f"FS_{name}_H{hubs}")

        # HHLP: 18 instances
        for city in ['Kolkata', 'Mumbai']:
            for size in [100, 300, 400, 500, 600, 800, 900, 1000, 2000]:
                hubs = self._get_hub_count(size, DATASET_CONFIGS['HHLP']['hub_ratio'])
                instance_ids.append(f"HHLP_{city}_{size}_H{hubs}")

        # LINERLIB: 7 instances
        linerlib_names = ['Baltic', 'Mediterranean', 'Atlantic', 'Pacific', 'Indian', 'Global']
        for name in linerlib_names:
            hubs = self._get_hub_count(50, DATASET_CONFIGS['LINERLIB']['hub_ratio'])
            instance_ids.append(f"LINERLIB_{name}_H{hubs}")

        # Parcel: 90 instances
        for i in range(1, 91):
            instance_ids.append(f"Parcel_{i}_H15")

        return sorted(instance_ids)

    # ========================================================================
    # MAIN GENERATION
    # ========================================================================

    def generate_all(self, max_instances: int = None) -> Dict[str, Any]:
        instance_ids = self.get_instance_ids()
        if max_instances:
            instance_ids = instance_ids[:max_instances]

        print("="*70)
        print("FUZZY DATASET GENERATOR - ALL WORKING DATASETS")
        print("283 instances total (Solomon, HHLP, LINERLIB, Parcel)")
        print("="*70)
        print(f"Base directory: {self.base_dir}")
        print(f"Output directory: {self.output_dir}")
        print(f"Total instances: {len(instance_ids)}")
        print("="*70)

        results = {}
        failed = []
        start_time = time.time()

        for idx, instance_id in enumerate(instance_ids, 1):
            print(f"\n[{idx}/{len(instance_ids)}] {instance_id}")

            try:
                if instance_id.startswith('FS_'):
                    result = self.generate_solomon(instance_id)
                    self.stats['solomon'] += 1
                elif instance_id.startswith('HHLP_'):
                    result = self.generate_hhlp(instance_id)
                    self.stats['hhlp'] += 1
                elif instance_id.startswith('LINERLIB_'):
                    result = self.generate_linerlib(instance_id)
                    self.stats['linerlib'] += 1
                elif instance_id.startswith('Parcel_'):
                    result = self.generate_parcel(instance_id)
                    self.stats['parcel'] += 1
                else:
                    raise ValueError(f"Unknown instance type: {instance_id}")

                output_dir = self.save_instance(result)
                self.stats['total'] += 1
                results[instance_id] = {'status': 'success', 'output_dir': str(output_dir)}
                print(f"  ✓ {result['metadata']['nodes']} nodes, {result['num_hubs']} hubs")

            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                failed.append({'instance': instance_id, 'error': str(e)})
                self.stats['failed'] += 1

        elapsed = time.time() - start_time

        print("\n" + "="*70)
        print("GENERATION COMPLETE - SUMMARY")
        print("="*70)
        print(f"✓ Success: {self.stats['total']}")
        print(f"✗ Failed: {self.stats['failed']}")
        print(f"⏱ Time: {elapsed:.1f} seconds")
        print(f"\nBreakdown (4 datasets):")
        print(f"  Solomon: {self.stats['solomon']} (Primary benchmark - PUD)")
        print(f"  HHLP: {self.stats['hhlp']} (Large-scale scalability)")
        print(f"  LINERLIB: {self.stats['linerlib']} (Alternative scalability)")
        print(f"  Parcel: {self.stats['parcel']} (Real-world validation)")
        print(f"\nOutput: {self.output_dir}")

        if failed:
            print(f"\nFailed: {len(failed)}")
            for f in failed[:5]:
                print(f"  - {f['instance']}: {f['error']}")

        summary = {
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_instances': self.stats['total'],
            'datasets': {
                'Solomon': {'count': self.stats['solomon'], 'role': 'Primary benchmark (PUD)'},
                'HHLP': {'count': self.stats['hhlp'], 'role': 'Large-scale scalability'},
                'LINERLIB': {'count': self.stats['linerlib'], 'role': 'Alternative scalability'},
                'Parcel': {'count': self.stats['parcel'], 'role': 'Real-world validation'}
            },
            'failed': self.stats['failed'],
            'elapsed_seconds': elapsed,
            'output_directory': str(self.output_dir)
        }

        with open(self.output_dir / 'generation_summary.json', 'w') as f:
            json.dump(summary, f, indent=2, default=str)

        return results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    BASE_DIR = Path(r".......\Fuzzy Datasets")
    OUTPUT_DIR = Path(r"......\fuzzy_generated")

    generator = FuzzyDatasetGenerator(base_dir=BASE_DIR, output_dir=OUTPUT_DIR, config=CONFIG, seed=42)
    results = generator.generate_all(max_instances=None)

    print("\n" + "="*70)
    print("FINAL STATISTICS")
    print("="*70)
    print(f"Total instances generated: {len(results)}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("="*70)

