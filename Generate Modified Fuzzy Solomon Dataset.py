
## Code 1: Generate Modified Fuzzy Solomon Dataset

The following script generates **168 fuzzy Solomon instances** from the standard Solomon VRPTW benchmark. It converts deterministic VRPTW instances into fuzzy intermodal hub-and-spoke problems with graduated hub assignment (2 hubs for 25 customers, 3 hubs for 50 customers, 4 hubs for 100 customers).

```python
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
        non_depot = [c for c in self.customers if not c['is_depot']]
        
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
        self.spokes = [c['id'] for c in non_depot if c['id'] not in self.hubs]
        
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
        if num_customers <= 30:
            return 2
        elif num_customers <= 70:
            return 3
        else:
            return 4
    
    def generate(self) -> Dict:
        num_customers = self.solomon['num_customers']
        num_hubs = self.get_num_hubs(num_customers)
        
        transformer = HubSpokeTransformer(self.solomon['customers'], num_hubs)
        transformer.select_hubs()
        
        node_dict = {c['id']: c for c in self.solomon['customers']}
        depot = next(c for c in self.solomon['customers'] if c['is_depot'])
        
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


def main():
    input_dirs = {
        25: r"path/to/solomon_25",
        50: r"path/to/solomon_50",
        100: r"path/to/solomon_100"
    }
    
    output_dir = r"path/to/fuzzy_solomon_modified"
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("FUZZY SOLOMON INSTANCE GENERATOR")
    print("168 instances: 56 files × 3 sizes (one instance per file)")
    print("="*60)
    
    total = 0
    for size, path in input_dirs.items():
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
            print(f"  ✓ {solomon['name']} | {size}cust | {instance['num_hubs']}hubs")
    
    print(f"COMPLETE: {total} instances generated")


if __name__ == "__main__":
    np.random.seed(42)
    rnd.seed(42)
    main()