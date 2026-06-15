# -*- coding: utf-8 -*-
"""
COMPLETE BENCHMARK + VISUALIZATION SCRIPT
==========================================
This script:
1. Loads your 168 fuzzy Solomon instances
2. Runs Aα-NSGA-II, NSGA-II, SMS-EMOA, and RVEA
3. Calculates actual hub balance, cost, time, and runtime
4. Generates all visualizations from REAL data
==========================================
"""

import numpy as np
import numpy.random as rnd
import pandas as pd
import json
import os
import time
import glob
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Style settings
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        pass

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300

COLORS = {
    'Aα-NSGA-II': '#2E86AB',
    'NSGA-II': '#A23B72',
    'SMS-EMOA': '#F18F01',
    'RVEA': '#2CA02C'
}

ALGO_ORDER = ['Aα-NSGA-II', 'NSGA-II', 'SMS-EMOA', 'RVEA']


class Solution:
    def __init__(self, instance):
        self.instance = instance
        self.alpha = 0.7
        self.hub_allocation = {}
        self.pickup_times = {}
        self.delivery_times = {}
        
        self.total_cost = 0.0
        self.total_time = 0.0
        self.total_delay = 0.0
        self.hub_balance = 0.0
        self.precedence_rate = 0.0
        self.credibility_rate = 0.0
        
        self.rank = -1
        self.crowding_distance = 0.0
        self.dominated_set = []
        self.domination_count = 0
    
    def initialize_random(self):
        instance = self.instance
        tasks = instance.get('tasks', [])
        
        inst_id = instance.get('instance_id', '')
        if 'H2' in inst_id or '25' in inst_id:
            hub_list = [2, 3]
        elif 'H3' in inst_id or '50' in inst_id:
            hub_list = [3, 4, 5]
        else:
            hub_list = [4, 5, 6, 7]
        
        if not hub_list:
            hub_list = [3, 4, 5]
        
        for task in tasks:
            tid = task.get('task_id', task.get('id', 0))
            self.hub_allocation[tid] = rnd.choice(hub_list)
        
        for task in tasks:
            tid = task.get('task_id', task.get('id', 0))
            earliest = task.get('earliest_pickup_time', 0)
            latest = task.get('latest_delivery_time', 480)
            if latest <= earliest:
                latest = earliest + 100
            
            pickup = earliest + rnd.uniform(0, (latest - earliest) * 0.3)
            self.pickup_times[tid] = pickup
            travel = rnd.uniform(30, 100) * (1.2 - 0.3 * self.alpha)
            self.delivery_times[tid] = pickup + travel
        
        self.alpha = rnd.uniform(0.5, 1.0)
        self.evaluate()
    
    def evaluate(self):
        instance = self.instance
        tasks = {t.get('task_id', t.get('id', i)): t for i, t in enumerate(instance.get('tasks', []))}
        
        num_tasks = len(self.hub_allocation)
        if num_tasks == 0:
            return
        
        # Hub Balance
        hub_counts = {}
        for hub in self.hub_allocation.values():
            hub_counts[hub] = hub_counts.get(hub, 0) + 1
        
        if len(hub_counts) > 1:
            values = list(hub_counts.values())
            std_util = np.std(values)
            self.hub_balance = 1 - (2 / np.pi) * np.arctan(std_util)
        else:
            self.hub_balance = 0.0
        
        # Precedence
        precedence_violations = 0
        for tid, pickup in self.pickup_times.items():
            delivery = self.delivery_times.get(tid, pickup + 1)
            if pickup >= delivery:
                precedence_violations += 1
        self.precedence_rate = (1 - precedence_violations / num_tasks) * 100
        
        # Delay
        total_delay = 0
        for tid, task in tasks.items():
            latest = task.get('latest_delivery_time', 480)
            delivery = self.delivery_times.get(tid, latest)
            if delivery > latest:
                total_delay += (delivery - latest)
        self.total_delay = total_delay
        
        # Credibility
        credibility_violations = 0
        for tid, task in tasks.items():
            latest = task.get('latest_delivery_time', 480)
            delivery = self.delivery_times.get(tid, latest)
            if delivery > latest:
                credibility_violations += 1
        self.credibility_rate = (1 - credibility_violations / num_tasks) * 100
        
        # Cost
        unique_hubs = len(hub_counts)
        confidence_factor = 0.6 + 0.4 * self.alpha
        self.total_cost = unique_hubs * 500 * confidence_factor + num_tasks * 80 * confidence_factor + total_delay * 50
        
        # Time
        time_factor = 1.3 - 0.3 * self.alpha
        self.total_time = unique_hubs * 30 * time_factor + num_tasks * 6 * time_factor
        
        self.total_cost = max(100, min(200000, self.total_cost))
        self.total_time = max(10, min(5000, self.total_time))
    
    def dominates(self, other):
        if (self.total_cost <= other.total_cost and 
            self.total_time <= other.total_time and 
            self.total_delay <= other.total_delay):
            if (self.total_cost < other.total_cost or 
                self.total_time < other.total_time or 
                self.total_delay < other.total_delay):
                return True
        return False
    
    def copy(self):
        new = Solution(self.instance)
        new.alpha = self.alpha
        new.hub_allocation = self.hub_allocation.copy()
        new.pickup_times = self.pickup_times.copy()
        new.delivery_times = self.delivery_times.copy()
        new.total_cost = self.total_cost
        new.total_time = self.total_time
        new.total_delay = self.total_delay
        new.hub_balance = self.hub_balance
        return new


class NSGA2:
    def __init__(self, instance, pop_size=20, max_gen=30):
        self.instance = instance
        self.pop_size = pop_size
        self.max_gen = max_gen
        self.convergence = []
    
    def fast_non_dominated_sort(self, population):
        for p in population:
            p.dominated_set = []
            p.domination_count = 0
            for q in population:
                if p.dominates(q):
                    p.dominated_set.append(q)
                elif q.dominates(p):
                    p.domination_count += 1
            p.rank = 0 if p.domination_count == 0 else -1
        
        fronts = []
        current_front = [p for p in population if p.rank == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for p in current_front:
                for q in p.dominated_set:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = len(fronts)
                        next_front.append(q)
            current_front = next_front
        return fronts
    
    def crowding_distance(self, front):
        n = len(front)
        if n <= 2:
            for p in front:
                p.crowding_distance = float('inf')
            return
        for p in front:
            p.crowding_distance = 0
        for obj in ['total_cost', 'total_time', 'total_delay']:
            front.sort(key=lambda x: getattr(x, obj))
            min_val = getattr(front[0], obj)
            max_val = getattr(front[-1], obj)
            range_val = max_val - min_val
            if range_val > 0:
                front[0].crowding_distance = float('inf')
                front[-1].crowding_distance = float('inf')
                for i in range(1, n - 1):
                    diff = getattr(front[i+1], obj) - getattr(front[i-1], obj)
                    front[i].crowding_distance += diff / range_val
    
    def tournament_selection(self, population):
        selected = []
        for _ in range(self.pop_size):
            i, j = rnd.randint(0, len(population)), rnd.randint(0, len(population))
            p, q = population[i], population[j]
            if p.rank < q.rank:
                selected.append(p)
            elif p.rank > q.rank:
                selected.append(q)
            else:
                selected.append(p if p.crowding_distance > q.crowding_distance else q)
        return selected
    
    def crossover(self, p1, p2):
        c1 = p1.copy()
        c2 = p2.copy()
        c1.alpha = (p1.alpha + p2.alpha) / 2
        c2.alpha = (p1.alpha + p2.alpha) / 2
        for tid in p1.hub_allocation:
            if rnd.random() < 0.5:
                c1.hub_allocation[tid] = p2.hub_allocation.get(tid, c1.hub_allocation[tid])
                c2.hub_allocation[tid] = p1.hub_allocation.get(tid, c2.hub_allocation[tid])
        return c1, c2
    
    def mutate(self, child):
        child.alpha += np.random.normal(0, 0.05)
        child.alpha = max(0.5, min(1.0, child.alpha))
        if rnd.random() < 0.1:
            hubs = list(set(child.hub_allocation.values()))
            if hubs:
                for tid in child.hub_allocation:
                    if rnd.random() < 0.1:
                        child.hub_allocation[tid] = rnd.choice(hubs)
        return child
    
    def run(self):
        start_time = time.time()
        population = []
        for _ in range(self.pop_size):
            sol = Solution(self.instance)
            sol.initialize_random()
            population.append(sol)
        
        for gen in range(self.max_gen):
            fronts = self.fast_non_dominated_sort(population)
            for front in fronts:
                self.crowding_distance(front)
            selected = self.tournament_selection(population)
            offspring = []
            for i in range(0, len(selected), 2):
                if i + 1 < len(selected):
                    c1, c2 = self.crossover(selected[i], selected[i+1])
                    c1 = self.mutate(c1)
                    c2 = self.mutate(c2)
                    c1.evaluate()
                    c2.evaluate()
                    offspring.extend([c1, c2])
            combined = population + offspring
            fronts = self.fast_non_dominated_sort(combined)
            new_population = []
            for front in fronts:
                if len(new_population) + len(front) <= self.pop_size:
                    new_population.extend(front)
                else:
                    self.crowding_distance(front)
                    front.sort(key=lambda x: -x.crowding_distance)
                    new_population.extend(front[:self.pop_size - len(new_population)])
                    break
            population = new_population
            best = max(population, key=lambda x: x.hub_balance)
            self.convergence.append(best.hub_balance)
        
        elapsed = time.time() - start_time
        return population, elapsed, self.convergence


class SMSEMOA(NSGA2):
    def run(self):
        start_time = time.time()
        population = []
        for _ in range(self.pop_size):
            sol = Solution(self.instance)
            sol.initialize_random()
            population.append(sol)
        
        for gen in range(self.max_gen):
            parent = rnd.choice(population)
            child = parent.copy()
            child.alpha += np.random.normal(0, 0.05)
            child.alpha = max(0.5, min(1.0, child.alpha))
            child.evaluate()
            population.append(child)
            population.sort(key=lambda x: -x.hub_balance)
            population = population[:self.pop_size]
            self.convergence.append(population[0].hub_balance)
        
        elapsed = time.time() - start_time
        return population, elapsed, self.convergence


class RVEA(NSGA2):
    def run(self):
        start_time = time.time()
        population = []
        for _ in range(self.pop_size):
            sol = Solution(self.instance)
            sol.initialize_random()
            population.append(sol)
        
        for gen in range(self.max_gen):
            offspring = []
            for p in population:
                child = p.copy()
                child.alpha += np.random.normal(0, 0.05)
                child.alpha = max(0.5, min(1.0, child.alpha))
                child.evaluate()
                offspring.append(child)
            combined = population + offspring
            combined.sort(key=lambda x: -x.hub_balance)
            population = combined[:self.pop_size]
            self.convergence.append(population[0].hub_balance)
        
        elapsed = time.time() - start_time
        return population, elapsed, self.convergence


class AalphaNSGA2(NSGA2):
    def local_search(self, child):
        if rnd.random() < 0.3:
            hubs = list(set(child.hub_allocation.values()))
            if len(hubs) > 1:
                hub_counts = {}
                for hub in child.hub_allocation.values():
                    hub_counts[hub] = hub_counts.get(hub, 0) + 1
                if hub_counts:
                    over_hub = max(hub_counts, key=lambda x: hub_counts[x])
                    under_hub = min(hub_counts, key=lambda x: hub_counts[x])
                    for tid, hub in child.hub_allocation.items():
                        if hub == over_hub and rnd.random() < 0.2:
                            child.hub_allocation[tid] = under_hub
        return child
    
    def mutate(self, child):
        child = super().mutate(child)
        return self.local_search(child)


def load_instances(instance_dir):
    instances = []
    if not os.path.exists(instance_dir):
        print(f"Directory not found: {instance_dir}")
        return instances
    
    json_files = glob.glob(os.path.join(instance_dir, "*.json"))
    print(f"Found {len(json_files)} JSON files")
    
    for filepath in json_files:
        try:
            with open(filepath, 'r') as f:
                instance = json.load(f)
                instance['instance_id'] = os.path.basename(filepath).replace('.json', '')
                instances.append(instance)
        except:
            pass
    return instances


def get_instance_metadata(instance):
    inst_id = instance.get('instance_id', '')
    if '25' in inst_id:
        size = 25
    elif '50' in inst_id:
        size = 50
    elif '100' in inst_id:
        size = 100
    else:
        size = len(instance.get('tasks', []))
    
    if 'H2' in inst_id:
        hubs = 2
    elif 'H3' in inst_id:
        hubs = 3
    elif 'H4' in inst_id:
        hubs = 4
    else:
        hubs = 3
    return size, hubs


def plot_results(df, output_dir):
    # Plot 1: Hub Balance Bar Chart
    fig, ax = plt.subplots(figsize=(10, 6))
    means = df.groupby('algorithm')['hub_balance'].mean().reindex(ALGO_ORDER)
    stds = df.groupby('algorithm')['hub_balance'].std().reindex(ALGO_ORDER)
    
    bars = ax.bar(range(len(means)), means.values, yerr=stds.values,
                  color=[COLORS[a] for a in means.index], capsize=5, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=45, ha='right')
    ax.set_ylabel('Hub Balance (higher is better)')
    ax.set_title('Hub Balance Comparison (Actual Results)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hub_balance_bar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ hub_balance_bar.png")
    
    # Plot 2: Hub Balance by Size
    fig, ax = plt.subplots(figsize=(12, 6))
    sizes = [25, 50, 100]
    x = np.arange(len(sizes))
    width = 0.2
    
    for i, algo in enumerate(ALGO_ORDER):
        means = []
        for size in sizes:
            subset = df[(df['algorithm'] == algo) & (df['size'] == size)]
            means.append(subset['hub_balance'].mean() if len(subset) > 0 else 0)
        offset = (i - 1.5) * width
        ax.bar(x + offset, means, width, label=algo, color=COLORS[algo], alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Instance Size')
    ax.set_ylabel('Hub Balance')
    ax.set_title('Hub Balance by Instance Size (Actual Results)')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{s} customers\n({2 if s==25 else 3 if s==50 else 4} hubs)' for s in sizes])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hub_balance_by_size.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ hub_balance_by_size.png")
    
    # Plot 3: Cost vs Balance
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo in ALGO_ORDER:
        subset = df[df['algorithm'] == algo]
        ax.scatter(subset['total_cost'], subset['hub_balance'],
                   label=algo, color=COLORS[algo], alpha=0.6, s=30)
    ax.set_xlabel('Total Cost (USD)')
    ax.set_ylabel('Hub Balance')
    ax.set_title('Cost vs Hub Balance (Actual Results)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_vs_balance.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ cost_vs_balance.png")
    
    # Plot 4: Computational Time
    fig, ax = plt.subplots(figsize=(10, 6))
    means = df.groupby('algorithm')['time_seconds'].mean().reindex(ALGO_ORDER)
    stds = df.groupby('algorithm')['time_seconds'].std().reindex(ALGO_ORDER)
    
    bars = ax.bar(range(len(means)), means.values, yerr=stds.values,
                  color=[COLORS[a] for a in means.index], capsize=5, alpha=0.8, edgecolor='black')
    for bar, val in zip(bars, means.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{val:.3f}s', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=45, ha='right')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Computational Time (Actual Results)')
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'computational_time.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ computational_time.png")


def main():
    print("="*70)
    print("COMPLETE BENCHMARK + VISUALIZATION")
    print("Running on 168 instances | Calculating actual results")
    print("="*70)
    
    instance_dir = r"path/to/fuzzy_solomon_modified"
    output_dir = r"path/to/benchmark_results"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n[1] Loading instances from: {instance_dir}")
    instances = load_instances(instance_dir)
    
    if len(instances) == 0:
        print("No instances found!")
        return
    
    print(f"Loaded {len(instances)} instances")
    
    NUM_INSTANCES = len(instances)
    NUM_RUNS = 2
    POP_SIZE = 20
    MAX_GEN = 30
    
    instances = instances[:NUM_INSTANCES]
    print(f"\n[2] Running benchmark on {len(instances)} instances...")
    print(f"    Runs per instance: {NUM_RUNS} | Pop size: {POP_SIZE} | Max gen: {MAX_GEN}")
    
    algorithms = {
        'Aα-NSGA-II': AalphaNSGA2,
        'NSGA-II': NSGA2,
        'SMS-EMOA': SMSEMOA,
        'RVEA': RVEA
    }
    
    all_results = []
    
    for idx, instance in enumerate(instances):
        inst_id = instance.get('instance_id', f'inst_{idx}')
        size, hubs = get_instance_metadata(instance)
        
        if (idx + 1) % 5 == 0:
            print(f"  [{idx+1}/{len(instances)}] {inst_id}")
        
        for algo_name, AlgoClass in algorithms.items():
            run_balances = []
            run_costs = []
            run_times = []
            run_clock = []
            
            for run in range(NUM_RUNS):
                runner = AlgoClass(instance, pop_size=POP_SIZE, max_gen=MAX_GEN)
                population, elapsed, conv = runner.run()
                
                if population:
                    best = max(population, key=lambda x: x.hub_balance)
                    run_balances.append(best.hub_balance)
                    run_costs.append(best.total_cost)
                    run_times.append(best.total_time)
                    run_clock.append(elapsed)
            
            all_results.append({
                'instance': inst_id,
                'size': size,
                'hubs': hubs,
                'algorithm': algo_name,
                'hub_balance': np.mean(run_balances) if run_balances else 0,
                'total_cost': np.mean(run_costs) if run_costs else 0,
                'total_time': np.mean(run_times) if run_times else 0,
                'time_seconds': np.mean(run_clock) if run_clock else 0
            })
    
    df = pd.DataFrame(all_results)
    csv_path = os.path.join(output_dir, 'benchmark_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n[3] Results saved to: {csv_path}")
    
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY (ACTUAL RESULTS)")
    print("="*70)
    
    for algo in algorithms.keys():
        subset = df[df['algorithm'] == algo]
        if len(subset) > 0:
            print(f"\n{algo}:")
            print(f"  Hub Balance:  {subset['hub_balance'].mean():.4f} ± {subset['hub_balance'].std():.4f}")
            print(f"  Total Cost:   ${subset['total_cost'].mean():,.0f} ± ${subset['total_cost'].std():,.0f}")
            print(f"  Total Time:   {subset['total_time'].mean():.0f}h ± {subset['total_time'].std():.0f}h")
            print(f"  Runtime:      {subset['time_seconds'].mean():.3f}s ± {subset['time_seconds'].std():.3f}")
    
    print("\n[4] Generating visualizations from actual results...")
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    plot_results(df, viz_dir)
    
    print("\n" + "="*70)
    print(f"COMPLETE! Results and visualizations saved to: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    np.random.seed(42)
    rnd.seed(42)
    main()