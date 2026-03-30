#!/usr/bin/env python3
"""
Speed-only deconfliction benchmark — main entry point.

Runs all classical algorithms on the same 1000-agent scenario with
dynamic wind perturbations and outputs a comparison table.

Usage:
    python main.py [--agents N] [--seed S] [--no-wind] [--verbose]
"""

from __future__ import annotations

import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import argparse
import os
import json
import numpy as np
from datetime import datetime

import config as cfg
from environment import Environment
from algorithms import ALL_ALGORITHMS
from benchmark import run_benchmark, format_results_table, export_to_excel, RunMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Speed-only deconfliction benchmark")
    p.add_argument("--agents", type=int, default=cfg.NUM_AGENTS, help="Number of agents")
    p.add_argument("--seed", type=int, default=cfg.SEED, help="Random seed")
    p.add_argument("--no-wind", action="store_true", help="Disable wind perturbations")
    p.add_argument("--verbose", "-v", action="store_true", help="Print progress during runs")
    p.add_argument("--output", type=str, default=None, help="Save results JSON to file")
    return p.parse_args()


def main():
    args = parse_args()

    # apply overrides
    cfg.NUM_AGENTS = args.agents
    cfg.SEED = args.seed
    if args.no_wind:
        cfg.WIND_ENABLED = False

    print(f"=== Speed-Only Deconfliction Benchmark ===")
    print(f"Agents: {cfg.NUM_AGENTS}  |  Seed: {cfg.SEED}  |  Wind: {'ON' if cfg.WIND_ENABLED else 'OFF'}")
    print(f"World: {cfg.WORLD_SIZE:.0f}m  |  DT: {cfg.DT}s  |  Max time: {cfg.MAX_TIME}s")
    print(f"Preferred speed: {cfg.PREFERRED_SPEED} m/s  |  Collision dist: {cfg.COLLISION_DIST} m")
    print()

    # create environment and generate scenario
    env = Environment(seed=cfg.SEED)
    env.generate_agents(cfg.NUM_AGENTS)
    print(f"Generated {env.n} agents with random OD pairs.")
    avg_route = float(np.mean(env.route_lengths))
    print(f"Average route length: {avg_route:.0f} m")
    print()

    # run each algorithm
    results: list[RunMetrics] = []
    for name, AlgoClass in ALL_ALGORITHMS.items():
        print(f"Running {name}...")
        algo = AlgoClass()
        metrics = run_benchmark(env, algo, verbose=args.verbose)
        results.append(metrics)
        print(f"  -> {metrics.agents_arrived}/{cfg.NUM_AGENTS} arrived, "
              f"{metrics.total_collisions} collisions, "
              f"algo time {metrics.computation_time:.3f}s, "
              f"total {metrics.total_sim_time:.1f}s")
        print()

    # print comparison table
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(format_results_table(results))

    # save to file
    output_path = args.output or os.path.join("results", f"benchmark_{datetime.now():%Y%m%d_%H%M%S}.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "agents": cfg.NUM_AGENTS,
            "seed": cfg.SEED,
            "wind": cfg.WIND_ENABLED,
            "world_size": cfg.WORLD_SIZE,
            "dt": cfg.DT,
            "max_time": cfg.MAX_TIME,
            "preferred_speed": cfg.PREFERRED_SPEED,
            "collision_dist": cfg.COLLISION_DIST,
        },
        "results": [
            {
                "algorithm": r.algorithm,
                "complexity": r.complexity,
                "total_collisions": r.total_collisions,
                "collision_rate": r.collision_rate,
                "success_rate": r.success_rate,
                "makespan": r.makespan,
                "avg_arrival_time": r.avg_arrival_time,
                "avg_speed_utilization": r.avg_speed_utilization,
                "computation_time": r.computation_time,
                "total_sim_time": r.total_sim_time,
                "steps": r.steps,
            }
            for r in results
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Excel export
    excel_path = output_path.replace(".json", ".xlsx")
    export_to_excel(results, excel_path)
    print(f"Excel saved to {excel_path}")


if __name__ == "__main__":
    main()
