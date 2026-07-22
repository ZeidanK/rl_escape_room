"""Extract final numerical results for README."""
import csv
import json

# Matched comparison
with open("storage/experiments/final/sarsa_vs_q_learning_matched.json") as f:
    data = json.load(f)

print("=== MATCHED COMPARISON ===")
for algo in ["sarsa", "q_learning"]:
    results = data["matched_comparison"][algo]
    srs = [r["success_rate"] for r in results]
    rets = [r["mean_return"] for r in results]
    steps = [r["mean_successful_steps"] for r in results]
    coll = sum(r["total_collisions"] for r in results)
    traps = sum(r["total_traps"] for r in results)
    n = len(results)
    print(f"{algo}: SR={sum(srs)/n:.2%}, Ret={sum(rets)/n:.2f}, Steps={sum(steps)/n:.1f}, Coll={coll}, Traps={traps}")

pd = data["matched_comparison"]["paired_differences"]
diffs = [d["diff_success_rate"] for d in pd]
print(f"Paired SR diff: mean={sum(diffs)/len(diffs):.4f}")
print()

# Tuned comparison
print("=== TUNED COMPARISON ===")
for r in data["tuned_comparison"]:
    algo = r["algorithm"]
    cfg = r["config"]
    sr_m = r["success_rate_mean"]
    sr_s = r["success_rate_std"]
    ret_m = r["mean_return_mean"]
    steps_m = r["mean_steps_mean"]
    coll = r["total_collisions"]
    traps = r["total_traps"]
    print(f"{algo} {cfg}: SR={sr_m:.2%}+-{sr_s:.2%}, Ret={ret_m:.1f}, Steps={steps_m:.1f}, Coll={coll}, Traps={traps}")
print()

# Room 4 best
with open("storage/experiments/final/room4_approximate_sarsa_confirmation.json") as f:
    r4 = json.load(f)
best = r4["confirmation_results"][0] if r4.get("confirmation_results") else None
if best:
    print(f"Room 4 Best: {best['config_tuple']}")
    print(f"  Fixed training SR: {best['fixed_training_start_success_rate']:.2%}")
    print(f"  Fixed unseen SR: {best['fixed_unseen_starts_success_rate']:.2%}")
    print(f"  Random lower-left SR: {best['random_lower_left_success_rate']:.2%}")
    print(f"  Random room SR: {best['random_room_success_rate']:.2%}")
print()

# Summary CSV
print("=== FINAL SUMMARY ===")
with open("storage/experiments/final/final_summary.csv", newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        print(f"{row['room']:10} {row['algorithm']:20} {row['best_config_id']:50} SR={row['success_rate_mean']}")
