import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_report(results_path='results/baselines/comparison_results.csv', output_dir='analyses'):
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return

    df = pd.read_csv(results_path)
    
    # --- 1. ATE Error Plot ---
    plt.figure(figsize=(10, 6))
    plt.bar(df['model'], df['ate_error'], color=['gray', 'skyblue', 'orange', 'green'])
    plt.title('Average Treatment Effect (ATE) Error by Model')
    plt.ylabel('Absolute Error (lower is better)')
    plt.xlabel('Model')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    error_plot_path = os.path.join(output_dir, 'ate_error_comparison.png')
    plt.savefig(error_plot_path)
    plt.close()
    print(f"Generated error plot: {error_plot_path}")

    # --- 2. ATE Estimates with Confidence Intervals ---
    plt.figure(figsize=(10, 6))
    # Calculate 95% CI width roughly as 1.96 * std_err
    ci_95 = 1.96 * df['std_err'] if 'std_err' in df.columns else None
    
    plt.bar(df['model'], df['ate_est'], yerr=ci_95, capsize=10, color=['gray', 'skyblue', 'orange', 'green'], alpha=0.7)
    plt.axhline(y=1.0, color='r', linestyle='--', label='True ATE (1.0)')
    plt.title('ATE Estimates with 95% Confidence Intervals')
    plt.ylabel('Estimated ATE')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    ci_plot_path = os.path.join(output_dir, 'ate_estimates_ci.png')
    plt.savefig(ci_plot_path)
    plt.close()
    print(f"Generated CI plot: {ci_plot_path}")

    # --- 3. Efficiency Frontier (Time vs Error) ---
    if 'training_time' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.scatter(df['training_time'], df['ate_error'], s=100, c=['gray', 'skyblue', 'orange', 'green'])
        
        # Add labels
        for i, row in df.iterrows():
            plt.annotate(row['model'], (row['training_time'], row['ate_error']), 
                         xytext=(5, 5), textcoords='offset points')
            
        plt.title('Efficiency Frontier: Accuracy vs Compute')
        plt.ylabel('ATE Error (Lower is Better)')
        plt.xlabel('Training Time (s) (Lower is Better)')
        plt.grid(True, linestyle='--', alpha=0.5)
        
        frontier_plot_path = os.path.join(output_dir, 'efficiency_frontier.png')
        plt.savefig(frontier_plot_path)
        plt.close()
        print(f"Generated frontier plot: {frontier_plot_path}")

    # --- 4. Training Time Plot ---
    if 'training_time' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.bar(df['model'], df['training_time'], color='lightgreen')
        plt.title('Training Time by Model')
        plt.ylabel('Time (seconds)')
        plt.xlabel('Model')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        time_plot_path = os.path.join(output_dir, 'training_time_comparison.png')
        plt.savefig(time_plot_path)
        plt.close()
        print(f"Generated time plot: {time_plot_path}")

    # --- 5. Markdown Report ---
    report_content = f"""
# Causal Engine Benchmark Report

## Overview
This report compares the performance of the **JEPA-DML** engine against standard causal inference baselines.

## Results Summary

{df.to_markdown(index=False)}

## Performance Analysis

### 1. ATE Estimation Accuracy
![ATE Error](ate_error_comparison.png)

*   **Lower is better.**
*   The **JEPA-DML** approach aims to minimize bias by learning robust representations of confounders.
*   **VICReg-DML** serves as a strong self-supervised baseline.
*   **DragonNet** represents end-to-end supervised causal learning.

### 2. Estimates & Confidence Intervals
![ATE Estimates](ate_estimates_ci.png)

*   The red dashed line indicates the **True ATE (1.0)**.
*   Models with bars overlapping the red line are statistically consistent with the truth.
*   Narrower error bars indicate higher precision (lower variance).

### 3. Efficiency Frontier
![Efficiency](efficiency_frontier.png)

*   **Bottom-Left is ideal** (Fast & Accurate).
*   JEPA-DML and VICReg-DML invest more compute (pre-training) to achieve lower error.
*   DragonNet is faster but may have higher bias in this setting.

### 4. Computational Efficiency
![Training Time](training_time_comparison.png)

*   Training time includes representation learning and nuisance model training.
*   Orchestration caching (not shown here) significantly reduces subsequent runtimes for JEPA models.

## Conclusion
Based on the current run, the best performing model is **{df.loc[df['ate_error'].idxmin()]['model']}** with an error of **{df['ate_error'].min():.4f}**.
"""

    report_path = os.path.join(output_dir, 'BENCHMARK_REPORT.md')
    with open(report_path, 'w') as f:
        f.write(report_content)
    
    print(f"Generated full report: {report_path}")

if __name__ == "__main__":
    generate_report()
