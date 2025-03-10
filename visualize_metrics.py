import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from utils import save_plot

def visualize_metric_window(data, metric_name, figsize=(12, 8), save_plots=True):
    """Helper function to create and display individual metric windows"""
    plt.figure(num=f'Top Papers by {metric_name}', figsize=figsize)
    top_papers = data.nlargest(10, metric_name)
    top_papers['Short Title'] = top_papers['Title'].apply(lambda x: x[:20] + '...' if len(x) > 20 else x)
    sns.barplot(x=metric_name, y='Short Title', data=top_papers)
    plt.title(f'Top 10 Papers by {metric_name}')
    plt.tight_layout()
    if save_plots:
        save_plot(plt, f'top_{metric_name.lower().replace(" ", "_")}')
    plt.show()
    plt.close()

def visualize_all_metrics(analysis_results, show_plots=True, save_plots=True):
    """Visualize all metrics from the analysis results in separate windows"""
    
    # Create output directory if it doesn't exist
    import os
    os.makedirs('plots', exist_ok=True)
    
    metrics_df = analysis_results['metrics_df']

    if show_plots:
        # Institution Citation Matrix (shown first)
        plt.figure(num='Institution Citations', figsize=(10, 8))
        sns.heatmap(analysis_results['institution_matrix'], 
                    annot=True, fmt='g', cmap='YlOrRd')
        plt.title('Citation Flow Between Institutions')
        if save_plots:
            save_plot(plt, 'institution_citations')
        plt.show()
        plt.close()
        input("Press Enter to show individual metrics...")

        # Then show individual metrics
        for metric in ['PageRank', 'Citations', 'Eigenvector', 'Betweenness', 'Hub Score', 'Authority Score']:
            visualize_metric_window(metrics_df, metric, save_plots=save_plots)
            input(f"Press Enter to show next visualization...")

        # Correlation Matrix
        plt.figure(num='Correlation Matrix', figsize=(12, 10))
        numeric_cols = ['Citations', 'References', 'PageRank', 'Eigenvector', 
                       'Betweenness', 'Closeness', 'Hub Score', 'Authority Score']
        correlation_matrix = metrics_df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation between Different Centrality Measures')
        if save_plots:
            save_plot(plt, 'centrality_correlations')
        plt.show()
        plt.close()
        input("Press Enter to show next visualization...")

        # Professor Performance
        plt.figure(num='Professor Performance', figsize=(15, 6))
        prof_df = analysis_results['professor_df']
        x = np.arange(len(prof_df))
        width = 0.35

        plt.bar(x - width/2, prof_df['Papers'], width, label='Papers')
        plt.bar(x + width/2, prof_df['Citations'], width, label='Citations')
        plt.xlabel('Professors')
        plt.ylabel('Count')
        plt.title('Professor Publication Impact')
        plt.xticks(x, prof_df['Name'], rotation=45, ha='right')
        plt.legend()
        plt.tight_layout()
        if save_plots:
            save_plot(plt, 'professor_impact')
        plt.show()
        plt.close()
        input("Press Enter to show centrality distributions...")

        # Centrality Distributions (4 subplots in one window)
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 15), num='Centrality Distributions')
        plt.subplots_adjust(hspace=0.3, wspace=0.3)

        # Convert centrality values to lists
        eigenvector_values = list(analysis_results['eigenvector_centrality'].values())
        betweenness_values = list(analysis_results['betweenness_centrality'].values())
        closeness_values = list(analysis_results['closeness_centrality'].values())
        hub_values = list(analysis_results['hubs'].values())
        authority_values = list(analysis_results['authorities'].values())

        # Create the distributions
        sns.histplot(data=eigenvector_values, bins=20, kde=True, ax=ax1)
        ax1.set_title('Eigenvector Centrality Distribution')
        ax1.set_xlabel('Eigenvector Centrality')

        sns.histplot(data=betweenness_values, bins=20, kde=True, ax=ax2)
        ax2.set_title('Betweenness Centrality Distribution')
        ax2.set_xlabel('Betweenness Centrality')

        sns.histplot(data=closeness_values, bins=20, kde=True, ax=ax3)
        ax3.set_title('Closeness Centrality Distribution')
        ax3.set_xlabel('Closeness Centrality')

        sns.scatterplot(x=hub_values, y=authority_values, ax=ax4)
        ax4.set_title('Hubs vs Authorities')
        ax4.set_xlabel('Hub Score')
        ax4.set_ylabel('Authority Score')

        plt.suptitle('Centrality Measure Distributions', fontsize=16)
        plt.tight_layout()
        if save_plots:
            save_plot(plt, 'centrality_distributions')
        plt.show()
        plt.close()

    print("All visualizations have been generated!")
    if save_plots:
        print("Plots have been saved with timestamps in the 'plots' directory.")