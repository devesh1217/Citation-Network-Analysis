from datetime import datetime
import os
import shutil
import networkx as nx

def create_analysis_report(network_data, analysis_results, institutions, plot_dir='plots', output_dir='outputs'):
    """Generate a comprehensive analysis report combining all outputs"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'analysis_report_{timestamp}')
    os.makedirs(report_path, exist_ok=True)

    # Create report file
    with open(os.path.join(report_path, 'report.txt'), 'w', encoding='utf-8') as f:
        # Write header
        f.write("Citation Network Analysis Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Institutions analyzed: {', '.join(institutions)}\n\n")

        # Network Overview
        metrics_df = analysis_results['metrics_df']
        f.write("1. Network Overview\n")
        f.write("=" * 30 + "\n")
        f.write(f"Total papers: {len(metrics_df)}\n")
        f.write(f"Total citations: {metrics_df['Citations'].sum()}\n")
        
        # Papers by institution
        f.write("\n2. Papers by Institution\n")
        f.write("=" * 30 + "\n")
        inst_papers = metrics_df.groupby('Institution')['Title'].count()
        for inst, count in inst_papers.items():
            f.write(f"{inst}: {count} papers\n")

        # Network Metrics
        f.write("\n3. Network Centrality Measures\n")
        f.write("=" * 30 + "\n")
        
        # Calculate mean and max values for each centrality measure
        centrality_measures = [
            ('PageRank', 'PageRank'),
            ('Citations', 'Citation Count'),
            ('Eigenvector', 'Eigenvector Centrality'),
            ('Betweenness', 'Betweenness Centrality'),
            ('Closeness', 'Closeness Centrality'),
            ('Hub Score', 'Hub Score'),
            ('Authority Score', 'Authority Score')
        ]

        for col, name in centrality_measures:
            mean_val = metrics_df[col].mean()
            max_val = metrics_df[col].max()
            f.write(f"\n{name}:\n")
            f.write(f"  Average: {mean_val:.4f}\n")
            f.write(f"  Maximum: {max_val:.4f}\n")
            
            # Top 5 papers by this measure
            f.write(f"  Top 5 Papers:\n")
            top_papers = metrics_df.nlargest(5, col)
            for idx, row in top_papers.iterrows():
                f.write(f"    - {row['Title'][:50]}... ({row['Institution']}): {row[col]:.4f}\n")

        # Cross-institution Analysis
        f.write("\n4. Cross-institution Analysis\n")
        f.write("=" * 30 + "\n")
        institution_matrix = analysis_results['institution_matrix']
        f.write("\nCitation Matrix:\n")
        f.write(str(institution_matrix))
        
        # Professor Analysis
        f.write("\n\n5. Professor Statistics\n")
        f.write("=" * 30 + "\n")
        prof_df = analysis_results['professor_df']
        
        # Overall professor statistics
        f.write("\nAggregate Statistics:\n")
        f.write(f"Total professors: {len(prof_df)}\n")
        f.write(f"Average papers per professor: {prof_df['Papers'].mean():.2f}\n")
        f.write(f"Average citations per professor: {prof_df['Citations'].mean():.2f}\n")
        
        # Individual professor statistics
        f.write("\nIndividual Statistics:\n")
        for idx, row in prof_df.iterrows():
            f.write(f"\n{row['Name']} ({row['Institution']}):\n")
            f.write(f"  Papers: {row['Papers']}\n")
            f.write(f"  Citations: {row['Citations']}\n")
            f.write(f"  Avg Citations/Paper: {row['Avg Citations Per Paper']:.2f}\n")

        # Year-wise Analysis
        f.write("\n6. Temporal Analysis\n")
        f.write("=" * 30 + "\n")
        yearly_stats = metrics_df.groupby('Year').agg({
            'Title': 'count',
            'Citations': 'sum'
        }).reset_index()
        yearly_stats.columns = ['Year', 'Papers', 'Citations']
        
        f.write("\nYear-wise Statistics:\n")
        for _, row in yearly_stats.iterrows():
            f.write(f"Year {row['Year']}:\n")
            f.write(f"  Papers: {row['Papers']}\n")
            f.write(f"  Citations: {row['Citations']}\n")

        # Plot References
        f.write("\n7. Visualization Reference\n")
        f.write("=" * 30 + "\n")
        f.write("\nThe following visualizations have been generated:\n")
        f.write("1. Citation Network Visualization (citation_network.png)\n")
        f.write("2. Institution Citation Matrix (institution_matrix.png)\n")
        f.write("3. Professor Citation Analysis (professor_citations.png)\n")
        f.write("4. Citation Trends (citation_trends.png)\n")
        f.write("5. Centrality Distributions (centrality_distributions.png)\n")
        f.write("6. Various Metric-specific Visualizations\n")

    # Copy all plots to report directory
    plots_dir = os.path.join(report_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    for filename in os.listdir(plot_dir):
        if filename.endswith('.png'):
            shutil.copy2(
                os.path.join(plot_dir, filename),
                os.path.join(plots_dir, filename)
            )

    print(f"\nComprehensive report generated successfully at: {report_path}")
    return report_path