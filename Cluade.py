import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict

# Sample data structure - in real project, this would come from scraping 
# Google Scholar, Scopus or using an API

# Format: {paper_id: {title, author, institution, year, cited_by}}
sample_papers = {
    'p1': {'title': 'Machine Learning Applications in IoT', 'author': 'A. Kumar', 'institution': 'IIT BHU', 'year': 2022, 'cited_by': ['p3', 'p4', 'p7']},
    'p2': {'title': 'Deep Learning for Image Processing', 'author': 'B. Singh', 'institution': 'IIT BHU', 'year': 2022, 'cited_by': ['p5', 'p6']},
    'p3': {'title': 'Network Security using AI', 'author': 'C. Sharma', 'institution': 'NIT Allahabad', 'year': 2023, 'cited_by': ['p8']},
    'p4': {'title': 'Cloud Computing Architectures', 'author': 'D. Patel', 'institution': 'NIT Allahabad', 'year': 2023, 'cited_by': ['p6', 'p9']},
    'p5': {'title': 'Quantum Computing Applications', 'author': 'A. Kumar', 'institution': 'IIT BHU', 'year': 2023, 'cited_by': ['p8', 'p10']},
    'p6': {'title': 'Blockchain Technology Review', 'author': 'C. Sharma', 'institution': 'NIT Allahabad', 'year': 2023, 'cited_by': []},
    'p7': {'title': 'AI Ethics Framework', 'author': 'B. Singh', 'institution': 'IIT BHU', 'year': 2024, 'cited_by': ['p10']},
    'p8': {'title': 'Smart Grid Optimization', 'author': 'D. Patel', 'institution': 'NIT Allahabad', 'year': 2024, 'cited_by': []},
    'p9': {'title': 'Edge Computing Systems', 'author': 'A. Kumar', 'institution': 'IIT BHU', 'year': 2024, 'cited_by': []},
    'p10': {'title': 'Cybersecurity Protocols', 'author': 'C. Sharma', 'institution': 'NIT Allahabad', 'year': 2024, 'cited_by': []}
}

# Create the network
G = nx.DiGraph()

# Add nodes
for paper_id, paper_info in sample_papers.items():
    G.add_node(paper_id, 
              title=paper_info['title'],
              author=paper_info['author'],
              institution=paper_info['institution'],
              year=paper_info['year'])

# Add edges (citations)
for paper_id, paper_info in sample_papers.items():
    for cited_paper in paper_info['cited_by']:
        G.add_edge(cited_paper, paper_id)  # Direction: cited_paper -> paper_id

# Function to calculate and display network metrics
def analyze_citation_network(G):
    print("Network Analysis Metrics:\n")
    
    # Basic network statistics
    print(f"Number of papers: {G.number_of_nodes()}")
    print(f"Number of citations: {G.number_of_edges()}")
    print(f"Network density: {nx.density(G):.4f}")
    
    # Centrality measures
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    betweenness = nx.betweenness_centrality(G)
    
    # Calculate PageRank instead of eigenvector centrality
    # PageRank works well for directed networks and handles disconnected components
    pagerank = nx.pagerank(G, alpha=0.85)
    
    # For completeness, also calculate HITS algorithm scores (hub and authority)
    hits_scores = nx.hits(G, max_iter=100)
    hub_scores = hits_scores[0]
    authority_scores = hits_scores[1]
    
    # Create DataFrame for metrics
    metrics_df = pd.DataFrame({
        'In-Degree': pd.Series(in_degree),
        'Out-Degree': pd.Series(out_degree),
        'Betweenness': pd.Series(betweenness),
        'PageRank': pd.Series(pagerank),
        'Hub Score': pd.Series(hub_scores),
        'Authority Score': pd.Series(authority_scores)
    })
    
    # Add paper info to metrics
    for paper_id in metrics_df.index:
        metrics_df.at[paper_id, 'Title'] = G.nodes[paper_id]['title']
        metrics_df.at[paper_id, 'Author'] = G.nodes[paper_id]['author']
        metrics_df.at[paper_id, 'Institution'] = G.nodes[paper_id]['institution']
        metrics_df.at[paper_id, 'Year'] = G.nodes[paper_id]['year']
    
    # Check if the graph is weakly connected
    if not nx.is_weakly_connected(G):
        print(f"Note: The graph is disconnected with {nx.number_weakly_connected_components(G)} weakly connected components")
        
    # Calculate component statistics
    component_sizes = [len(c) for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True)]
    print(f"Component sizes: {component_sizes}")
    
    return metrics_df

# Analyze the network
metrics_df = analyze_citation_network(G)

# Calculate institution-level metrics
def institution_metrics(metrics_df):
    inst_metrics = defaultdict(lambda: {'papers': 0, 'citations': 0, 'pagerank': 0})
    
    for _, row in metrics_df.iterrows():
        inst = row['Institution']
        inst_metrics[inst]['papers'] += 1
        inst_metrics[inst]['citations'] += row['In-Degree']
        inst_metrics[inst]['pagerank'] += row['PageRank']
    
    # Create DataFrame for institution metrics
    inst_df = pd.DataFrame.from_dict(inst_metrics, orient='index')
    inst_df['avg_citations'] = inst_df['citations'] / inst_df['papers']
    inst_df['avg_pagerank'] = inst_df['pagerank'] / inst_df['papers']
    
    return inst_df

# Get institution metrics
inst_df = institution_metrics(metrics_df)

# Visualize the citation network
def visualize_citation_network(G):
    plt.figure(figsize=(12, 10))
    
    # Node positions using spring layout
    pos = nx.spring_layout(G, seed=42)
    
    # Node colors based on institution
    node_colors = []
    for node in G.nodes():
        if G.nodes[node]['institution'] == 'IIT BHU':
            node_colors.append('skyblue')
        else:
            node_colors.append('salmon')
    
    # Node sizes based on in-degree (citations received)
    node_sizes = [100 + 300 * G.in_degree(node) for node in G.nodes()]
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15, alpha=0.6)
    
    # Add labels
    labels = {node: f"{node}\n({G.nodes[node]['author']})" for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    
    # Add legend
    iit_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=15, label='IIT BHU')
    nit_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=15, label='NIT Allahabad')
    plt.legend(handles=[iit_patch, nit_patch], loc='upper right')
    
    plt.title("Citation Network: IIT BHU and NIT Allahabad (2022-2024)", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("citation_network.png", dpi=300)
    plt.show()

# Visualize the metrics with heatmap for top papers
def visualize_metrics(metrics_df):
    # Create a heatmap of metrics
    plt.figure(figsize=(14, 8))
    
    # Select metrics for heatmap
    heatmap_data = metrics_df[['In-Degree', 'Out-Degree', 'Betweenness', 'PageRank', 'Hub Score', 'Authority Score']]
    
    # Normalize data for better visualization
    heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    
    # Create heatmap
    ax = sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=0.5)
    plt.title('Normalized Citation Network Metrics by Paper', fontsize=15)
    plt.tight_layout()
    plt.savefig("citation_metrics_heatmap.png", dpi=300)
    plt.show()
    
    # Institution comparison
    plt.figure(figsize=(10, 6))
    inst_df[['avg_citations', 'avg_pagerank']].plot(kind='bar', color=['steelblue', 'darkgreen'])
    plt.title('Average Citations and PageRank by Institution', fontsize=15)
    plt.ylabel('Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("institution_comparison.png", dpi=300)
    plt.show()
    
    # Author influence
    author_metrics = metrics_df.groupby('Author').agg({
        'In-Degree': 'sum',
        'PageRank': 'mean'
    }).sort_values('In-Degree', ascending=False)
    
    plt.figure(figsize=(10, 6))
    ax = author_metrics.plot(kind='bar', color=['coral', 'teal'])
    plt.title('Author Influence: Citations and Average PageRank', fontsize=15)
    plt.ylabel('Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("author_influence.png", dpi=300)
    plt.show()

    # Create a visualization for component analysis
    component_sizes = [len(c) for c in sorted(nx.weakly_connected_components(G), key=len, reverse=True)]
    plt.figure(figsize=(10, 6))
    plt.bar(range(1, len(component_sizes) + 1), component_sizes, color='purple')
    plt.xlabel('Component Number')
    plt.ylabel('Number of Papers')
    plt.title('Size of Weakly Connected Components')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("component_analysis.png", dpi=300)
    plt.show()

# Run the visualization functions
visualize_citation_network(G)
visualize_metrics(metrics_df)

# Display the top papers by various metrics
print("\nTop Papers by Citations Received (In-Degree):")
print(metrics_df.sort_values('In-Degree', ascending=False)[['Title', 'Author', 'Institution', 'In-Degree']].head())

print("\nTop Papers by Influence (PageRank):")
print(metrics_df.sort_values('PageRank', ascending=False)[['Title', 'Author', 'Institution', 'PageRank']].head())

print("\nTop Papers by Hub Score (papers that cite many important papers):")
print(metrics_df.sort_values('Hub Score', ascending=False)[['Title', 'Author', 'Institution', 'Hub Score']].head())

print("\nTop Papers by Authority Score (papers cited by many important papers):")
print(metrics_df.sort_values('Authority Score', ascending=False)[['Title', 'Author', 'Institution', 'Authority Score']].head())

print("\nInstitution Comparison:")
print(inst_df)

# Additional connectivity analysis
print("\nConnectivity Analysis:")
print(f"Is the graph strongly connected? {nx.is_strongly_connected(G)}")
print(f"Is the graph weakly connected? {nx.is_weakly_connected(G)}")
if not nx.is_weakly_connected(G):
    print(f"Number of weakly connected components: {nx.number_weakly_connected_components(G)}")