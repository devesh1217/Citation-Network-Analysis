import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict
import scholarly  # Import the scholarly library
import time
import re
from tqdm import tqdm

def fetch_professor_publications(professor_name, affiliation, max_papers=5):
    """
    Fetch publications for a professor from Google Scholar using current scholarly API
    """
    print(f"Searching for professor: {professor_name} ({affiliation})")
    
    # Create a scholarly instance and search query using the current API
    try:
        # Create a search query using the current API
        query = scholarly.scholarly.search_author(professor_name)
        
        # Find the author with matching affiliation
        author = None
        for result in query:
            if affiliation.lower() in str(result.get('affiliation', '')).lower():
                author = result
                break
        
        if not author:
            # If no exact match for affiliation, take the first result
            query = scholarly.scholarly.search_author(professor_name)
            author = next(query)
            
        print(f"Found author: {author['name']}")
        
        # Fill in all available details
        author = scholarly.scholarly.fill(author)
        
        # Get recent publications (last 2-3 years)
        recent_pubs = []
        years = [2022, 2023, 2024]
        
        for pub in author['publications']:
            # Filter by year if available
            pub_filled = scholarly.scholarly.fill(pub)
            pub_year = pub_filled.get('bib', {}).get('pub_year')
            
            # Convert pub_year to int if it's a string and contains a 4-digit year
            if isinstance(pub_year, str):
                match = re.search(r'\b(19|20)\d{2}\b', pub_year)
                if match:
                    pub_year = int(match.group())
            
            if pub_year in years:
                recent_pubs.append(pub_filled)
                if len(recent_pubs) >= max_papers:
                    break
        
        print(f"Found {len(recent_pubs)} recent publications")
        return recent_pubs
    except StopIteration:
        print(f"Could not find author: {professor_name}")
        return []
    except Exception as e:
        print(f"Error fetching publications for {professor_name}: {e}")
        return []

def fetch_citation_data(professors):
    """
    Fetch citation data for a list of professors
    """
    all_papers = {}
    paper_details = {}
    
    for prof in professors:
        pubs = fetch_professor_publications(prof['name'], prof['affiliation'])
        
        for pub in pubs:
            paper_id = f"{pub.get('author_id', 'unknown')}_{pub.get('pub_year', 'unknown')}_{pub.get('citedby', 0)}"
            
            # Store paper details
            paper_details[paper_id] = {
                'title': pub.get('bib', {}).get('title', 'Unknown Title'),
                'author': prof['name'],
                'institution': prof['affiliation'],
                'year': pub.get('bib', {}).get('pub_year', 'Unknown Year'),
                'cited_by': []
            }
            
            # Get citing papers if available
            if pub.get('citedby', 0) > 0:
                try:
                    # Limit to maximum 5 citing papers per publication to avoid overloading
                    citing_papers = []
                    citations = scholarly.scholarly.citedby(pub, max_results=5)
                    for citation in citations:
                        citing_papers.append(citation)
                        
                    for citing in citing_papers:
                        citing_id = f"{citing.get('author_id', 'unknown')}_{citing.get('pub_year', 'unknown')}_{citing.get('citedby', 0)}"
                        paper_details[paper_id]['cited_by'].append(citing_id)
                        
                        # Also add the citing paper to our collection
                        if citing_id not in paper_details:
                            paper_details[citing_id] = {
                                'title': citing.get('bib', {}).get('title', 'Unknown Title'),
                                'author': citing.get('bib', {}).get('author', 'Unknown Author'),
                                'institution': 'Other', # We don't know the institution of citing papers
                                'year': citing.get('bib', {}).get('pub_year', 'Unknown Year'),
                                'cited_by': []
                            }
                    
                    # Be nice to Google Scholar - add delay between requests
                    time.sleep(2)
                
                except Exception as e:
                    print(f"Error fetching citations for paper {paper_id}: {e}")
    
    return paper_details

def build_citation_network(paper_details):
    """
    Build a citation network from paper details
    """
    G = nx.DiGraph()
    
    # Add nodes for all papers
    for paper_id, details in paper_details.items():
        G.add_node(paper_id, 
                  title=details['title'],
                  author=details['author'],
                  institution=details['institution'],
                  year=details['year'])
    
    # Add edges for citations
    for paper_id, details in paper_details.items():
        for cited_paper in details['cited_by']:
            if cited_paper in paper_details:  # Make sure the cited paper is in our dataset
                G.add_edge(cited_paper, paper_id)  # Direction: cited_paper -> paper_id
    
    return G

def analyze_citation_network(G):
    """
    Analyze the citation network
    """
    print("\nNetwork Analysis Metrics:")
    
    # Basic network statistics
    print(f"Number of papers: {G.number_of_nodes()}")
    print(f"Number of citations: {G.number_of_edges()}")
    print(f"Network density: {nx.density(G):.4f}")
    
    # Centrality measures
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    try:
        betweenness = nx.betweenness_centrality(G)
    except:
        print("Warning: Could not calculate betweenness centrality")
        betweenness = {node: 0 for node in G.nodes()}
    
    # Calculate PageRank
    pagerank = nx.pagerank(G, alpha=0.85)
    
    # For completeness, also calculate HITS algorithm scores (hub and authority)
    try:
        hits_scores = nx.hits(G, max_iter=100)
        hub_scores = hits_scores[0]
        authority_scores = hits_scores[1]
    except:
        print("Warning: Could not calculate HITS scores")
        hub_scores = {node: 0 for node in G.nodes()}
        authority_scores = {node: 0 for node in G.nodes()}
    
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

def calculate_institution_metrics(metrics_df):
    """
    Calculate metrics at the institution level
    """
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

def visualize_citation_network(G):
    """
    Visualize the citation network
    """
    plt.figure(figsize=(14, 12))
    
    # Node positions using spring layout
    pos = nx.spring_layout(G, seed=42, k=0.3)
    
    # Node colors based on institution
    node_colors = []
    for node in G.nodes():
        institution = G.nodes[node]['institution']
        if 'IIT' in institution:
            node_colors.append('skyblue')
        elif 'NIT' in institution:
            node_colors.append('salmon')
        else:
            node_colors.append('lightgray')
    
    # Node sizes based on in-degree (citations received)
    node_sizes = [100 + 300 * G.in_degree(node) for node in G.nodes()]
    
    # Draw the network
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, arrowsize=15, alpha=0.6, connectionstyle='arc3,rad=0.1')
    
    # Add labels for nodes with high degree only to avoid clutter
    high_degree_nodes = [node for node, degree in G.in_degree() if degree > 0]
    labels = {node: f"{G.nodes[node]['author'].split()[-1]}" for node in high_degree_nodes}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10)
    
    # Add legend
    iit_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='skyblue', markersize=15, label='IIT')
    nit_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='salmon', markersize=15, label='NIT')
    other_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray', markersize=15, label='Other Institutions')
    plt.legend(handles=[iit_patch, nit_patch, other_patch], loc='upper right')
    
    plt.title("Citation Network: IIT BHU and NIT Allahabad (2022-2024)", fontsize=15)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("citation_network.png", dpi=300)
    plt.show()

def visualize_metrics(metrics_df, G, inst_df):
    """
    Visualize various metrics
    """
    # Create a heatmap of metrics for top 15 papers by PageRank
    plt.figure(figsize=(16, 10))
    
    # Select top papers
    top_papers = metrics_df.sort_values('PageRank', ascending=False).head(15)
    
    # Select metrics for heatmap
    heatmap_data = top_papers[['In-Degree', 'Out-Degree', 'Betweenness', 'PageRank', 'Hub Score', 'Authority Score']]
    
    # Normalize data for better visualization
    heatmap_data = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min())
    
    # Create heatmap
    ax = sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt='.2f', linewidths=0.5)
    plt.title('Normalized Citation Network Metrics by Paper (Top 15 by PageRank)', fontsize=15)
    plt.tight_layout()
    plt.savefig("citation_metrics_heatmap.png", dpi=300)
    plt.show()
    
    # Institution comparison (filter to show only IIT and NIT)
    plt.figure(figsize=(10, 6))
    inst_filter = inst_df.index.str.contains('IIT|NIT')
    filtered_inst_df = inst_df[inst_filter]
    filtered_inst_df[['avg_citations', 'avg_pagerank']].plot(kind='bar', color=['steelblue', 'darkgreen'])
    plt.title('Average Citations and PageRank by Institution', fontsize=15)
    plt.ylabel('Value')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("institution_comparison.png", dpi=300)
    plt.show()
    
    # Author influence
    plt.figure(figsize=(12, 6))
    author_metrics = metrics_df.groupby('Author').agg({
        'In-Degree': 'sum',
        'PageRank': 'mean'
    }).sort_values('In-Degree', ascending=False).head(10)
    
    author_metrics.plot(kind='bar', color=['coral', 'teal'])
    plt.title('Top 10 Author Influence: Citations and Average PageRank', fontsize=15)
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
    
    # Citation distribution
    plt.figure(figsize=(10, 6))
    citation_counts = list(dict(G.in_degree()).values())
    plt.hist(citation_counts, bins=range(max(citation_counts)+2), color='green', alpha=0.7)
    plt.xlabel('Number of Citations')
    plt.ylabel('Number of Papers')
    plt.title('Distribution of Citations')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("citation_distribution.png", dpi=300)
    plt.show()

def main():
    # List of professors to analyze
    professors = [
        {'name': 'Rajeev Srivastava', 'affiliation': 'IIT BHU'},
        {'name': 'P. K. Jain', 'affiliation': 'IIT BHU'},
        {'name': 'S. D. Joshi', 'affiliation': 'NIT Allahabad'},
        {'name': 'Neetesh Purohit', 'affiliation': 'NIT Allahabad'}
    ]
    
    print("Starting citation network analysis...")
    print("This may take some time as we fetch data from Google Scholar")
    
    # Fetch citation data
    paper_details = fetch_citation_data(professors)
    
    # If you want to save the raw data for future use
    with open('paper_details.txt', 'w') as f:
        f.write(str(paper_details))
    
    # Build the citation network
    G = build_citation_network(paper_details)
    
    # If the network is too small, add some more data
    if G.number_of_nodes() < 10:
        print("Network is very small. You may want to add more professors or papers.")
    
    # Analyze the network
    metrics_df = analyze_citation_network(G)
    
    # Calculate institutional metrics
    inst_df = calculate_institution_metrics(metrics_df)
    
    # Visualize the network and metrics
    visualize_citation_network(G)
    visualize_metrics(metrics_df, G, inst_df)
    
    # Display top papers by various metrics
    print("\nTop Papers by Citations Received (In-Degree):")
    print(metrics_df.sort_values('In-Degree', ascending=False)[['Title', 'Author', 'Institution', 'In-Degree']].head(10))
    
    print("\nTop Papers by Influence (PageRank):")
    print(metrics_df.sort_values('PageRank', ascending=False)[['Title', 'Author', 'Institution', 'PageRank']].head(10))
    
    print("\nTop Papers by Hub Score:")
    print(metrics_df.sort_values('Hub Score', ascending=False)[['Title', 'Author', 'Institution', 'Hub Score']].head(10))
    
    print("\nTop Papers by Authority Score:")
    print(metrics_df.sort_values('Authority Score', ascending=False)[['Title', 'Author', 'Institution', 'Authority Score']].head(10))
    
    print("\nInstitution Comparison:")
    print(inst_df)
    
    # Additional connectivity analysis
    print("\nConnectivity Analysis:")
    print(f"Is the graph strongly connected? {nx.is_strongly_connected(G)}")
    print(f"Is the graph weakly connected? {nx.is_weakly_connected(G)}")
    if not nx.is_weakly_connected(G):
        print(f"Number of weakly connected components: {nx.number_weakly_connected_components(G)}")
    
    # Save the full metrics to CSV for further analysis
    metrics_df.to_csv('citation_metrics.csv')
    inst_df.to_csv('institution_metrics.csv')
    
    print("\nAnalysis complete. Results have been saved to CSV files and visualizations saved as PNG files.")

if __name__ == "__main__":
    main()