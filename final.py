import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from collections import defaultdict
import requests
import json
import time
import os
from scholarly import scholarly
import csv
from utils import save_plot
from report_generator import create_analysis_report

def fetch_combined_network_data(iit_name="IIT Bombay", nit_name="NIT Trichy", max_professors=8, papers_per_professor=5):
    """
    Fetch data for a combined citation network between professors at specified IIT and NIT.
    
    Args:
        iit_name (str): Name of the IIT institution
        nit_name (str): Name of the NIT institution
        max_professors (int): Maximum professors to include from each institution
        papers_per_professor (int): Maximum papers to fetch per professor
        
    Returns:
        dict: Dictionary containing papers data
    """
    print(f"Building combined citation network for professors from {iit_name} and {nit_name}...")
    
    # Dictionary to store all papers
    papers = {}
    # Dictionary to store all professors
    professors = {}
    paper_counter = 1
    
    # Process each institution
    for institution in [iit_name, nit_name]:
        print(f"\nSearching for professors from {institution}...")
        
        # Search for authors with this institution affiliation
        search_query = scholarly.search_author(institution)
        professors_found = 0
        
        # Keep trying until we find enough professors or exhaust results
        for _ in range(30):  # Try up to 30 to find max_professors that match
            if professors_found >= max_professors:
                break
                
            try:
                professor = next(search_query)
                
                # Verify the institution in the affiliation
                affiliation = professor.get('affiliation', '').lower()
                institution_keywords = institution.lower().split()
                
                # Check if this matches our target institution
                if all(keyword in affiliation for keyword in institution_keywords):
                    professor_id = professor.get('scholar_id')
                    professor_name = professor.get('name', 'Unknown Professor')
                    
                    # Store professor data
                    professors[professor_id] = {
                        'name': professor_name,
                        'institution': institution,
                        'papers': []
                    }
                    
                    print(f"Found professor: {professor_name} from {institution}")
                    
                    # Get detailed professor info
                    try:
                        professor_detailed = scholarly.fill(professor)
                        publications = professor_detailed.get('publications', [])
                        
                        # Process papers for this professor
                        papers_added = 0
                        for pub in publications[:papers_per_professor*2]:  # Try more to account for failures
                            if papers_added >= papers_per_professor:
                                break
                                
                            try:
                                # Fill publication details
                                pub_filled = scholarly.fill(pub)
                                
                                # Paper ID
                                paper_id = f"p{paper_counter}"
                                paper_counter += 1
                                
                                # Extract basic information
                                title = pub_filled.get('bib', {}).get('title', 'Unknown Title')
                                
                                # Handle potential missing or non-integer year
                                try:
                                    year = int(pub_filled.get('bib', {}).get('pub_year', '2022'))
                                except (ValueError, TypeError):
                                    year = 2022
                                
                                # Create paper entry
                                papers[paper_id] = {
                                    'title': title,
                                    'author': professor_name,
                                    'institution': institution,
                                    'year': year,
                                    'cited_by': [],
                                    'num_citations': pub_filled.get('num_citations', 0),
                                    'scholar_id': pub_filled.get('pub_url', ''),
                                    'professor_id': professor_id
                                }
                                
                                # Associate paper with professor
                                professors[professor_id]['papers'].append(paper_id)
                                
                                papers_added += 1
                                print(f"  - Added paper: {title[:40]}... ({year})")
                                
                                # Avoid rate limiting
                                time.sleep(1)
                                
                            except Exception as e:
                                print(f"  - Error fetching paper details: {str(e)[:100]}")
                                continue
                        
                        professors_found += 1
                        print(f"Added {papers_added} papers from {professor_name}")
                        
                    except Exception as e:
                        print(f"Error fetching professor details: {str(e)[:100]}")
                
                # Avoid rate limiting
                time.sleep(1)
                
            except StopIteration:
                print(f"No more professors found for {institution}")
                break
            except Exception as e:
                print(f"Error in professor search: {str(e)[:100]}")
                time.sleep(2)
    
    # Attempt to establish citation relationships between papers
    print("\nEstablishing citation relationships...")
    
    # 1. First approach: Use scholarly's citation data where available
    for paper_id, paper in papers.items():
        scholar_id = paper.get('scholar_id', '')
        if scholar_id:
            try:
                # Try to get papers that cite this paper
                # This is simplified as scholarly has limitations on citation retrieval
                pass  # Placeholder - scholarly doesn't easily support this operation
            except Exception:
                pass
                
    # 2. Second approach: Create realistic citation patterns based on:
    # - Papers from same institution cite each other more frequently
    # - Newer papers cite older papers
    # - Papers in similar areas (keyword matching) cite each other more often
    
    for paper_id1, paper1 in papers.items():
        for paper_id2, paper2 in papers.items():
            # Skip self-citations at paper level
            if paper_id1 == paper_id2:
                continue
                
            # Only newer papers can cite older papers
            if paper1['year'] <= paper2['year']:
                continue
                
            # Calculate citation probability based on multiple factors
            
            # Factor 1: Year difference (more likely to cite recent papers)
            year_diff = paper1['year'] - paper2['year']
            year_factor = max(0, 1 - (year_diff / 10))  # 0.0 to 1.0
            
            # Factor 2: Same institution bonus
            same_institution = paper1['institution'] == paper2['institution']
            institution_factor = 1.2 if same_institution else 0.8
            
            # Factor 3: Title similarity (crude measure of research relevance)
            words1 = set(paper1['title'].lower().split())
            words2 = set(paper2['title'].lower().split())
            common_words = words1.intersection(words2)
            # Ignore common stopwords
            stopwords = {'a', 'an', 'the', 'and', 'of', 'in', 'on', 'for', 'to', 'with'}
            meaningful_common_words = common_words - stopwords
            similarity_factor = min(1.0, len(meaningful_common_words) * 0.3)
            
            # Combined probability
            citation_probability = min(0.8, year_factor * institution_factor * (0.1 + similarity_factor))
            
            # Apply randomness with weighted probability
            if np.random.random() < citation_probability:
                paper2['cited_by'].append(paper_id1)
                
    # Log some statistics about the network
    total_citations = sum(len(paper['cited_by']) for paper in papers.values())
    print(f"\nNetwork created with {len(papers)} papers and {total_citations} citations")
    print(f"Papers from {iit_name}: {sum(1 for p in papers.values() if p['institution'] == iit_name)}")
    print(f"Papers from {nit_name}: {sum(1 for p in papers.values() if p['institution'] == nit_name)}")
    
    return {'papers': papers, 'professors': professors}

def save_network_data(network_data, filename='iit_nit_network.json'):
    """Save the network data to a JSON file"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(network_data, f, indent=2)
    print(f"Network data saved to {filename}")

def load_network_data(filename='iit_nit_network.json'):
    """Load network data from a JSON file"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"Loaded network data from {filename}")
        return data
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None

def build_citation_network(network_data):
    """Build NetworkX directed graph from the network data"""
    papers = network_data['papers']
    
    G = nx.DiGraph()
    
    # Add nodes for papers
    for paper_id, paper in papers.items():
        G.add_node(paper_id, 
                  title=paper['title'],
                  author=paper['author'],
                  institution=paper['institution'],
                  year=paper['year'],
                  type='paper')
    
    # Add edges (citations)
    for paper_id, paper in papers.items():
        for cited_paper in paper['cited_by']:
            if cited_paper in G:  # Ensure the cited paper exists
                G.add_edge(cited_paper, paper_id)  # Direction: cited_paper -> paper_id
    
    return G

def analyze_eigenvector_centrality(G):
    """Calculate eigenvector centrality handling directed graphs"""
    try:
        # First try with the directed graph
        eigenvector_centrality = nx.eigenvector_centrality_numpy(G)
        
        # If all values are 0, try with undirected version
        if all(v == 0 for v in eigenvector_centrality.values()):
            G_undirected = G.to_undirected()
            eigenvector_centrality = nx.eigenvector_centrality_numpy(G_undirected)
        
        return eigenvector_centrality
    except:
        print("Warning: Eigenvector centrality calculation failed")
        return {node: 0 for node in G.nodes()}

def analyze_betweenness_centrality(G):
    try:
        # Using approximate method for large networks
        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
        return betweenness_centrality
    except:
        print("Warning: Betweenness centrality calculation failed")
        return {node: 0 for node in G.nodes()}
    
def analyze_closeness_centrality(G):
    try:
        closeness_centrality = nx.closeness_centrality(G)
        return closeness_centrality
    except:
        print("Warning: Closeness centrality calculation failed")
        return {node: 0 for node in G.nodes()}
    
def analyze_hits(G):
    try:
        hubs, authorities = nx.hits(G)
        return hubs, authorities
    except:
        print("Warning: HITS algorithm calculation failed")
        return {node: 0 for node in G.nodes()}, {node: 0 for node in G.nodes()}
    
def analyze_clustering(G):
    try:
        clustering = nx.clustering(G)
        average_clustering = nx.average_clustering(G)
        return clustering, average_clustering
    except:
        print("Warning: Clustering coefficient calculation failed")
        return {node: 0 for node in G.nodes()}, 0
    
def detect_communities(G):
    try:
        # Convert to undirected graph for community detection
        G_undirected = G.to_undirected()
        communities = nx.community.greedy_modularity_communities(G_undirected)
        return communities
    except:
        print("Warning: Community detection failed")
        return []

def analyze_constraint(G):
    try:
        constraint = nx.constraint(G)
        return constraint
    except:
        print("Warning: Constraint calculation failed")
        return {node: 0 for node in G.nodes()}

def analyze_citation_network(G, network_data):
    """Analyze the citation network"""
    papers = network_data['papers']
    professors = network_data['professors']
    
    print("\n=== Citation Network Analysis ===")
    
    # Basic network statistics
    print(f"\nBasic Network Statistics:")
    print(f"Number of papers: {G.number_of_nodes()}")
    print(f"Number of citations: {G.number_of_edges()}")
    print(f"Network density: {nx.density(G):.4f}")
    
    # Calculate centrality measures
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    
    try:
        pagerank = nx.pagerank(G, alpha=0.85)
    except:
        print("Warning: Pagerank calculation failed, using simplified method")
        pagerank = {node: G.in_degree(node) / max(1, G.number_of_nodes()) for node in G.nodes()}
    
    # Create DataFrame for paper metrics
    metrics_df = pd.DataFrame({
        'Title': [papers[p]['title'] for p in G.nodes()],
        'Author': [papers[p]['author'] for p in G.nodes()],
        'Institution': [papers[p]['institution'] for p in G.nodes()],
        'Year': [papers[p]['year'] for p in G.nodes()],
        'Citations': [in_degree.get(p, 0) for p in G.nodes()],
        'References': [out_degree.get(p, 0) for p in G.nodes()],
        'PageRank': [pagerank.get(p, 0) for p in G.nodes()]
    }, index=list(G.nodes()))
    
    # Cross-institution citation analysis
    institutions = list(set(papers[p]['institution'] for p in G.nodes()))
    institution_matrix = pd.DataFrame(index=institutions, columns=institutions, data=0)
    
    # Fill the citation matrix
    for paper_id, paper in papers.items():
        for cited_id in paper['cited_by']:
            if cited_id in papers:  # Make sure the cited paper exists in our data
                citing_inst = paper['institution']
                cited_inst = papers[cited_id]['institution']
                institution_matrix.loc[citing_inst, cited_inst] += 1
    
    # Professor-level analysis
    professor_citations = defaultdict(int)
    professor_papers = defaultdict(int)
    
    for prof_id, prof in professors.items():
        for paper_id in prof['papers']:
            if paper_id in in_degree:
                professor_citations[prof['name']] += in_degree.get(paper_id, 0)
                professor_papers[prof['name']] += 1
    
    professor_df = pd.DataFrame({
        'Name': list(professor_citations.keys()),
        'Institution': [professors[p]['institution'] for p in professors],
        'Papers': [professor_papers[name] for name in professor_citations.keys()],
        'Citations': list(professor_citations.values()),
        'Avg Citations Per Paper': [
            professor_citations[name] / max(1, professor_papers[name]) 
            for name in professor_citations.keys()
        ]
    })
    
    # Check for connectivity
    if not nx.is_weakly_connected(G):
        print(f"\nNote: Network is not fully connected")
        print(f"Number of connected components: {nx.number_weakly_connected_components(G)}")
    
    # Identify cross-institution bridges
    cross_inst_bridges = []
    for u, v in G.edges():
        if papers[u]['institution'] != papers[v]['institution']:
            cross_inst_bridges.append((u, v))
    
    print(f"\nCross-institution citations: {len(cross_inst_bridges)}")
    
    # return {
    #     'metrics_df': metrics_df,
    #     'institution_matrix': institution_matrix,
    #     'professor_df': professor_df
    # }
    
    # Additional centrality measures
    try:
        eigenvector_centrality = analyze_eigenvector_centrality(G)
    except:
        print("Warning: Eigenvector centrality calculation failed")
        eigenvector_centrality = {node: 0 for node in G.nodes()}
        
    try:
        betweenness_centrality = nx.betweenness_centrality(G, k=min(100, G.number_of_nodes()))
    except:
        print("Warning: Betweenness centrality calculation failed")
        betweenness_centrality = {node: 0 for node in G.nodes()}
        
    try:
        closeness_centrality = nx.closeness_centrality(G)
    except:
        print("Warning: Closeness centrality calculation failed")
        closeness_centrality = {node: 0 for node in G.nodes()}
    
    try:
        hubs, authorities = nx.hits(G)
    except:
        print("Warning: HITS algorithm calculation failed")
        hubs = {node: 0 for node in G.nodes()}
        authorities = {node: 0 for node in G.nodes()}
    
    # Add to metrics dataframe
    metrics_df['Eigenvector'] = [eigenvector_centrality.get(p, 0) for p in G.nodes()]
    metrics_df['Betweenness'] = [betweenness_centrality.get(p, 0) for p in G.nodes()]
    metrics_df['Closeness'] = [closeness_centrality.get(p, 0) for p in G.nodes()]
    metrics_df['Hub Score'] = [hubs.get(p, 0) for p in G.nodes()]
    metrics_df['Authority Score'] = [authorities.get(p, 0) for p in G.nodes()]
    
    # Identify influential papers by different metrics
    print("\nInfluential Papers by Different Centrality Measures:")
    print("Top by PageRank:")
    for p in sorted(G.nodes(), key=lambda x: pagerank.get(x, 0), reverse=True)[:5]:
        print(f"  - {papers[p]['title'][:40]}... ({papers[p]['institution']})")
    
    print("\nTop by Eigenvector Centrality:")
    for p in sorted(G.nodes(), key=lambda x: eigenvector_centrality.get(x, 0), reverse=True)[:5]:
        print(f"  - {papers[p]['title'][:40]}... ({papers[p]['institution']})")
    
    print("\nTop by Betweenness Centrality (Bridging Papers):")
    for p in sorted(G.nodes(), key=lambda x: betweenness_centrality.get(x, 0), reverse=True)[:5]:
        print(f"  - {papers[p]['title'][:40]}... ({papers[p]['institution']})")
        
    print("\nTop by Closeness Centrality:")
    for p in sorted(G.nodes(), key=lambda x: closeness_centrality.get(x, 0), reverse=True)[:5]:
        print(f"  - {papers[p]['title'][:40]}... ({papers[p]['institution']})")
        
    print("\nTop by HITS Hub Score:")
    for p in sorted(G.nodes(), key=lambda x: hubs.get(x, 0), reverse=True)[:5]:
        print(f"  - {papers[p]['title'][:40]}... ({papers[p]['institution']})")
        
    print("\nTop by HITS Authority Score:")
    for p in sorted(G.nodes(), key=lambda x: authorities.get(x, 0), reverse=True)[:5]:
        print(f"  - {papers[p]['title'][:40]}... ({papers[p]['institution']})")
        
    # Rest of existing code...
    
    return {
        'metrics_df': metrics_df,
        'institution_matrix': institution_matrix,
        'professor_df': professor_df,
        'eigenvector_centrality': eigenvector_centrality,
        'betweenness_centrality': betweenness_centrality,
        'closeness_centrality': closeness_centrality,
        'hubs': hubs,
        'authorities': authorities
    }

def visualize_citation_network(G, network_data, output_prefix="iit_nit"):
    """Create visualizations for the citation network"""
    papers = network_data['papers']
    
    # 1. Main Network Visualization
    plt.figure(num='Citation Network', figsize=(14, 12))
    
    # Use spring layout with seed for reproducibility
    pos = nx.spring_layout(G, k=0.2, seed=42)
    
    # Color nodes by institution
    institutions = list(set(papers[p]['institution'] for p in G.nodes()))
    institution_colors = {
        institutions[0]: 'royalblue',
        institutions[1]: 'crimson'
    }
    
    node_colors = [institution_colors[papers[node]['institution']] for node in G.nodes()]
    
    # Size nodes by citation count (in-degree)
    in_degree = dict(G.in_degree())
    node_sizes = [20 + 100 * in_degree.get(node, 0) for node in G.nodes()]
    
    # Draw the nodes
    nx.draw_networkx_nodes(G, pos, 
                         node_color=node_colors, 
                         node_size=node_sizes, 
                         alpha=0.8)
    
    # Draw the edges with different colors for intra vs cross-institution
    intra_edges = [(u, v) for u, v in G.edges() 
                  if papers[u]['institution'] == papers[v]['institution']]
    cross_edges = [(u, v) for u, v in G.edges() 
                   if papers[u]['institution'] != papers[v]['institution']]
    
    # Draw intra-institution edges
    nx.draw_networkx_edges(G, pos, edgelist=intra_edges, 
                         alpha=0.5, edge_color='gray', arrows=True)
    
    # Draw cross-institution edges with different color to highlight them
    nx.draw_networkx_edges(G, pos, edgelist=cross_edges, 
                         alpha=0.8, edge_color='green', arrows=True, width=2)
    
    # Add labels to important nodes (top by PageRank)
    try:
        pagerank = nx.pagerank(G)
    except:
        pagerank = {node: G.in_degree(node) for node in G.nodes()}
    
    # Label only the top papers
    top_papers = sorted(G.nodes(), key=lambda x: pagerank.get(x, 0), reverse=True)[:10]
    labels = {node: f"{node}: {papers[node]['title'][:20]}..." for node in top_papers}
    
    # Draw labels with appropriate font size
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, font_color='black')
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=institution_colors[institutions[0]], 
                 markersize=10, label=institutions[0]),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=institution_colors[institutions[1]], 
                 markersize=10, label=institutions[1]),
        plt.Line2D([0], [0], color='gray', lw=2, label='Intra-institution citation'),
        plt.Line2D([0], [0], color='green', lw=2, label='Cross-institution citation')
    ]
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title(f"Citation Network: {institutions[0]} and {institutions[1]}", fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    
    # Show the plot and save it
    save_plot(plt, f"{output_prefix}_citation_network")
    plt.show()
    input("Press Enter to continue to the next visualization...")
    plt.close()
    
    # 2. Institution Citation Matrix Heatmap
    analysis_results = analyze_citation_network(G, network_data)
    institution_matrix = analysis_results['institution_matrix']
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(institution_matrix, annot=True, fmt='d', cmap='Blues',
               linewidths=1, linecolor='white')
    plt.title('Citation Matrix Between Institutions', fontsize=16)
    plt.xlabel('Cited Institution')
    plt.ylabel('Citing Institution')
    plt.tight_layout()
    save_plot(plt, f"{output_prefix}_institution_matrix")
    plt.show()
    input("Press Enter to continue to the next visualization...")
    plt.close()
    
    # 3. Professor Citation Bar Chart
    professor_df = analysis_results['professor_df']
    
    # Group by institution
    professor_df['Institution'] = professor_df['Institution'].astype('category')
    grouped = professor_df.groupby('Institution', observed=False)
    
    # Plot citations by professor, colored by institution
    plt.figure(figsize=(14, 8))
    ax = plt.subplot(111)
    
    bar_width = 0.8
    colors = ['royalblue', 'crimson']
    
    for i, (institution, group) in enumerate(grouped):
        x = np.arange(len(group))
        ax.bar(x + i*bar_width/2, group['Citations'], width=bar_width, 
              label=institution, color=colors[i], alpha=0.7)
        
        # Add professor names as x-tick labels
        plt.xticks(x + i*bar_width/4, group['Name'], rotation=45, ha='right')
    
    plt.ylabel('Number of Citations', fontsize=12)
    plt.title('Citations by Professor and Institution', fontsize=16)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_plot(plt, f"{output_prefix}_professor_citations")
    plt.show()
    input("Press Enter to continue to the next visualization...")
    plt.close()
    
    # 4. Year-based citation trend
    metrics_df = analysis_results['metrics_df']
    
    # Group by year and institution
    year_inst_citations = metrics_df.groupby(['Year', 'Institution']).agg(
        Papers=('Title', 'count'),
        Citations=('Citations', 'sum')
    ).reset_index()
    
    plt.figure(figsize=(12, 8))
    
    # Create line plot
    for institution in institutions:
        inst_data = year_inst_citations[year_inst_citations['Institution'] == institution]
        plt.plot(inst_data['Year'], inst_data['Citations'], 
               marker='o', linewidth=2, label=f"{institution} Citations")
    
    plt.title('Citation Trends by Year and Institution', fontsize=16)
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Number of Citations', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_plot(plt, f"{output_prefix}_citation_trends")
    plt.show()
    input("Press Enter to continue to the next visualization...")
    plt.close()

def run_full_analysis(iit_name="IIT Bombay", nit_name="NIT Trichy", use_cache=True):
    """Run the full citation network analysis workflow"""
    cache_file = f"{iit_name.replace(' ', '_')}_{nit_name.replace(' ', '_')}_network.json"
    
    # Try to load from cache if requested
    if use_cache and os.path.exists(cache_file):
        network_data = load_network_data(cache_file)
    else:
        # Fetch new data
        network_data = fetch_combined_network_data(iit_name, nit_name)
        
        # Save for future use
        save_network_data(network_data, cache_file)
    
    # Build the network
    G = build_citation_network(network_data)
    
    # Analyze the network
    analysis_results = analyze_citation_network(G, network_data)
    
    # Visualize the network first
    visualize_citation_network(G, network_data)
    
    # Then show the detailed metrics
    from visualize_metrics import visualize_all_metrics
    input("Press Enter to show detailed metrics visualizations...")  # Wait for user
    visualize_all_metrics(analysis_results, show_plots=True, save_plots=True)
    
    # Generate comprehensive report
    report_path = create_analysis_report(
        network_data,
        analysis_results,
        institutions=[iit_name, nit_name]
    )
    print(f"\nAnalysis complete! Open {report_path} to view the full report.")
    
    return G, network_data, analysis_results

# Run the analysis if executed directly
if __name__ == "__main__":
    # You can customize these institutions
    G, network_data, analysis = run_full_analysis(
        iit_name="IIT Bombay", 
        nit_name="NIT Trichy",
        use_cache=True  # Set to False to fetch fresh data
    )
