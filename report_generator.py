from datetime import datetime
import os
import shutil
import networkx as nx
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def create_analysis_report(network_data, analysis_results, institutions, plot_dir='plots', output_dir='outputs'):
    """Generate a comprehensive analysis report combining all outputs"""
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join(output_dir, f'analysis_report_{timestamp}')
    os.makedirs(report_path, exist_ok=True)

    # Create PDF document
    pdf_path = os.path.join(report_path, 'report.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    
    # Custom styles
    custom_styles = {
        'CustomTitle': ParagraphStyle(
            name='CustomTitle',
            parent=styles['Heading1'],
            fontSize=24,
            spaceAfter=30
        ),
        'CustomHeading2': ParagraphStyle(
            name='CustomHeading2',
            parent=styles['Heading2'],
            fontSize=16,
            spaceBefore=20,
            spaceAfter=10
        )
    }

    # Add custom styles if they don't exist
    for style_name, style in custom_styles.items():
        if style_name not in styles:
            styles.add(style)

    # Build content
    content = []
    metrics_df = analysis_results['metrics_df']

    # Title and Header
    content.append(Paragraph("Citation Network Analysis Report", styles['CustomTitle']))
    content.append(Paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    content.append(Paragraph(f"Institutions analyzed: {', '.join(institutions)}", styles['Normal']))
    content.append(Spacer(1, 20))

    # 1. Network Overview
    content.append(Paragraph("1. Network Overview", styles['CustomHeading2']))
    overview_data = [
        ["Total Papers", str(len(metrics_df))],
        ["Total Citations", str(metrics_df['Citations'].sum())]
    ]
    t = Table(overview_data, colWidths=[2*inch, 2*inch])
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (0, -1), colors.grey),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
    ]))
    content.append(t)
    content.append(Spacer(1, 15))

    # 2. Papers by Institution
    content.append(Paragraph("2. Papers by Institution", styles['CustomHeading2']))
    inst_papers = metrics_df.groupby('Institution')['Title'].count()
    inst_data = [[inst, str(count)] for inst, count in inst_papers.items()]
    t = Table([["Institution", "Papers"]] + inst_data, colWidths=[3*inch, 1*inch])
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
    ]))
    content.append(t)
    content.append(Spacer(1, 15))

    # 3. Network Metrics
    content.append(Paragraph("3. Network Centrality Measures", styles['CustomHeading2']))
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
        content.append(Paragraph(f"{name}", styles['Normal']))
        mean_val = metrics_df[col].mean()
        max_val = metrics_df[col].max()
        
        # Top 5 papers table
        top_papers = metrics_df.nlargest(5, col)
        table_data = [["Paper Title", "Institution", "Score"]]
        for _, row in top_papers.iterrows():
            table_data.append([
                row['Title'][:50] + "...",
                row['Institution'],
                f"{row[col]:.4f}"
            ])
        
        t = Table(table_data, colWidths=[4*inch, 2*inch, 1*inch])
        t.setStyle(TableStyle([
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ]))
        content.append(t)
        content.append(Spacer(1, 15))

    # 4. Cross-institution Analysis
    content.append(Paragraph("4. Cross-institution Analysis", styles['CustomHeading2']))
    institution_matrix = analysis_results['institution_matrix']
    matrix_data = [['']+list(institution_matrix.columns)]
    for idx, row in institution_matrix.iterrows():
        matrix_data.append([idx] + [str(val) for val in row])
    
    t = Table(matrix_data)
    t.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('BACKGROUND', (0, 0), (0, -1), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('TEXTCOLOR', (0, 0), (0, -1), colors.whitesmoke),
    ]))
    content.append(t)

    # Copy and embed plots
    plots_dir = os.path.join(report_path, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    content.append(Paragraph("5. Visualizations", styles['CustomHeading2']))
    for filename in os.listdir(plot_dir):
        if filename.endswith('.png'):
            src_path = os.path.join(plot_dir, filename)
            dst_path = os.path.join(plots_dir, filename)
            shutil.copy2(src_path, dst_path)
            
            # Add plot to PDF
            img = Image(dst_path, width=6*inch, height=4*inch)
            content.append(img)
            content.append(Paragraph(f"Figure: {filename[:-4]}", styles['Normal']))
            content.append(Spacer(1, 15))

    # Build PDF
    doc.build(content)
    
    print(f"\nComprehensive report generated successfully at: {report_path}")
    return report_path