import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain
from nltk.corpus import wordnet as wn
from whodidwhat.nlp_utils import compute_valence


def plot_svo_graph(svo_list, subject_filter=None):
    """
    Plot a graph of the SVO triples with subjects on the left, verbs in the center, and objects on the right.

    Args:
        svo_list (list): A list of SVO triples.
        subject_filter (str): A subject to filter the graph by.
    """
    G = svo_to_graph(svo_list, subject_filter)
    plot_graph(G)


def add_node_with_type(G, node_id, label, node_type):
    """
    Add a node to the graph with a specific type.
    If the node already exists, update its type to include the new type.
    """
    if G.has_node(node_id):
        if 'type' in G.nodes[node_id]:
            G.nodes[node_id]['type'].add(node_type)
        else:
            G.nodes[node_id]['type'] = set([node_type])
    else:
        G.add_node(node_id, type=set([node_type]), label=label)
        
def svo_to_graph(df, subject_filter=None):
    """
    Convert a pandas DataFrame of SVO data into a graph.
    Optionally filters the data based on the subject_filter.
    """
    G = nx.Graph()

    if subject_filter is not None:
        # Identify all unique hypergraphs that contain the subject_filter in 'Node 1'
        relevant_hypergraphs = df.loc[df['Node 1'] == subject_filter, 'Hypergraph'].unique()
        # Filter the DataFrame to include all rows that are in these hypergraphs
        df = df[df['Hypergraph'].isin(relevant_hypergraphs)]

    for index, row in df.iterrows():
        node1 = row['Node 1']
        wdw1 = row['WDW']
        node2 = row['Node 2']
        wdw2 = row['WDW2']
        hypergraph = row['Hypergraph']
        sem_synt = row['Semantic-Syntactic']

        # Determine node types based on WDW and WDW2
        node1_type = 'subject' if wdw1 == 'Who' else 'verb' if wdw1 == 'Did' else 'object'
        node2_type = 'subject' if wdw2 == 'Who' else 'verb' if wdw2 == 'Did' else 'object'

        # Create unique node IDs to differentiate between subjects, verbs, and objects
        node1_id = node1 + '_' + node1_type[0]
        node2_id = node2 + '_' + node2_type[0]

        # Add nodes with labels and types
        add_node_with_type(G, node1_id, label=node1, node_type=node1_type)
        add_node_with_type(G, node2_id, label=node2, node_type=node2_type)

        # Determine relation type based on 'Semantic-Syntactic' column
        relation_type = 'synonym' if sem_synt == 1 else 'syntactic'

        # Add edge with attributes
        G.add_edge(node1_id, node2_id, relation=relation_type, hypergraph=hypergraph)

    return G




# Include your plot_graph function here
def plot_graph(G):
    """
    Plot the SVO graph with subjects on the left, verbs in the center, and objects on the right,
    incorporating node valence for coloring, edge weights, and rectangular labels.
    
    Args:
        G (networkx.Graph): The graph to plot.
    """
    num_nodes = G.number_of_nodes()
    figsize = (10 + num_nodes * 0.05, 6 + num_nodes * 0.2)
    plt.figure(figsize=figsize)
    
    # Get nodes by type
    subjects = [node for node, attr in G.nodes(data=True) if 'subject' in attr.get('type', set())]
    verbs = [node for node, attr in G.nodes(data=True) if 'verb' in attr.get('type', set())]
    objects = [node for node, attr in G.nodes(data=True) if 'object' in attr.get('type', set())]
    
    # Assign node colors based on valence using compute_valence()
    node_colors = []
    valences = {}  # To store valences for later use
    for node in G.nodes():
        label = G.nodes[node].get('label', node)
        valence = compute_valence(label)
        valences[node] = valence  # Store valence for later
        if valence == 'positive':
            node_colors.append("#1f77b4")  # Blue
        elif valence == 'negative':
            node_colors.append("#d62728")  # Red
        else:
            node_colors.append("#7f7f7f")  # Grey
    
    # Calculate maximum number of nodes to align y positions
    max_nodes = max(len(subjects), len(verbs), len(objects))
    
    # Set positions
    pos = {}
    y_max = max_nodes
    y_min = 1  # Start from 1 to avoid zero position
    
    # Helper function to set positions
    def set_positions(nodes, x_pos):
        n = len(nodes)
        if n > 1:
            y_positions = np.linspace(y_max, y_min, n)
        else:
            y_positions = [(y_max + y_min) / 2]
        for i, node in enumerate(nodes):
            pos[node] = (x_pos, y_positions[i])
    
    # Set positions for subjects, verbs, and objects
    set_positions(subjects, x_pos=0)
    set_positions(verbs, x_pos=1)
    set_positions(objects, x_pos=2)
    
    # Collect all y positions for setting plot limits
    all_y_positions = [pos[node][1] for node in pos]
    min_y = min(all_y_positions) - 1  # Padding
    max_y = max(all_y_positions) + 1  # Padding
    
    # Determine if the graph is weighted
    is_weighted = any('weight' in data for _, _, data in G.edges(data=True))
    
    # Get edge weights; default to 1 if not specified
    edge_counts = nx.get_edge_attributes(G, 'weight')
    if not edge_counts:
        edge_counts = {edge: 1 for edge in G.edges()}
    
    # Calculate min and max edge widths
    max_count = max(edge_counts.values())
    # Adjusted min and max widths to be closer
    min_width = (6 if is_weighted else 3) * (figsize[0] / 12)
    max_width = (10 if is_weighted else 3) * (figsize[0] / 12)
    
    # Draw edges with varying thickness, colors, and styles
    for start, end, data in G.edges(data=True):
        count = edge_counts.get((start, end), 1)
        start_label = G.nodes[start].get('label', start)
        end_label = G.nodes[end].get('label', end)
        
        start_valence = valences[start]
        end_valence = valences[end]
        
        # Determine edge color based on valence of start and end nodes
        if data.get('relation') == 'synonym':
            color = '#009E73'  # Green
        else:
            if start_valence == 'positive' and end_valence == 'positive':
                color = "#1f77b4"  # Blue
            elif start_valence == 'negative' and end_valence == 'negative':
                color = "#d62728"  # Red
            elif (start_valence == 'positive' and end_valence == 'negative') or (start_valence == 'negative' and end_valence == 'positive'):
                color = "#9467bd"  # Purple
            elif (start_valence == 'positive' and end_valence == 'neutral') or (end_valence == 'positive' and start_valence == 'neutral'):
                color = "#b4cad6"  # Grayish blue
            elif (start_valence == 'negative' and end_valence == 'neutral') or (end_valence == 'negative' and start_valence == 'neutral'):
                color = "#dc9f9e"  # Grayish red
            else:
                color = "#7f7f7f"  # Grey
        
        # Calculate edge width
        edge_width = min_width + (count / max_count) * (max_width - min_width)
        
        # Determine edge style
        start_type = G.nodes[start].get('type', set())
        end_type = G.nodes[end].get('type', set())
        if 'subject' in start_type and 'subject' in end_type:
            style = 'dashed'
        elif 'object' in start_type and 'object' in end_type:
            style = 'dashed'
        else:
            style = 'solid'
    
        nx.draw_networkx_edges(
            G, pos, edgelist=[(start, end)], width=edge_width, alpha=0.45, edge_color=color, arrows=False
        )
    
    # Draw labels with custom rectangular backgrounds
    # Calculate label font size
    width, height = figsize
    reference_width = 10  # Reference width for scaling
    base_font_size = 10 - height * 0.07
    scaled_font_size = base_font_size * (width / reference_width)
    
    # Prepare labels using 'label' attribute
    labels_dict = {node: attr.get('label', node) for node, attr in G.nodes(data=True)}
    labels = nx.draw_networkx_labels(G, pos, labels=labels_dict, font_size=scaled_font_size, font_color="white")
    
    # Customize label backgrounds to be rectangular
    for node, label in labels.items():
        color = node_colors[list(G.nodes()).index(node)]
        label.set_bbox(
            dict(
                facecolor=color,
                edgecolor="none",
                alpha=0.9,
                pad=0.6,
                boxstyle="square",  # Makes the label background rectangular
                )
            )
    
    # Add group titles
    y_title = max_y + 0.5
    plt.text(0, y_title, 'Who', fontsize=16, ha='center')
    plt.text(1, y_title, 'Did', fontsize=16, ha='center')
    plt.text(2, y_title, 'What', fontsize=16, ha='center')
    
    plt.axis('off')
    plt.ylim(min_y, y_title + 1)
    plt.show()
