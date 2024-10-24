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


def plot_graph(G):
    num_nodes = G.number_of_nodes()
    # Scale figure size based on number of nodes
    figsize = (max(12, 8 + (num_nodes**0.5) * 1.1), max(8, 5+ (num_nodes**0.5) * 0.9))
    plt.figure(figsize=figsize)

    # Separate nodes by type
    subjects = [node for node, attr in G.nodes(data=True) if 'subject' in attr.get('type', set())]
    verbs = [node for node, attr in G.nodes(data=True) if 'verb' in attr.get('type', set())]
    objects = [node for node, attr in G.nodes(data=True) if 'object' in attr.get('type', set())]

    # Compute valences first
    valences = {}
    node_colors = {}
    for node in G.nodes():
        label = G.nodes[node].get('label', node)
        valence = compute_valence(label)
        valences[node] = valence
        if valence == 'positive':
            node_colors[node] = "#1f77b4"  # Blue
        elif valence == 'negative':
            node_colors[node] = "#d62728"  # Red
        else:
            node_colors[node] = "#7f7f7f"  # Grey

    # Find connected components
    sentences = []
    for v in verbs:
        s_nodes = set()
        for s in subjects:
            if G.has_edge(s, v):
                s_nodes.add(s)
        o_nodes = set()
        for o in objects:
            if G.has_edge(v, o):
                o_nodes.add(o)
        if s_nodes or o_nodes:
            sentences.append((s_nodes, {v}, o_nodes))

    # Position calculation with increased spacing
    pos = {}
    spacing = 1.5  # Increased from 1.2 for more space between subgraphs

    def get_available_position(base_x, base_y, existing_positions, x_range=0.35, tolerance=0.35):
        x = base_x + np.random.uniform(-x_range, x_range)
        y = base_y
        attempts = 0
        while attempts < 50 and any(abs(ex_y - y) < tolerance and abs(ex_x - x) < tolerance
                                  for ex_x, ex_y in existing_positions.values()):
            x = base_x + np.random.uniform(-x_range, x_range)
            y += spacing * 0.3  # Increased from 0.25 for more vertical spacing
            attempts += 1
        return x, y

    # Initial positioning with more spread between sentences
    sentence_base_y = -2
    for s_nodes, v_nodes, o_nodes in sentences:
        # Increase vertical gap between subgraphs
        sentence_base_y += spacing * 1.2  # Increased multiplier for more space between subgraphs
        
        # Add some random variation to prevent perfect alignment
        base_y = sentence_base_y + np.random.uniform(-0.2, 0.2)

        # Position subjects with more vertical spread
        for s in s_nodes:
            x, y = get_available_position(-1, base_y + np.random.uniform(-0.4, 0.4), pos)  # Increased range
            pos[s] = (x, y)

        # Position verbs
        for v in v_nodes:
            x, y = get_available_position(0, base_y, pos, x_range=0.25)
            pos[v] = (x, y)

        # Position objects with more vertical spread
        for o in o_nodes:
            x, y = get_available_position(1, base_y + np.random.uniform(-0.4, 0.4), pos)  # Increased range
            pos[o] = (x, y)

    # Position remaining nodes
    remaining_nodes = set(G.nodes()) - set(pos.keys())
    for node in remaining_nodes:
        if node in subjects:
            base_x, base_y = -1, sentence_base_y + np.random.uniform(-spacing, spacing)
        elif node in verbs:
            base_x, base_y = 0, sentence_base_y + np.random.uniform(-spacing, spacing)
        else:
            base_x, base_y = 1, sentence_base_y + np.random.uniform(-spacing, spacing)
        x, y = get_available_position(base_x, base_y, pos)
        pos[node] = (x, y)

    # Draw edges
    for (u, v, data) in G.edges(data=True):
        # Determine edge style based on node types
        start_type = G.nodes[u].get('type', set())
        end_type = G.nodes[v].get('type', set())

        if ('subject' in start_type and 'subject' in end_type) or \
           ('object' in start_type and 'object' in end_type):
            style = '--'
        else:
            style = '-'

        # Determine edge color based on valence of start and end nodes
        start_valence = valences[u]
        end_valence = valences[v]

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

        plt.plot([pos[u][0], pos[v][0]],
                [pos[u][1], pos[v][1]],
                style,
                color=color,
                alpha=0.3,
                linewidth=1)

    # Draw labels with rectangular backgrounds
    for node in G.nodes():
        x, y = pos[node]
        label = G.nodes[node].get('label', node)
        color = node_colors[node]

        bbox_props = dict(boxstyle="square,pad=0.3", fc=color, ec="none", alpha=0.9)
        plt.text(x, y, label, color='white',
                horizontalalignment='center',
                verticalalignment='center',
                bbox=bbox_props,
                fontsize=10)


    # Add column labels (increase the 0.5 to a larger value, like 1.0)
    plt.text(-1, max(pos.values(), key=lambda x: x[1])[1] + 2.0, 'WHO', fontsize=20, ha='center')
    plt.text(0, max(pos.values(), key=lambda x: x[1])[1] + 2.0, 'DID', fontsize=20, ha='center')
    plt.text(1, max(pos.values(), key=lambda x: x[1])[1] + 2.0, 'WHAT', fontsize=20, ha='center')

    # And adjust the y-axis limits accordingly (change y_max + 1 to y_max + 1.5)
    y_max = max(pos.values(), key=lambda x: x[1])[1]
    y_min = min(pos.values(), key=lambda x: x[1])[1]
    plt.ylim(y_min - 0.5, y_max + 1.5)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
