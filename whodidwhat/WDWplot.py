import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations, chain
from nltk.corpus import wordnet as wn
from whodidwhat.nlp_utils import compute_valence


def plot_svo_graph(svo_list, subject_filter=None, object_filter=None):
    """
    Plot a graph of SVO data. 

    Args:
        svo_list (dataframe): a pandas DataFrame of SVO data.
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

def svo_to_graph(df, subject_filter=None, object_filter=None):
    """
    Convert a pandas DataFrame of SVO data into a graph.
    Optionally filters the data based on the subject_filter.
    """
    G = nx.Graph()

    if subject_filter is not None:
        relevant_hypergraphs_subject = df.loc[
            df['Node 1'].str.contains(subject_filter, case=False, na=False), 'Hypergraph'
        ].unique()
        df = df[df['Hypergraph'].isin(relevant_hypergraphs_subject)]

    # Apply object filter with partial matching
    if object_filter is not None:
        relevant_hypergraphs_object = df.loc[
            df['Node 2'].str.contains(object_filter, case=False, na=False), 'Hypergraph'
        ].unique()
        df = df[df['Hypergraph'].isin(relevant_hypergraphs_object)]

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

        if G.has_edge(node1_id, node2_id):
            # Edge exists, update the 'relation' attribute
            existing_relation = G.edges[node1_id, node2_id].get('relation', set())
            if isinstance(existing_relation, str):
                existing_relation = set([existing_relation])
            existing_relation.add(relation_type)
            G.edges[node1_id, node2_id]['relation'] = existing_relation
            # Also update hypergraph
            existing_hypergraph = G.edges[node1_id, node2_id].get('hypergraph', set())
            if isinstance(existing_hypergraph, str):
                existing_hypergraph = set([existing_hypergraph])
            existing_hypergraph.add(hypergraph)
            G.edges[node1_id, node2_id]['hypergraph'] = existing_hypergraph
            # Increment weight for syntactic relations
            if relation_type == 'syntactic':
                G.edges[node1_id, node2_id]['weight'] += 1
        else:
            # Initialize weight to 1 for syntactic relations
            weight = 1 if relation_type == 'syntactic' else 0
            G.add_edge(node1_id, node2_id, relation=set([relation_type]), hypergraph=set([hypergraph]), weight=weight)

    return G

def plot_graph(G):
    num_nodes = G.number_of_nodes()
    # Scale figure size based on number of nodes
    figsize = (max(12, 8 + (num_nodes**0.5) * 1.4), max(8, 5+ (num_nodes**0.5) * 1.15))
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
    spacing = 2  # Increased from 1.2 for more space between subgraphs

    def get_available_position(base_x, base_y, existing_positions, x_range=0.45, tolerance=0.85):
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
            x, y = get_available_position(-2, base_y + np.random.uniform(-0.4, 0.4), pos)  # Increased range
            pos[s] = (x, y)

        # Position verbs
        for v in v_nodes:
            x, y = get_available_position(0, base_y, pos, x_range=0.25)
            pos[v] = (x, y)

        # Position objects with more vertical spread
        for o in o_nodes:
            x, y = get_available_position(2, base_y + np.random.uniform(-0.4, 0.4), pos)  # Increased range
            pos[o] = (x, y)

    # Position remaining nodes
    remaining_nodes = set(G.nodes()) - set(pos.keys())
    for node in remaining_nodes:
        if node in subjects:
            base_x, base_y = -2, sentence_base_y + np.random.uniform(-spacing, spacing)
        elif node in verbs:
            base_x, base_y = 0, sentence_base_y + np.random.uniform(-spacing, spacing)
        else:
            base_x, base_y = 2, sentence_base_y + np.random.uniform(-spacing, spacing)
        x, y = get_available_position(base_x, base_y, pos)
        pos[node] = (x, y)

    # Collect weights of syntactic edges
    syntactic_weights = [data['weight'] for u, v, data in G.edges(data=True) if 'syntactic' in data.get('relation', set())]
    if syntactic_weights:
        min_weight = min(syntactic_weights)
        max_weight = max(syntactic_weights)
    else:
        min_weight = max_weight = 1

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

        relations = data.get('relation', set())
        if isinstance(relations, str):
            relations = set([relations])

        # Determine whether to plot the edge based on the relations
        if 'syntactic' in relations:
            # Plot syntactic edge with weight-based linewidth
            weight = data.get('weight', 1)
            if min_weight == max_weight:
                linewidth = 2  # Default linewidth if all weights are the same
            else:
                # Map weight to linewidth between 1 and 3
                linewidth = 1 + 2 * (weight - min_weight) / (max_weight - min_weight)

            # Determine edge color based on valence
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
                     linewidth=linewidth)

        elif 'synonym' in relations:
            # Only plot semantic edge if there is no syntactic edge
            linewidth = 2  # Semantic relations have fixed linewidth of 2
            color = '#009E73'  # Green

            plt.plot([pos[u][0], pos[v][0]],
                     [pos[u][1], pos[v][1]],
                     style,
                     color=color,
                     alpha=0.3,
                     linewidth=linewidth)

        else:
            # No valid relation to plot
            continue  # Skip this edge

    # Draw labels with rectangular backgrounds
    for node in G.nodes():
        x, y = pos[node]
        label = G.nodes[node].get('label', node)
        color = node_colors[node]

        bbox_props = dict(boxstyle="round,pad=0.2,rounding_size=0.2", fc=color, ec="none", alpha=0.8)
        plt.text(x, y, label, color='white',
                horizontalalignment='center',
                verticalalignment='center',
                bbox=bbox_props,
                fontsize=10)

    # Add column labels
    plt.text(-2, max(pos.values(), key=lambda x: x[1])[1] + 2.0, 'WHO', fontsize=20, ha='center')
    plt.text(0, max(pos.values(), key=lambda x: x[1])[1] + 2.0, 'DID', fontsize=20, ha='center')
    plt.text(2, max(pos.values(), key=lambda x: x[1])[1] + 2.0, 'WHAT', fontsize=20, ha='center')

    # Adjust the y-axis limits accordingly
    y_max = max(pos.values(), key=lambda x: x[1])[1]
    y_min = min(pos.values(), key=lambda x: x[1])[1]
    plt.ylim(y_min - 0.5, y_max + 1.5)
    x_min = min(pos.values(), key=lambda x: x[0])[0]
    x_max = max(pos.values(), key=lambda x: x[0])[0]
    plt.xlim(x_min - 0.1, x_max + 0.1)

    plt.axis('off')
    plt.tight_layout()
    plt.show()
