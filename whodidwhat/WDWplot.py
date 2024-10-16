import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from whodidwhat.resources import _valences

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

    Args:
        G (networkx.Graph): The graph to add the node to.
        node_id (str): The unique identifier for the node.
        label (str): The label to display for the node.
        node_type (str): The type of the node (e.g., 'subject', 'object', 'verb').
    """
    if G.has_node(node_id):
        if 'type' in G.nodes[node_id]:
            G.nodes[node_id]['type'].add(node_type)
        else:
            G.nodes[node_id]['type'] = set([node_type])
    else:
        G.add_node(node_id, type=set([node_type]), label=label)

def svo_to_graph(svo_list, subject_filter=None):
    """
    Convert a list of SVO tuples into a graph where nodes represent subjects, verbs, and objects,
    and edges represent the relationships between them.

    Args:
        svo_list (list): A list of SVO tuples extracted from sentences.
        subject_filter (str, optional): If provided, only include sentences with this subject.

    Returns:
        networkx.Graph: The constructed graph.
    """
    G = nx.Graph()
    for svo in svo_list:
        subjects, verbs, objects = svo

        # Check if the specified subject is in the subjects list
        subject_names = [subj for subj, _ in subjects]
        if subject_filter is not None:
            subjects = [(subj, preps) for subj, preps in subjects if subj == subject_filter]
        if subject_filter is not None and subject_filter not in subject_names:
            continue

        # Keep track of nodes added in this sentence
        sentence_subjects = []
        sentence_objects = []

        # Process subjects
        for subj, preps in subjects:
            node_id = subj + '_s'  # Unique identifier for subject nodes
            add_node_with_type(G, node_id, label=subj, node_type='subject')
            sentence_subjects.append(node_id)
            # Process prepositions attached to the subject
            for prep_phrase in preps:
                # Split prep phrase into preposition and object
                prep_parts = prep_phrase.split(' ', 1)
                if len(prep_parts) == 2:
                    prep, prep_obj = prep_parts
                    prep_obj_id = prep_obj + '_s'
                    add_node_with_type(G, prep_obj_id, label=prep_obj, node_type='subject')
                    # Add edge from subject to preposition object
                    G.add_edge(node_id, prep_obj_id)
                else:
                    prep_obj_id = prep_phrase + '_s'
                    add_node_with_type(G, prep_obj_id, label=prep_phrase, node_type='subject')
                    G.add_edge(node_id, prep_obj_id)

        # Process verbs
        for verb in verbs:
            add_node_with_type(G, verb, label=verb, node_type='verb')
            # Connect subjects to verbs
            for subj, _ in subjects:
                subj_id = subj + '_s'
                G.add_edge(subj_id, verb)

        # Process objects
        for obj, preps in objects:
            node_id = obj + '_o'  # Unique identifier for object nodes
            add_node_with_type(G, node_id, label=obj, node_type='object')
            sentence_objects.append(node_id)
            # Process prepositions attached to the object
            for prep_phrase in preps:
                prep_parts = prep_phrase.split(' ', 1)
                if len(prep_parts) == 2:
                    prep, prep_obj = prep_parts
                    prep_obj_id = prep_obj + '_o'
                    add_node_with_type(G, prep_obj_id, label=prep_obj, node_type='object')
                    # Add edge from object to preposition object
                    G.add_edge(node_id, prep_obj_id)
                else:
                    prep_obj_id = prep_phrase + '_o'
                    add_node_with_type(G, prep_obj_id, label=prep_phrase, node_type='object')
                    G.add_edge(node_id, prep_obj_id)
            # Connect verbs to objects
            for verb in verbs:
                G.add_edge(verb, node_id)

        # Add edges between subjects in the same sentence
        for i in range(len(sentence_subjects)):
            for j in range(i+1, len(sentence_subjects)):
                G.add_edge(sentence_subjects[i], sentence_subjects[j])

        # Add edges between objects in the same sentence
        for i in range(len(sentence_objects)):
            for j in range(i+1, len(sentence_objects)):
                G.add_edge(sentence_objects[i], sentence_objects[j])

    ## Optional: Connect subject and object nodes of the same entity
    #for node in G.nodes():
    #    if node.endswith('_s'):
    #        counterpart = node[:-2] + '_o'
    #        if G.has_node(counterpart):
    #            G.add_edge(node, counterpart, style='dashed', color='grey')

    return G

def plot_graph(G):
    """
    Plot the SVO graph with subjects on the left, verbs in the center, and objects on the right,
    incorporating node valence for coloring, edge weights, and rectangular labels.

    Args:
        G (networkx.Graph): The graph to plot.
    """
    figsize = (12, 14)
    plt.figure(figsize=figsize)

    # Get nodes by type
    subjects = [node for node, attr in G.nodes(data=True) if 'subject' in attr.get('type', set())]
    verbs = [node for node, attr in G.nodes(data=True) if 'verb' in attr.get('type', set())]
    objects = [node for node, attr in G.nodes(data=True) if 'object' in attr.get('type', set())]

    # Get nodes by valence (assuming _valences function is defined)
    positive, negative, ambivalent = _valences('english')

    # Assign node colors based on valence
    node_colors = []
    for node in G.nodes():
        label = G.nodes[node].get('label', node)
        if label in positive:
            node_colors.append("#1f77b4")  # Blue
        elif label in negative:
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

    # Draw edges with varying thickness and colors
    for edge in G.edges():
        start, end = edge
        count = edge_counts.get(edge, 1)
        start_label = G.nodes[start].get('label', start)
        end_label = G.nodes[end].get('label', end)

        if start_label in positive and end_label in positive:
            color = "#1f77b4"  # Blue
        elif start_label in negative and end_label in negative:
            color = "#d62728"  # Red
        elif (start_label in positive and end_label in negative) or (start_label in negative and end_label in positive):
            color = "#9467bd"  # Purple
        elif (start_label in positive and end_label not in negative) or (end_label in positive and start_label not in negative):
            color = "#b4cad6"  # Grayish blue
        elif (start_label in negative and end_label not in negative) or (end_label in negative and start_label not in positive):
            color = "#dc9f9e"  # Grayish red
        else:
            color = "#7f7f7f"  # Grey

        # Calculate edge width
        edge_width = min_width + (count / max_count) * (max_width - min_width)

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
