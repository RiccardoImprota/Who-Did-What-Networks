import networkx as nx
import matplotlib.pyplot as plt
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


def svo_to_graph(svo_list, subject_filter=None):
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
            # Add subject node
            G.add_node(subj, type='subject')
            sentence_subjects.append(subj)
            # Process preps
            for prep_phrase in preps:
                # Split prep phrase into preposition and object
                prep_parts = prep_phrase.split(' ', 1)
                if len(prep_parts) == 2:
                    prep, prep_obj = prep_parts
                    # Add prep object node
                    G.add_node(prep_obj, type='subject')  
                    # Add edge from subject to prep_obj
                    G.add_edge(subj, prep_obj)
                else:
                    G.add_node(prep_phrase, type='subject')
                    G.add_edge(subj, prep_phrase)
        # Process verbs
        for verb in verbs:
            G.add_node(verb, type='verb')
            # Connect subjects to verbs
            for subj, _ in subjects:
                G.add_edge(subj, verb)
        # Process objects
        for obj, preps in objects:
            # Add object node
            G.add_node(obj, type='object')
            sentence_objects.append(obj)
            # Process preps
            for prep_phrase in preps:
                prep_parts = prep_phrase.split(' ', 1)
                if len(prep_parts) == 2:
                    prep, prep_obj = prep_parts
                    G.add_node(prep_obj, type='object')
                    G.add_edge(obj, prep_obj)
                else:
                    G.add_node(prep_phrase, type='object')
                    G.add_edge(obj, prep_phrase)
            # Connect verbs to objects
            for verb in verbs:
                G.add_edge(verb, obj)
        # Add edges between subjects in the same sentence
        for i in range(len(sentence_subjects)):
            for j in range(i+1, len(sentence_subjects)):
                G.add_edge(sentence_subjects[i], sentence_subjects[j])
        # Add edges between objects in the same sentence
        for i in range(len(sentence_objects)):
            for j in range(i+1, len(sentence_objects)):
                G.add_edge(sentence_objects[i], sentence_objects[j])
    return G

def plot_graph(G): 
    """
    Plot the SVO graph with subjects on the left, verbs in the center, and objects on the right,
    incorporating node valence for coloring, edge weights, and rectangular labels.

    Args:
        G (networkx.DiGraph): The graph to plot.
    """
    figsize = (12, 14)
    plt.figure(figsize=figsize)

    # Get nodes by type
    subjects = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'subject']
    verbs = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'verb']
    objects = [node for node, attr in G.nodes(data=True) if attr.get('type') == 'object']

    # Get nodes by valence
    positive, negative, ambivalent = _valences('english')

    # Assign node colors based on valence
    node_colors = []
    for node in G.nodes():
        if node in positive:
            node_colors.append("#1f77b4")  # Blue
        elif node in negative:
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
        if start in positive and end in positive:
            color = "#1f77b4"  # Blue
        elif start in negative and end in negative:
            color = "#d62728"  # Red
        elif (start in positive and end in negative) or (start in negative and end in positive):
            color = "#9467bd"  # Purple
        elif (start in positive and end not in negative) or (end in positive and start not in negative):
            color = "#b4cad6"  # Grayish blue
        elif (start in negative and end not in negative) or (end in negative and start not in positive):
            color = "#dc9f9e"  # Grayish red
        else:
            color = "#7f7f7f"  # Grey

        # Calculate edge width
        edge_width = min_width + (count / max_count) * (max_width - min_width)

        nx.draw_networkx_edges(
            G, pos, edgelist=[edge], width=edge_width, alpha=0.45, edge_color=color, arrows=False
        )

    # Draw labels with custom rectangular backgrounds
    # Calculate label font size
    width, height = figsize
    reference_width = 10  # Reference width for scaling
    base_font_size = 10 - height * 0.07
    scaled_font_size = base_font_size * (width / reference_width)

    labels = nx.draw_networkx_labels(G, pos, font_size=scaled_font_size, font_color="white")

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

