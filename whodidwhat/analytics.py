import networkx as nx


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

def filter_subjects(df, subject_term):
    """
    Filters the DataFrame to keep rows where the 'svo_id' corresponds to entries
    where 'Node 1' contains the subject term and 'WDW' indicates a subject ('Who').

    Parameters:
    - df: The input DataFrame.
    - subject_term: The term to search for in subjects.

    Returns:
    - A filtered DataFrame containing only rows with matching 'svo_id's.
    """
    # Identify 'svo_id's where 'Node 1' contains the subject term and 'WDW' is 'Who'
    subject_svo_ids = set(df[
        (df['Node 1'].str.contains(subject_term, case=False, na=False)) &
        (df['WDW'] == 'Who')
    ]['svo_id'].unique())

    # Filter the DataFrame to only include rows with these 'svo_id's
    filtered_df = df[df['svo_id'].isin(subject_svo_ids)]
    return filtered_df

def filter_objects(df, object_term):
    """
    Filters the DataFrame to keep rows where the 'svo_id' corresponds to entries
    where 'Node 2' contains the object term and 'WDW' indicates an object ('What').

    Parameters:
    - df: The input DataFrame.
    - subject_term: The term to search for in subjects.

    Returns:
    - A filtered DataFrame containing only rows with matching 'svo_id's.
    """
    # Identify 'svo_id's where 'Node 12' contains the subject term and 'WDW' is 'What'
    object_svo_ids = set(df[
        (df['Node 2'].str.contains(object_term, case=False, na=False)) &
        (df['WDW'] == 'What')
    ]['svo_id'].unique())
    # Filter the DataFrame to only include rows with these 'svo_id's
    filtered_df = df[df['svo_id'].isin(object_svo_ids)]
    return filtered_df



def svo_to_graph(df, subject_filter=None, object_filter=None):
    """
    Convert a pandas DataFrame of SVO data into a graph.
    Optionally filters the data based on the subject_filter.
    """
    G = nx.Graph()

    if subject_filter is not None:
        df = filter_subjects(df, subject_filter)

    # Apply object filter with partial matching
    if object_filter is not None:
        df = filter_subjects(df, object_filter)

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
