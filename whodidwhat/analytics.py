import networkx as nx
from .nlp_utils import compute_valence
from collections import defaultdict



################################################################################################
## Stuff to extract svos. 
################################################################################################

def export_hypergraphs(df):
    """
    Extracts hypergraphs from the DataFrame where Semantic-Syntactic is 0,
    allowing duplicates across sentences but not within the same sentence.
    
    Args:
        df (pandas.DataFrame): DataFrame containing the hypergraph data
        
    Returns:
        list: A list of hypergraph strings, potentially with duplicates
    """
    # Assume we have a 'sentence_id' column in the DataFrame
    # If not, you'll need to add logic to identify sentence boundaries

    hypergraphs = []
    sentence_hypergraphs = defaultdict(set)

    # Filter rows where Semantic-Syntactic is 0
    filtered_df = df[df['Semantic-Syntactic'] == 0]

    # Group by sentence_id and collect hypergraphs
    for svo_id, group in filtered_df.groupby('svo_id'):
        for hypergraph in group['Hypergraph']:
            if hypergraph != 'N/A' and hypergraph not in sentence_hypergraphs[svo_id]:
                hypergraphs.append(hypergraph)
                sentence_hypergraphs[svo_id].add(hypergraph)

    return hypergraphs


def export_subj(df):
    """
    Extracts a set of all unique subjects from the DataFrame.
    Each element in the set is a tuple of (subject, valence).
    """
    # Filter rows where WDW is 'Who' (indicative of subjects)
    df_subjects = df[df['WDW'] == 'Who']
    # Get unique subjects from 'Node 1'
    subjects = df_subjects['Node 1'].unique()
    # Compute valence and create a set of tuples
    subject_tuples = set()
    for subj in subjects:
        valence = compute_valence(subj)
        subject_tuples.add((subj, valence))
    return subject_tuples


def export_obj(df):
    """
    Extracts a set of all unique objects from the DataFrame.
    Each element in the set is a tuple of (object, valence).
    """
    # Filter rows where WDW2 is 'What' (indicative of objects)
    df_objects = df[df['WDW2'] == 'What']
    # Get unique objects from 'Node 2'
    objects = df_objects['Node 2'].unique()
    # Compute valence and create a set of tuples
    object_tuples = set()
    for obj in objects:
        valence = compute_valence(obj)
        object_tuples.add((obj, valence))
    return object_tuples

def export_verb(df):
    """
    Extracts a set of all unique verbs from the DataFrame.
    Each element in the set is a tuple of (verb, valence).
    """
    # Verbs can be in 'Node 1' where WDW is 'Did' or in 'Node 2' where WDW2 is 'Did'
    verbs_node1 = df[df['WDW'] == 'Did']['Node 1']
    verbs_node2 = df[df['WDW2'] == 'Did']['Node 2']
    # Combine and get unique verbs
    verbs = pd.concat([verbs_node1, verbs_node2]).unique()
    # Compute valence and create a set of tuples
    verb_tuples = set()
    for verb in verbs:
        valence = compute_valence(verb)
        verb_tuples.add((verb, valence))
    return verb_tuples


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

    if object_filter is not None:
        df = filter_objects(df, object_filter)

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
