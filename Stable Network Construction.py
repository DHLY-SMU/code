import pandas as pd
import networkx as nx
import os
import sys

# --- 1. Configuration ---
# Please set your input and output paths here.
INPUT_DIR = '/input_data' 
OUTPUT_DIR = '/output_data' 

# Input filenames
GDM_EDGES_FILE = 'GDM_GDMfeces_edgelist.csv'
NONGDM_EDGES_FILE = 'NonGDM_NonGDMfeces_edgelist.csv'

# Output filenames
STABLE_EDGES_CSV = 'stable_edges_GDM_vs_NonGDM.csv'
STABLE_NETWORK_GRAPHML = 'stable_network_GDM_vs_NonGDM.graphml'

# --- 2. Helper Functions ---

def load_data(gdm_path, nongdm_path):
    """
    Loads GDM and NonGDM edge list files.
    """
    try:
        print(f"Loading GDM file: {gdm_path}")
        edge_df_gdm = pd.read_csv(gdm_path)
        
        print(f"Loading NonGDM file: {nongdm_path}")
        edge_df_nongdm = pd.read_csv(nongdm_path)
        
        print("Successfully loaded GDM and NonGDM edge lists.")
        return edge_df_gdm, edge_df_nongdm
        
    except FileNotFoundError as e:
        print(f"Error: File not found.")
        if not os.path.exists(gdm_path):
            print(f"  - Missing: {gdm_path}")
        if not os.path.exists(nongdm_path):
            print(f"  - Missing: {nongdm_path}")
        print(f"Full error details: {e}")
        sys.exit(1) # Exit script if files are missing
    except Exception as e:
        print(f"An error occurred while loading files: {e}")
        sys.exit(1)

def standardize_edge_pairs(df):
    """
    Standardizes edge pairs to ensure (A, B) is treated the same as (B, A).
    Creates a 'pair' column with a sorted tuple of (Source, Target).
    """
    df['pair'] = df.apply(
        lambda row: tuple(sorted((str(row['Source']), str(row['Target'])))), 
        axis=1
    )
    return df

# --- 3. Main Analysis Script ---

def find_stable_network(gdm_df, nongdm_df, output_csv_path):
    """
    Identifies stable edges present in both networks with a consistent sign.
    Saves the stable edge list to a CSV file.
    """
    print("Standardizing edge pairs...")
    edge_df_gdm = standardize_edge_pairs(gdm_df)
    edge_df_nongdm = standardize_edge_pairs(nongdm_df)

    # Prepare dataframes for comparison
    gdm_edges = edge_df_gdm[['pair', 'Sign', 'Weight', 'FDR']].copy()
    gdm_edges['group'] = 'GDM'

    nongdm_edges = edge_df_nongdm[['pair', 'Sign', 'Weight', 'FDR']].copy()
    nongdm_edges['group'] = 'NonGDM'

    # Combine all edges and group by the standardized pair
    all_edges_combined = pd.concat([gdm_edges, nongdm_edges])
    grouped_edges = all_edges_combined.groupby('pair')

    print("Identifying stable edges...")
    stable_edges_list = []
    for pair, group_df in grouped_edges:
        # Condition 1: Edge must exist in both groups
        if len(group_df['group'].unique()) == 2:
            # Condition 2: The sign (e.g., positive/negative) must be consistent
            signs = group_df['Sign'].unique()
            if len(signs) == 1:
                # This is a stable edge
                stable_edges_list.append({
                    'Source': pair[0],
                    'Target': pair[1],
                    'Sign': signs[0],
                    'AvgWeight_GDM': group_df[group_df['group'] == 'GDM']['Weight'].mean(),
                    'AvgWeight_NonGDM': group_df[group_df['group'] == 'NonGDM']['Weight'].mean(),
                    'FDR_GDM': group_df[group_df['group'] == 'GDM']['FDR'].min(),
                    'FDR_NonGDM': group_df[group_df['group'] == 'NonGDM']['FDR'].min()
                })

    stable_edge_final_df = pd.DataFrame(stable_edges_list)
    print(f"Found {len(stable_edge_final_df)} stable edges common to both groups.")

    # Save the stable edge list
    stable_edge_final_df.to_csv(output_csv_path, index=False)
    print(f"Stable edge list saved to: {output_csv_path}")
    
    return stable_edge_final_df

def create_graphml_from_stable_edges(stable_edge_df, output_graphml_path):
    """
    Loads the stable edge list and converts it into a GraphML network file.
    """
    print(f"Creating network from stable edges...")
    G_stable = nx.Graph()

    # Add all unique nodes from the stable edge list
    stable_otus = pd.concat([stable_edge_df['Source'], stable_edge_df['Target']]).unique()
    G_stable.add_nodes_from(stable_otus)
    print(f"Added {len(stable_otus)} nodes to the stable network.")

    # Add edges and their attributes
    added_edges_count = 0
    for index, row in stable_edge_df.iterrows():
        source_node = str(row['Source'])
        target_node = str(row['Target'])
        
        G_stable.add_edge(
            source_node,
            target_node,
            sign=row['Sign'],
            # Other attributes can be added here if needed
            # avg_weight_gdm=row['AvgWeight_GDM'],
            # avg_weight_nongdm=row['AvgWeight_NonGDM'],
        )
        added_edges_count += 1

    print(f"Added {added_edges_count} edges to the stable network.")
    print(f"Final stable network: {G_stable.number_of_nodes()} nodes, {G_stable.number_of_edges()} edges.")

    # Save the stable network as a GraphML file
    try:
        nx.write_graphml(G_stable, output_graphml_path)
        print(f"Stable network saved as GraphML file: {output_graphml_path}")
    except Exception as e:
        print(f"Error saving GraphML file: {e}")

# --- 4. Execution ---

if __name__ == "__main__":
    
    # Ensure output directory exists
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"Created output directory: {OUTPUT_DIR}")

    # Define full file paths
    gdm_file_path = os.path.join(INPUT_DIR, GDM_EDGES_FILE)
    nongdm_file_path = os.path.join(INPUT_DIR, NONGDM_EDGES_FILE)
    stable_edge_output_path = os.path.join(OUTPUT_DIR, STABLE_EDGES_CSV)
    stable_network_output_file = os.path.join(OUTPUT_DIR, STABLE_NETWORK_GRAPHML)

    # --- Step 1: Load Data ---
    edge_df_gdm, edge_df_nongdm = load_data(gdm_file_path, nongdm_file_path)

    # --- Step 2: Find Stable Edges and save CSV ---
    stable_df = find_stable_network(edge_df_gdm, edge_df_nongdm, stable_edge_output_path)

    # --- Step 3: Create and save GraphML from stable edges ---
    if not stable_df.empty:
        create_graphml_from_stable_edges(stable_df, stable_network_output_file)
    else:
        print("No stable edges were found. Skipping GraphML creation.")

    print("Analysis complete.")