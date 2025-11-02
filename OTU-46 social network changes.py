# --- 0. Import Dependencies ---
import pandas as pd
import networkx as nx
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import traceback

print("--- Starting OTU-46 Neighbor Analysis and Visualization Script ---")

# --- 1. Define Paths and Filenames ---
# Assumes script is run from the directory containing the data files.
work_dir = '.' 
# Saves results to a subfolder named 'network_output'
output_dir = os.path.join(work_dir, 'network_output') 

gdm_network_file = os.path.join(work_dir, 'gdm_network.graphml')
nongdm_network_file = os.path.join(work_dir, 'nongdm_network.graphml')
guild_assignment_file = os.path.join(work_dir, 'stable_network_guild_assignments.csv')

# --- 2. Load Data ---
print("\n--- 2. Loading Networks and Guild Definitions ---")
try:
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output directory created at: {output_dir}")

    G_gdm = nx.read_graphml(gdm_network_file)
    G_gdm = nx.relabel_nodes(G_gdm, str) # Ensure node labels are strings
    print(f"Loaded GDM Network: {G_gdm.number_of_nodes()} N, {G_gdm.number_of_edges()} E")
    
    G_nongdm = nx.read_graphml(nongdm_network_file)
    G_nongdm = nx.relabel_nodes(G_nongdm, str) # Ensure node labels are strings
    print(f"Loaded NonGDM Network: {G_nongdm.number_of_nodes()} N, {G_nongdm.number_of_edges()} E")
    
    guild_assignments = pd.read_csv(guild_assignment_file)
    # Assumes OTU ID is in 'Label' and Guild ID in 'Modularity Class'
    guild_map = guild_assignments.set_index('Label')['Modularity Class'].astype(str).to_dict()
    print(f"Loaded Guild Definitions: {len(guild_map)} OTUs assigned.")

except FileNotFoundError as e:
    print(f"ERROR: File not found. Please check paths. {e}")
    exit()
except KeyError as e:
    print(f"ERROR: Guild assignment file missing expected column. {e}")
    exit()
except Exception as e:
    print(f"Error during data loading: {e}")
    exit()

# --- 3. Define Target OTU ---
target_otu_id = 'OTU-46'
# Automatically find the Guild ID for the target OTU from the map
target_guild_id = guild_map.get(target_otu_id, '1') # Default to '1' if not found

print(f"\n--- 3. Target OTU: {target_otu_id} (Identified as Guild: {target_guild_id}) ---")


# --- 4. Neighbor Analysis Function ---
def analyze_neighbors_full(target_otu, target_guild_id, G_gdm, G_nongdm, guild_map_dict):
    """
    Analyzes all neighbors of a target OTU across both GDM and Non-GDM networks.
    """
    results = []
    neighbors_gdm = set(G_gdm.neighbors(target_otu)) if target_otu in G_gdm else set()
    neighbors_nongdm = set(G_nongdm.neighbors(target_otu)) if target_otu in G_nongdm else set()

    all_neighbors = neighbors_gdm.union(neighbors_nongdm)
    if not all_neighbors:
        return results

    for neighbor in all_neighbors:
        neighbor_guild = guild_map_dict.get(neighbor, 'OutsideGuilds')
        in_gdm = neighbor in neighbors_gdm
        in_nongdm = neighbor in neighbors_nongdm

        status = "Unknown"
        sign_gdm, weight_gdm = "N/A", np.nan
        sign_nongdm, weight_nongdm = "N/A", np.nan

        if in_gdm and in_nongdm:
            status = "Common"
            sign_gdm = G_gdm.get_edge_data(target_otu, neighbor, default={}).get('sign', 'N/A')
            weight_gdm = G_gdm.get_edge_data(target_otu, neighbor, default={}).get('weight', np.nan)
            sign_nongdm = G_nongdm.get_edge_data(target_otu, neighbor, default={}).get('sign', 'N/A')
            weight_nongdm = G_nongdm.get_edge_data(target_otu, neighbor, default={}).get('weight', np.nan)
        elif in_gdm:
            status = "GDM_Only"
            sign_gdm = G_gdm.get_edge_data(target_otu, neighbor, default={}).get('sign', 'N/A')
            weight_gdm = G_gdm.get_edge_data(target_otu, neighbor, default={}).get('weight', np.nan)
        elif in_nongdm:
            status = "NonGDM_Only" # Corrected typo from NonGDM
            sign_nongdm = G_nongdm.get_edge_data(target_otu, neighbor, default={}).get('sign', 'N/A')
            weight_nongdm = G_nongdm.get_edge_data(target_otu, neighbor, default={}).get('weight', np.nan)

        results.append({
            'TargetOTU': target_otu,
            'TargetGuild': target_guild_id,
            'NeighborOTU': neighbor,
            'NeighborGuild': neighbor_guild,
            'Status': status,
            'Sign_GDM': sign_gdm,
            'Weight_GDM': weight_gdm,
            'Sign_NonGDM': sign_nongdm,
            'Weight_NonGDM': weight_nongdm
        })
    return results

# --- 5. Plotting Function (Saves to separate PDFs) ---
def plot_ego_networks_to_separate_pdfs(target_otu, full_neighbor_df, guild_map_dict, output_dir):
    """
    Generates two separate PDF ego network plots for the target OTU:
    1. GDM Network
    2. Non-GDM Network
    """
    print(f"--- 5.1. Generating PDF network plots for {target_otu} ---")
    otu_neighbors_df = full_neighbor_df[full_neighbor_df['TargetOTU'] == target_otu]
    if otu_neighbors_df.empty:
        print(f"  > WARNING: No neighbor data found for OTU {target_otu}. Plotting skipped.")
        return

    # --- Common plot settings ---
    plt.rcParams['axes.unicode_minus'] = False
    center_color = 'gold'
    neighbor_color = 'lightgrey'
    center_font_size = 12
    neighbor_font_size = 6
    
    legend_elements = [
        Line2D([0], [0], color='k', ls='solid', lw=2, label='Common Connection'),
        Line2D([0], [0], color='k', ls='dashed', lw=2, label='GDM-Specific Connection'),
        Line2D([0], [0], color='k', ls='dotted', lw=2, label='Control-Specific Connection'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Positive Correlation'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, label='Negative Correlation')
    ]

    # --- Plot 1: GDM Patient Network ---
    try:
        fig_gdm, ax_gdm = plt.subplots(figsize=(10, 10))
        ax_gdm.set_title(f"GDM Patient Network for {target_otu}", fontsize=16)
        G_gdm_ego = nx.Graph()
        gdm_connections = otu_neighbors_df[otu_neighbors_df['Status'].isin(['Common', 'GDM_Only'])]

        if not gdm_connections.empty:
            nodes_gdm = [target_otu] + gdm_connections['NeighborOTU'].tolist()
            G_gdm_ego.add_nodes_from(nodes_gdm)
            
            for _, row in gdm_connections.iterrows():
                G_gdm_ego.add_edge(target_otu, row['NeighborOTU'],
                                   color='red' if row['Sign_GDM'] == 'positive' else 'blue',
                                   style='solid' if row['Status'] == 'Common' else 'dashed',
                                   width=max(0.5, 2.5 * abs(row['Weight_GDM'])))
            
            pos_gdm = nx.spring_layout(G_gdm_ego, seed=42, k=0.8)
            node_colors_gdm = [center_color if n == target_otu else neighbor_color for n in G_gdm_ego.nodes()]
            node_sizes_gdm = [1200 if n == target_otu else 500 for n in G_gdm_ego.nodes()]
            
            nx.draw(G_gdm_ego, pos_gdm, ax=ax_gdm, with_labels=False,
                    node_color=node_colors_gdm, node_size=node_sizes_gdm,
                    edge_color=[d['color'] for u,v,d in G_gdm_ego.edges(data=True)],
                    width=[d['width'] for u,v,d in G_gdm_ego.edges(data=True)],
                    style=[d['style'] for u,v,d in G_gdm_ego.edges(data=True)],
                    alpha=0.9)
            
            nx.draw_networkx_labels(G_gdm_ego, pos_gdm, ax=ax_gdm,
                                    labels={n: n for n in G_gdm_ego.nodes() if n == target_otu},
                                    font_size=center_font_size)
            nx.draw_networkx_labels(G_gdm_ego, pos_gdm, ax=ax_gdm,
                                    labels={n: n for n in G_gdm_ego.nodes() if n != target_otu},
                                    font_size=neighbor_font_size)
        else:
            ax_gdm.text(0.5, 0.5, "No neighbors in GDM network", ha='center', va='center', fontsize=12)

        fig_gdm.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=10)
        gdm_pdf_path = os.path.join(output_dir, f"EgoNet_GDM_{target_otu}.pdf")
        fig_gdm.savefig(gdm_pdf_path, format='pdf', bbox_inches='tight')
        plt.close(fig_gdm)
        print(f"  ✅ GDM network plot saved to: {gdm_pdf_path}")
    except Exception as e_gdm_plot:
        print(f"  ❌ ERROR while plotting GDM network: {e_gdm_plot}")
        plt.close(fig_gdm)

    # --- Plot 2: Non-GDM (Control) Network ---
    try:
        fig_nongdm, ax_nongdm = plt.subplots(figsize=(10, 10))
        ax_nongdm.set_title(f"Control (Non-GDM) Network for {target_otu}", fontsize=16)
        G_nongdm_ego = nx.Graph()
        nongdm_connections = otu_neighbors_df[otu_neighbors_df['Status'].isin(['Common', 'NonGDM_Only'])] # Corrected typo here

        if not nongdm_connections.empty:
            nodes_nongdm = [target_otu] + nongdm_connections['NeighborOTU'].tolist()
            G_nongdm_ego.add_nodes_from(nodes_nongdm)

            for _, row in nongdm_connections.iterrows():
                G_nongdm_ego.add_edge(target_otu, row['NeighborOTU'],
                                      # *** CRITICAL FIX: 'Sign_NonGGDM' -> 'Sign_NonGDM' ***
                                      color='red' if row['Sign_NonGDM'] == 'positive' else 'blue',
                                      style='solid' if row['Status'] == 'Common' else 'dotted',
                                      width=max(0.5, 2.5 * abs(row['Weight_NonGDM'])))

            pos_nongdm = nx.spring_layout(G_nongdm_ego, seed=42, k=0.8)
            node_colors_nongdm = [center_color if n == target_otu else neighbor_color for n in G_nongdm_ego.nodes()]
            node_sizes_nongdm = [1200 if n == target_otu else 500 for n in G_nongdm_ego.nodes()]
            
            nx.draw(G_nongdm_ego, pos_nongdm, ax=ax_nongdm, with_labels=False,
                    node_color=node_colors_nongdm, node_size=node_sizes_nongdm,
                    edge_color=[d['color'] for u,v,d in G_nongdm_ego.edges(data=True)],
                    width=[d['width'] for u,v,d in G_nongdm_ego.edges(data=True)],
                    style=[d['style'] for u,v,d in G_nongdm_ego.edges(data=True)],
                    alpha=0.9)
            
            nx.draw_networkx_labels(G_nongdm_ego, pos_nongdm, ax=ax_nongdm,
                                    labels={n: n for n in G_nongdm_ego.nodes() if n == target_otu},
                                    font_size=center_font_size)
            nx.draw_networkx_labels(G_nongdm_ego, pos_nongdm, ax=ax_nongdm,
                                    labels={n: n for n in G_nongdm_ego.nodes() if n != target_otu},
                                    font_size=neighbor_font_size)
        else:
            ax_nongdm.text(0.5, 0.5, "No neighbors in Non-GDM network", ha='center', va='center', fontsize=12)

        fig_nongdm.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02), ncol=3, fontsize=10)
        nongdm_pdf_path = os.path.join(output_dir, f"EgoNet_NonGDM_{target_otu}.pdf")
        fig_nongdm.savefig(nongdm_pdf_path, format='pdf', bbox_inches='tight')
        plt.close(fig_nongdm)
        print(f"  ✅ Non-GDM network plot saved to: {nongdm_pdf_path}")
    except Exception as e_nongdm_plot:
        print(f"  ❌ ERROR while plotting Non-GDM network: {e_nongdm_plot}")
        traceback.print_exc() # Print detailed error
        plt.close(fig_nongdm)


# --- 6. Main Execution Block ---
try:
    if target_otu_id not in G_gdm and target_otu_id not in G_nongdm:
        print(f"  ❌ ERROR: Target OTU {target_otu_id} not found in either network. Analysis aborted.")
    else:
        # 1. Run Analysis
        print(f"\n--- 4. Analyzing neighbors for {target_otu_id} ---")
        neighbor_results = analyze_neighbors_full(target_otu_id, target_guild_id, G_gdm, G_nongdm, guild_map)
        
        if neighbor_results:
            neighbor_df_full = pd.DataFrame(neighbor_results)
            
            # 2. Save Analysis CSV
            neighbor_filename = os.path.join(output_dir, f'Neighbor_Analysis_{target_otu_id}.csv')
            neighbor_df_full.to_csv(neighbor_filename, index=False)
            print(f"Neighbor analysis data saved to: {neighbor_filename}")

            # 3. Print Summary Statistics
            print("\n--- 4.1. Summary Statistics ---")
            print(f"Connection Status Distribution:\n{neighbor_df_full['Status'].value_counts()}\n")

            # 4. Call Plotting Function
            plot_ego_networks_to_separate_pdfs(target_otu_id, neighbor_df_full, guild_map, output_dir)

        else:
            print(f"  > WARNING: {target_otu_id} has no neighbors in either network.")

except Exception as e:
    print(f"\n---A critical error occurred during script execution ---")
    traceback.print_exc()

print("\n--- Script finished ---")