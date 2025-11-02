import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import random
import statistics
from tqdm import tqdm
import traceback

# --- 0. Set Paths ---
# Please ensure the following path is correct
work_dir = '/Users/data'
output_dir = work_dir 

if not os.path.isdir(work_dir):
    print(f"Error: Working directory not found: {work_dir}")
    exit()
print(f"Working and Output Directory: {work_dir}")

# --- 1. Define Input Filenames ---
stable_network_filename = 'graphml'
guild_assignments_filename = 'csv'

stable_network_file = os.path.join(work_dir, stable_network_filename)
guild_assignments_file = os.path.join(work_dir, guild_assignments_filename)

# --- 2. Load Files ---
G_stable = None
guild_df = None

# Load Stable Network
try:
    if not os.path.exists(stable_network_file):
        raise FileNotFoundError(f"Stable network file not found: {stable_network_file}")
    G_stable = nx.read_graphml(stable_network_file)
    # Ensure node IDs are strings for consistency
    G_stable = nx.relabel_nodes(G_stable, str)
    print(f"Loaded Stable Network: {G_stable.number_of_nodes()} nodes, {G_stable.number_of_edges()} edges")
except Exception as e:
    print(f"Error: Failed to load Stable Network: {e}")
    traceback.print_exc()
    exit()

# Load Guild Assignments
try:
    if not os.path.exists(guild_assignments_file):
        raise FileNotFoundError(f"Guild assignment file not found: {guild_assignments_file}")
    guild_df = pd.read_csv(guild_assignments_file)
    print(f"Loaded Guild Assignment File: {guild_assignments_filename}")

    required_columns = ['OTU ID', 'Modularity Class']
    missing_columns = [col for col in required_columns if col not in guild_df.columns]
    if missing_columns:
        print(f"Warning: Guild assignment file is missing required columns: {', '.join(missing_columns)}.")
        guild_df = None
    else:
        guild_df['OTU ID'] = guild_df['OTU ID'].astype(str)
        # Optional: If 'Modularity Class' column might be string '1' instead of number 1
        # guild_df['Modularity Class'] = pd.to_numeric(guild_df['Modularity Class'], errors='coerce')

except Exception as e:
    print(f"Error: Failed to load Guild Assignment File: {e}")
    traceback.print_exc()
    exit()

# --- 3. Extract Guild 1 Members (based on 'Modularity Class' == 1) ---
guild1_otus = []
if guild_df is not None:
    # Assuming 'Modularity Class' is a numeric column
    guild1_df = guild_df[guild_df['Modularity Class'] == 1]

    if not guild1_df.empty:
        guild1_otus = guild1_df['OTU ID'].tolist()
        print(f"Extracted {len(guild1_otus)} OTUs based on 'Modularity Class == 1'.")
    else:
        print("Warning: No members found for 'Modularity Class == 1'.")

    if not guild1_otus:
         print("Warning: No Guild 1 OTU members were extracted.")
else:
    print("Error: Cannot extract Guild 1 members due to issues with the assignment file.")
    exit()

# Filter for Guild 1 OTUs that are actually present in the stable network
guild1_otus_in_stable = [otu for otu in guild1_otus if otu in G_stable.nodes()]

if not guild1_otus_in_stable:
    print(f"Error: None of the {len(guild1_otus)} extracted Guild 1 OTUs are in the stable network.")
    exit() # No point in continuing if Guild 1 has no nodes in the network
else:
    print(f"{len(guild1_otus_in_stable)} Guild 1 members are present in the stable network.")
    if len(guild1_otus_in_stable) < len(guild1_otus):
        print(f"  Note: {len(guild1_otus) - len(guild1_otus_in_stable)} original Guild 1 OTUs were not in the stable network.")

# --- 4. Extract Guild 1 Subgraph ---
G_guild1 = G_stable.subgraph(guild1_otus_in_stable).copy()
guild1_node_count = G_guild1.number_of_nodes()
guild1_edge_count = G_guild1.number_of_edges()
print(f"Extracted Guild 1 Subgraph: {guild1_node_count} nodes, {guild1_edge_count} edges")

# --- 5. Set Plotting Font (Optional) ---
# This section is for Chinese fonts; can be removed or adapted for other fonts if needed.
try:
    plt.rcParams['font.sans-serif'] = ['SimHei'] # Example: use 'Arial' for English
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    print(f"Warning: Failed to set font 'SimHei': {e}")

# --- Analysis 1: Calculate Guild 1 Topological Metrics ---
print("\n--- Calculating Guild 1 Topological Metrics ---")
guild1_metrics = {}
if guild1_node_count > 1:
    try:
        guild1_metrics['Density'] = nx.density(G_guild1)
        guild1_metrics['Avg_Clustering_Coefficient'] = nx.average_clustering(G_guild1)

        # Handle Average Path Length (for disconnected graphs)
        if nx.is_connected(G_guild1):
            guild1_metrics['Avg_Path_Length'] = nx.average_shortest_path_length(G_guild1)
        else:
            print("  Guild 1 subgraph is not connected. Calculating weighted average path length...")
            weighted_avg_path_length = 0
            total_nodes_in_components = 0
            num_components_gt_1_node = 0
            for component in nx.connected_components(G_guild1):
                subgraph = G_guild1.subgraph(component)
                num_nodes = subgraph.number_of_nodes()
                if num_nodes > 1:
                    num_components_gt_1_node += 1
                    try:
                        avg_len = nx.average_shortest_path_length(subgraph)
                        weighted_avg_path_length += avg_len * num_nodes
                        total_nodes_in_components += num_nodes
                    except Exception:
                        pass # Skip components where path length can't be computed
            if total_nodes_in_components > 0 and num_components_gt_1_node > 0:
                guild1_metrics['Avg_Path_Length'] = weighted_avg_path_length / total_nodes_in_components
            else:
                guild1_metrics['Avg_Path_Length'] = np.nan

        # Centrality Metrics (Average and Max)
        degree_centrality = nx.degree_centrality(G_guild1)
        betweenness_centrality = {}
        try:
            # For disconnected graphs, betweenness is calculated within components
            betweenness_centrality = nx.betweenness_centrality(G_guild1)
        except Exception as e_bc:
            print(f"  Error calculating Betweenness Centrality: {e_bc}")
            betweenness_centrality = {node: 0.0 for node in G_guild1.nodes()}

        guild1_metrics['Avg_Degree_Centrality'] = statistics.mean(degree_centrality.values()) if degree_centrality else 0
        guild1_metrics['Max_Degree_Centrality'] = max(degree_centrality.values()) if degree_centrality else 0
        guild1_metrics['Avg_Betweenness_Centrality'] = statistics.mean(betweenness_centrality.values()) if betweenness_centrality else 0
        guild1_metrics['Max_Betweenness_Centrality'] = max(betweenness_centrality.values()) if betweenness_centrality else 0

        # Print calculated metrics
        for key, value in guild1_metrics.items():
            print(f"  {key}: {value:.4f}" if not np.isnan(value) else f"  {key}: NaN")

    except Exception as e:
        print(f"Error: Failed to calculate Guild 1 topology metrics: {e}")
        traceback.print_exc()
else:
    print("Guild 1 subgraph has < 2 nodes. Skipping topology calculations.")

# --- Analysis 2: Comparison with Random Networks ---
print("\n--- Comparing with Random Networks ---")
if guild1_node_count > 1:
    n_simulations = 1000 # Number of simulations
    print(f"Running {n_simulations} random network simulations...")

    random_metrics_list = {k: [] for k in [ # Initialize metric lists
        'Density', 'Avg_Clustering_Coefficient', 'Avg_Path_Length',
        'Avg_Degree_Centrality', 'Max_Degree_Centrality',
        'Num_Components', 'Largest_Component_Size'
    ]}

    all_stable_nodes = list(G_stable.nodes())
    num_nodes_to_select = guild1_node_count

    if len(all_stable_nodes) >= num_nodes_to_select:
        for _ in tqdm(range(n_simulations), desc="Simulating random networks"):
            current_random_metrics = {k: np.nan for k in random_metrics_list.keys()} # Metrics for this simulation
            try:
                # Null Model: Select same number of nodes as Guild 1, chosen randomly from the stable network
                random_nodes = random.sample(all_stable_nodes, num_nodes_to_select)
                # Build subgraph using edges from the original stable network
                G_random = G_stable.subgraph(random_nodes).copy()

                if G_random.number_of_nodes() > 1:
                    current_random_metrics['Density'] = nx.density(G_random)
                    current_random_metrics['Avg_Clustering_Coefficient'] = nx.average_clustering(G_random)

                    # Handle Avg Path Length for random graph
                    if nx.is_connected(G_random):
                        current_random_metrics['Avg_Path_Length'] = nx.average_shortest_path_length(G_random)
                    else:
                        weighted_avg_path_length_rand = 0
                        total_nodes_in_components_rand = 0
                        num_components_gt_1_node_rand = 0
                        components_rand = list(nx.connected_components(G_random))
                        for component_rand in components_rand:
                            subgraph_rand = G_random.subgraph(component_rand)
                            num_nodes_rand = subgraph_rand.number_of_nodes()
                            if num_nodes_rand > 1:
                                num_components_gt_1_node_rand +=1
                                try:
                                    avg_len_rand = nx.average_shortest_path_length(subgraph_rand)
                                    weighted_avg_path_length_rand += avg_len_rand * num_nodes_rand
                                    total_nodes_in_components_rand += num_nodes_rand
                                except: pass
                        if total_nodes_in_components_rand > 0 and num_components_gt_1_node_rand > 0:
                            current_random_metrics['Avg_Path_Length'] = weighted_avg_path_length_rand / total_nodes_in_components_rand

                    # Centrality
                    degree_cen_rand = nx.degree_centrality(G_random)
                    current_random_metrics['Avg_Degree_Centrality'] = statistics.mean(degree_cen_rand.values()) if degree_cen_rand else 0
                    current_random_metrics['Max_Degree_Centrality'] = max(degree_cen_rand.values()) if degree_cen_rand else 0

                    # Connectivity
                    components_rand = list(nx.connected_components(G_random))
                    current_random_metrics['Num_Components'] = len(components_rand)
                    current_random_metrics['Largest_Component_Size'] = max(len(c) for c in components_rand) if components_rand else 0

            except Exception:
                 pass # Will be recorded as NaN

            # Append this simulation's metrics to the main list
            for key in random_metrics_list.keys():
                random_metrics_list[key].append(current_random_metrics[key])

        # Process collected random metrics (remove NaNs)
        processed_random_metrics = {key: [x for x in values if not np.isnan(x)]
                                    for key, values in random_metrics_list.items()}

        # Calculate summary statistics
        random_summary = {}
        print("\nRandom Network Topology Summary:")
        for key, values in processed_random_metrics.items():
            if values:
                random_summary[key] = {
                    'Mean': statistics.mean(values),
                    'StdDev': statistics.stdev(values) if len(values) > 1 else 0,
                    'Min': min(values), 'Max': max(values), 'Count': len(values)
                }
                print(f"  {key} (n={random_summary[key]['Count']}): Mean={random_summary[key]['Mean']:.4f}, StdDev={random_summary[key]['StdDev']:.4f}")
            else:
                random_summary[key] = {'Mean': np.nan, 'StdDev': np.nan, 'Min': np.nan, 'Max': np.nan, 'Count': 0}
                print(f"  {key}: No valid data.")

        # Save random network summary
        random_summary_df = pd.DataFrame(random_summary).T
        random_summary_filename = os.path.join(output_dir, 'Random_Network_Metrics_Summary.csv')
        random_summary_df.to_csv(random_summary_filename)
        print(f"Random network summary saved: {random_summary_filename}")

        # Calculate Z-scores for comparison
        comparison_results = {}
        print("\nGuild 1 vs. Random Network Comparison (Z-score):")
        for key in guild1_metrics.keys(): # Only compare metrics calculated for Guild 1
            if key in random_summary and not np.isnan(guild1_metrics.get(key, np.nan)) and \
               not np.isnan(random_summary[key].get('Mean', np.nan)) and \
               not np.isnan(random_summary[key].get('StdDev', np.nan)):

                mean_rand = random_summary[key]['Mean']
                stddev_rand = random_summary[key]['StdDev']
                guild1_val = guild1_metrics[key]

                if stddev_rand > 1e-9: # Avoid division by zero
                    z_score = (guild1_val - mean_rand) / stddev_rand
                    comparison_results[key] = {'Guild1_Value': guild1_val, 'Random_Mean': mean_rand, 'Random_StdDev': stddev_rand, 'Z_Score': z_score}
                    print(f"  {key}: Z={z_score:.4f}")
                else: # Standard deviation is near zero
                    is_diff = not np.isclose(guild1_val, mean_rand)
                    comparison_results[key] = {'Guild1_Value': guild1_val, 'Random_Mean': mean_rand, 'Random_StdDev': stddev_rand, 'Z_Score': 'StdDev~0', 'Is_Different': is_diff}
                    print(f"  {key}: StdDev~0. Is Guild 1 different: {is_diff}")
            else:
                comparison_results[key] = {'Guild1_Value': guild1_metrics.get(key, np.nan), 'Random_Mean': np.nan, 'Random_StdDev': np.nan, 'Z_Score': np.nan}
                print(f"  {key}: Cannot compare.")

        # Save Guild 1 metrics
        guild1_metrics_df = pd.DataFrame.from_dict(guild1_metrics, orient='index', columns=['Guild1_Value'])
        guild1_metrics_filename = os.path.join(output_dir, 'Guild1_Topology_Metrics.csv')
        guild1_metrics_df.to_csv(guild1_metrics_filename)
        print(f"Guild 1 topology metrics saved: {guild1_metrics_filename}")

        # Save comparison results
        comparison_df = pd.DataFrame(comparison_results).T
        comparison_filename = os.path.join(output_dir, 'Guild1_Random_Comparison.csv')
        comparison_df.to_csv(comparison_filename)
        print(f"Guild 1 vs. Random comparison saved: {comparison_filename}")
    else:
        print("Not enough nodes in stable network to sample. Skipping random network comparison.")
else:
    print("Guild 1 has < 2 nodes or was not extracted. Skipping random network comparison.")

# --- Analysis 3: Structural Robustness Test ---
print("\n--- Structural Robustness Test ---")
if guild1_node_count > 1:
    num_removal_steps = guild1_node_count

    # 1. Targeted Removal (by Degree) from Guild 1
    print("Running Guild 1 targeted removal simulation (by degree)...")
    robustness_degree_data = {'Removed_Count': [0], 'Largest_Component_Fraction': [1.0]}
    nodes_sorted_by_degree = sorted(G_guild1.degree(), key=lambda item: item[1], reverse=True)
    nodes_to_remove_targeted = [node for node, degree in nodes_sorted_by_degree]
    G_temp_targeted = G_guild1.copy()
    for i in tqdm(range(num_removal_steps), desc="Targeted removal (Guild 1)"):
        if not nodes_to_remove_targeted: break
        node = nodes_to_remove_targeted.pop(0)
        if node in G_temp_targeted: G_temp_targeted.remove_node(node)
        robustness_degree_data['Removed_Count'].append(i + 1)
        if G_temp_targeted.number_of_nodes() > 0:
            lcc_size = max(len(c) for c in nx.connected_components(G_temp_targeted))
            robustness_degree_data['Largest_Component_Fraction'].append(lcc_size / guild1_node_count)
        else:
            robustness_degree_data['Largest_Component_Fraction'].append(0)
    robustness_degree_df = pd.DataFrame(robustness_degree_data)
    robustness_degree_filename = os.path.join(output_dir, 'Guild1_Robustness_Targeted_Removal_by_Degree.csv')
    robustness_degree_df.to_csv(robustness_degree_filename, index=False)
    print(f"Guild 1 targeted removal data saved: {robustness_degree_filename}")

    # 2. Random Removal from Guild 1
    print("Running Guild 1 random removal simulation (average)...")
    n_removal_simulations = 100 # Number of random removal simulations
    all_random_removal_trajectories_guild1 = []
    for _ in tqdm(range(n_removal_simulations), desc="Random removal (Guild 1)"):
        G_temp_random_g1 = G_guild1.copy()
        nodes_to_remove_random_g1 = list(G_temp_random_g1.nodes())
        random.shuffle(nodes_to_remove_random_g1)
        trajectory = {'Removed_Count': [0], 'Largest_Component_Fraction': [1.0]}
        for i in range(num_removal_steps):
            if not nodes_to_remove_random_g1: break
            node = nodes_to_remove_random_g1.pop(0)
            if node in G_temp_random_g1: G_temp_random_g1.remove_node(node)
            trajectory['Removed_Count'].append(i + 1)
            if G_temp_random_g1.number_of_nodes() > 0:
                lcc_size = max(len(c) for c in nx.connected_components(G_temp_random_g1))
                trajectory['Largest_Component_Fraction'].append(lcc_size / guild1_node_count)
            else:
                trajectory['Largest_Component_Fraction'].append(0)
        all_random_removal_trajectories_guild1.append(pd.DataFrame(trajectory))

    # Calculate average trajectory for Guild 1 random removal
    robustness_random_guild1_avg_df = None
    if all_random_removal_trajectories_guild1:
        combined_df_g1 = pd.concat(all_random_removal_trajectories_guild1)
        robustness_random_guild1_avg_df = combined_df_g1.groupby('Removed_Count')['Largest_Component_Fraction'].mean().reset_index()
        robustness_random_guild1_avg_df.rename(columns={'Largest_Component_Fraction': 'Avg_Largest_Component_Fraction'}, inplace=True)
        robustness_random_guild1_avg_filename = os.path.join(output_dir, 'Guild1_Robustness_Random_Removal_Avg.csv')
        robustness_random_guild1_avg_df.to_csv(robustness_random_guild1_avg_filename, index=False)
        print(f"Guild 1 random removal average data saved: {robustness_random_guild1_avg_filename}")
    else:
        print("Failed to generate Guild 1 random removal data.")

    # 3. Random Removal from Full Stable Network (Control)
    print("Running full network random removal simulation (control)...")
    all_random_removal_trajectories_full = []
    full_network_node_count = G_stable.number_of_nodes()
    robustness_random_full_avg_df = None
    if full_network_node_count >= num_removal_steps:
        for _ in tqdm(range(n_removal_simulations), desc="Random removal (Full Network)"):
            G_temp_random_full = G_stable.copy()
            nodes_to_remove_random_full = list(G_temp_random_full.nodes())
            random.shuffle(nodes_to_remove_random_full)
            trajectory_full = {'Removed_Count': [0], 'Largest_Component_Fraction': [1.0]}
            for i in range(num_removal_steps): # Only remove up to Guild 1's node count
                if not nodes_to_remove_random_full: break
                node = nodes_to_remove_random_full.pop(0)
                if node in G_temp_random_full: G_temp_random_full.remove_node(node)
                trajectory_full['Removed_Count'].append(i + 1)
                if G_temp_random_full.number_of_nodes() > 0:
                    lcc_size = max(len(c) for c in nx.connected_components(G_temp_random_full))
                    trajectory_full['Largest_Component_Fraction'].append(lcc_size / full_network_node_count) # Fraction of full network
                else:
                    trajectory_full['Largest_Component_Fraction'].append(0)
            all_random_removal_trajectories_full.append(pd.DataFrame(trajectory_full))

        if all_random_removal_trajectories_full:
            combined_df_full = pd.concat(all_random_removal_trajectories_full)
            robustness_random_full_avg_df = combined_df_full.groupby('Removed_Count')['Largest_Component_Fraction'].mean().reset_index()
            robustness_random_full_avg_df.rename(columns={'Largest_Component_Fraction': 'Avg_Largest_Component_Fraction'}, inplace=True)
            print("Full network random removal average data calculated.")
        else:
            print("Failed to generate full network random removal data.")
    else:
        print("Full network has insufficient nodes. Skipping control random removal simulation.")

    # 4. Plot Robustness Comparison
    print("\nPlotting robustness comparison...")
    plt.figure(figsize=(10, 7))
    # Plot Guild 1 Targeted Removal
    plt.plot(robustness_degree_df['Removed_Count'] / guild1_node_count,
             robustness_degree_df['Largest_Component_Fraction'],
             marker='o', linestyle='-', color='red', label=f'Guild 1 ({guild1_node_count} nodes) - Targeted (Degree)')
    # Plot Guild 1 Random Removal
    if robustness_random_guild1_avg_df is not None:
        plt.plot(robustness_random_guild1_avg_df['Removed_Count'] / guild1_node_count,
                 robustness_random_guild1_avg_df['Avg_Largest_Component_Fraction'],
                 marker='s', linestyle='--', color='blue', label='Guild 1 - Random (Avg)')
    # Plot Full Network Random Removal (Control)
    if robustness_random_full_avg_df is not None:
        plt.plot(robustness_random_full_avg_df['Removed_Count'] / guild1_node_count, # X-axis relative to Guild 1 size
                 robustness_random_full_avg_df['Avg_Largest_Component_Fraction'],
                 marker='^', linestyle=':', color='grey', label=f'Full Network ({full_network_node_count} nodes) - Random (Avg)')

    plt.xlabel("Fraction of Nodes Removed (relative to Guild 1 size)")
    plt.ylabel("Fraction of Nodes in Largest Connected Component")
    plt.title("Network Robustness Comparison")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xlim(0, 1.0); plt.ylim(0, 1.05)

    robustness_plot_filename = os.path.join(output_dir, 'Network_Robustness_Comparison_Plot.png')
    try:
        plt.savefig(robustness_plot_filename, dpi=300, bbox_inches='tight')
        print(f"Robustness comparison plot saved: {robustness_plot_filename}")
    except Exception as plot_e:
        print(f"Error: Failed to save robustness plot: {plot_e}")
    plt.close()

else:
    print("Guild 1 has < 2 nodes. Skipping structural robustness test.")

# --- End ---
print("\nScript execution finished.")