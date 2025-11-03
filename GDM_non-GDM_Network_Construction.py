# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, gmean
from statsmodels.sandbox.stats.multicomp import multipletests
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import os
import traceback 

print("Network construction script started.")

# --- Setup Paths and Parameters ---
base_work_dir = '.'
output_dir = 'network_output' 

otu_file = 'otu_table.txt'
metadata_file = 'metadata.txt'

group_column = 'Group' 
gdm_group_name = 'GDMfeces' 
non_gdm_group_name = 'non-GDMfeces' 

min_prevalence_percentage = 10
min_mean_abundance = 0.00005
pseudocount = 1e-9
corr_threshold = 0.3
fdr_threshold = 0.05

try:
    os.chdir(base_work_dir)
    print(f"Working directory set to: {os.getcwd()}")
except FileNotFoundError:
    print(f"ERROR: Base working directory '{base_work_dir}' not found. Please check the path.")
    raise
except Exception as e:
    print(f"Error changing working directory: {e}")
    raise

os.makedirs(output_dir, exist_ok=True)
print(f"Output files will be saved to: {output_dir}")

# --- Load OTU Table ---
print(f"\nLoading OTU file: {otu_file}...")
try:
    otu_table = pd.read_csv(otu_file, sep='\t', index_col=0, header=0)
    print(f"Successfully loaded OTU file, original shape: {otu_table.shape}")
    otu_table = otu_table.T
    print(f"Transposed OTU table shape (samples x features): {otu_table.shape}")
except FileNotFoundError: print(f"ERROR: OTU file '{otu_file}' not found in '{base_work_dir}'!"); raise
except Exception as e: print(f"Error loading OTU file: {e}"); raise

# --- Load Metadata ---
print(f"\nLoading Metadata file: {metadata_file}...")
try:
    metadata = pd.read_csv(metadata_file, sep='\t', index_col=0, header=0)
    print(f"Successfully loaded Metadata file, shape: {metadata.shape}")
    if group_column not in metadata.columns: raise ValueError(f"ERROR: Metadata file missing the specified group column '{group_column}'!")
except FileNotFoundError: print(f"ERROR: Metadata file '{metadata_file}' not found in '{base_work_dir}'!"); raise
except Exception as e: print(f"Error loading Metadata file: {e}"); raise

# --- Align Samples ---
print("\nAligning samples between OTU table and Metadata...")
common_samples = otu_table.index.intersection(metadata.index)
print(f"Found {len(common_samples)} common samples.")
if len(common_samples) == 0: raise ValueError("ERROR: No common sample IDs found between OTU table and Metadata.")
otu_table_aligned = otu_table.loc[common_samples]
metadata_aligned = metadata.loc[common_samples]
print(f"Aligned OTU table shape: {otu_table_aligned.shape}")
print(f"Aligned Metadata shape: {metadata_aligned.shape}")

# --- Split Data by Group ---
print("\nSplitting data based on Metadata group...")
gdm_samples = metadata_aligned[metadata_aligned[group_column] == gdm_group_name].index
non_gdm_samples = metadata_aligned[metadata_aligned[group_column] == non_gdm_group_name].index
print(f"Found {len(gdm_samples)} GDM samples ({gdm_group_name})")
print(f"Found {len(non_gdm_samples)} NonGDM samples ({non_gdm_group_name})")
if len(gdm_samples) == 0 or len(non_gdm_samples) == 0: print("WARNING: One or both groups have zero samples. Network comparison may fail.")
otu_gdm = otu_table_aligned.loc[gdm_samples]
otu_non_gdm = otu_table_aligned.loc[non_gdm_samples]

def build_and_analyze_network(abundance_df, group_name,
                                min_prev_pct=10, min_mean_abund=0.00005,
                                pseudo_cnt=1e-9,
                                corr_th=0.3, fdr_th=0.05,
                                output_dir=".", output_prefix="network"):
    """
    Builds, analyzes, and returns a microbial co-occurrence network from an abundance dataframe.
    """
    print(f"\n--- Processing Group: {group_name} ---")
    if abundance_df.empty or abundance_df.shape[0] < 2:
        print(f"ERROR: Insufficient samples (<2) for group '{group_name}'. Skipping network construction.")
        return None, None
    print(f"Input samples: {abundance_df.shape[0]}, Original features: {abundance_df.shape[1]}")

    # --- 4.1 Data Filtering ---
    print("Step 4.1: Filtering features...")
    abundance_df = abundance_df.loc[:, (abundance_df != 0).any(axis=0)] # Remove all-zero features
    if abundance_df.shape[1] == 0: print("ERROR: No features remaining after removing all-zero columns."); return None, None

    prevalence = (abundance_df > 0).sum(axis=0) / abundance_df.shape[0] * 100
    relative_abundance = abundance_df.apply(lambda x: x / x.sum() if x.sum() > 0 else x, axis=1)
    mean_abundance = relative_abundance.mean(axis=0)
    keep_otu_prev = prevalence >= min_prev_pct
    keep_otu_abund = mean_abundance >= min_mean_abund
    keep_otu = keep_otu_prev & keep_otu_abund
    df_filtered = abundance_df.loc[:, keep_otu]
    print(f"Features remaining after filtering: {df_filtered.shape[1]}")
    if df_filtered.shape[1] < 2: print("ERROR: Fewer than 2 features remain after filtering. Cannot build network."); return None, None

    # --- 4.2 CLR Transformation ---
    print("Step 4.2: CLR Transformation...")
    df_pseudo = df_filtered + pseudo_cnt
    try:
        log_df_pseudo = np.log(df_pseudo); log_geo_means = log_df_pseudo.mean(axis=1); clr_data = log_df_pseudo.sub(log_geo_means, axis=0)
        print("CLR transformation complete.")
    except Exception as e: print(f"Error during CLR transformation: {e}"); return None, df_filtered

    # --- 4.3 Spearman Correlation ---
    print("Step 4.3: Calculating Spearman Correlation...")
    if clr_data.shape[1] < 2: print("ERROR: Fewer than 2 features after CLR."); return None, df_filtered
    try:
        rho_matrix, pval_matrix = spearmanr(clr_data)
        if np.isscalar(rho_matrix): print("ERROR: Only one feature, cannot calculate correlation matrix."); return None, df_filtered
        features = clr_data.columns; rho_df = pd.DataFrame(rho_matrix, index=features, columns=features); pval_df = pd.DataFrame(pval_matrix, index=features, columns=features)
        np.fill_diagonal(pval_df.values, np.nan); print("Correlation calculation complete.")
    except Exception as e: print(f"Error during Spearman correlation: {e}"); return None, df_filtered

    # --- 4.4 FDR Correction ---
    print("Step 4.4: Multiple testing correction (FDR)...")
    if not isinstance(pval_df, pd.DataFrame) or pval_df.empty: print("ERROR: Invalid P-value DataFrame."); return None, df_filtered
    upper_indices = np.triu_indices_from(pval_df.values, k=1); pval_flat = pval_df.values[upper_indices]
    valid_pval_indices = ~np.isnan(pval_flat)
    if np.sum(valid_pval_indices) == 0:
        print("WARNING: No valid P-values for correction."); p_adj_df = pval_df.copy(); np.fill_diagonal(p_adj_df.values, 1.0)
    else:
        try:
            reject, p_adj_flat_corrected, _, _ = multipletests(pval_flat[valid_pval_indices], method='fdr_bh')
            p_adj_values = np.ones_like(pval_df.values)
            p_adj_values[upper_indices[0][valid_pval_indices], upper_indices[1][valid_pval_indices]] = p_adj_flat_corrected
            lower_indices = (upper_indices[1][valid_pval_indices], upper_indices[0][valid_pval_indices])
            p_adj_values[lower_indices] = p_adj_flat_corrected; np.fill_diagonal(p_adj_values, 1.0)
            p_adj_df = pd.DataFrame(p_adj_values, index=features, columns=features); print("FDR correction complete.")
        except Exception as e: print(f"Error during FDR correction: {e}"); return None, df_filtered

    # --- 4.5 Network Construction ---
    print(f"Step 4.5: Building network based on thresholds (|rho| >= {corr_th}, FDR < {fdr_th})...")
    G = nx.Graph(); significant_edges = 0; positive_edges = 0; negative_edges = 0
    for node in features: G.add_node(node)
    for u, v in itertools.combinations(features, 2):
        if u not in rho_df.index or v not in rho_df.columns or u not in p_adj_df.index or v not in p_adj_df.columns: continue
        rho = rho_df.loc[u, v]; p_adj = p_adj_df.loc[u, v]
        if abs(rho) >= corr_th and not np.isnan(p_adj) and p_adj < fdr_th:
            sign = 'positive' if rho > 0 else 'negative'; G.add_edge(u, v, weight=rho, sign=sign, p_adj=p_adj)
            significant_edges += 1;
            if sign == 'positive': positive_edges += 1
            else: negative_edges += 1
    print(f"Network construction complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} significant edges.")
    if G.number_of_edges() == 0: print("WARNING: No significant edges found. The network is empty.")

    # --- 4.6 Network Properties ---
    if G.number_of_edges() > 0:
        density = nx.density(G); print(f"Network Density: {density:.4f}"); print(f"Positive Edges: {positive_edges}"); print(f"Negative Edges: {negative_edges}")
    else: print("Network is empty, skipping density calculation.")

    # --- 4.7 Save Results ---
    print("Step 4.7: Saving result files...")
    try:
        df_filtered.to_csv(os.path.join(output_dir, f"{output_prefix}_{group_name}_filtered_otu.csv"))
        rho_df.to_csv(os.path.join(output_dir, f"{output_prefix}_{group_name}_correlation_rho.csv"))
        p_adj_df.to_csv(os.path.join(output_dir, f"{output_prefix}_{group_name}_correlation_padj.csv"))
        if G.number_of_edges() > 0:
            edge_list = [{'Source': u, 'Target': v, 'Weight': data['weight'], 'Sign': data['sign'], 'FDR': data['p_adj']} for u, v, data in G.edges(data=True)]
            edge_df = pd.DataFrame(edge_list); edge_df.to_csv(os.path.join(output_dir, f"{output_prefix}_{group_name}_edgelist.csv"), index=False)
        else: print("Network is empty, no edgelist file saved.")
        nx.write_graphml(G, os.path.join(output_dir, f"{output_prefix}_{group_name}_network.graphml")); print(f"Network files saved to {output_dir}")
    except Exception as e: print(f"Error saving result files: {e}")

    return G, df_filtered

# --- Build NonGDM Network ---
if non_gdm_samples.empty: print("NonGGDM samples not found, skipping network construction."); G_non_gdm = None; otu_non_gdm_filtered = None
else: G_non_gdm, otu_non_gdm_filtered = build_and_analyze_network(otu_non_gdm, non_gdm_group_name, min_prev_pct=min_prevalence_percentage, min_mean_abund=min_mean_abundance, pseudo_cnt=pseudocount, corr_th=corr_threshold, fdr_th=fdr_threshold, output_dir=output_dir, output_prefix="NonGDM")

# --- Build GDM Network ---
if gdm_samples.empty: print("GDM samples not found, skipping network construction."); G_gdm = None; otu_gdm_filtered = None
else: G_gdm, otu_gdm_filtered = build_and_analyze_network(otu_gdm, gdm_group_name, min_prev_pct=min_prevalence_percentage, min_mean_abund=min_mean_abundance, pseudo_cnt=pseudocount, corr_th=corr_threshold, fdr_th=fdr_threshold, output_dir=output_dir, output_prefix="GDM")

print("\n--- Network analysis script finished ---")