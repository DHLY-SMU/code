import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import traceback

# --- Visualize Z-score Results ---
print("\nPlotting Z-score comparison...")

if 'comparison_df' in locals() and not comparison_df.empty and 'output_dir' in locals():
    try:
        # Filter for valid, numeric Z-scores, excluding NaN and non-numeric entries like 'StdDev~0'
        plot_df = comparison_df[pd.to_numeric(comparison_df['Z_Score'], errors='coerce').notna()].copy()
        plot_df['Z_Score'] = pd.to_numeric(plot_df['Z_Score'])

        if not plot_df.empty:
            # Sort by Z-score (descending) for better visualization
            plot_df = plot_df.sort_values(by='Z_Score', ascending=False)

            plt.figure(figsize=(10, 6))
            # Create horizontal bars, colored by Z-score sign
            bars = plt.barh(plot_df.index, plot_df['Z_Score'],
                            color=np.where(plot_df['Z_Score'] > 0, 'skyblue', 'lightcoral'))

            plt.xlabel("Z-score (Guild 1 vs. Random Networks)")
            plt.ylabel("Topology Metric")
            plt.title("Guild 1 Topology Metrics: Z-score Comparison") # Simplified title

            # Add Z-score value labels to the bars
            for bar in bars:
                plt.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                         f'{bar.get_width():.2f}',
                         va='center', ha='left' if bar.get_width() >= 0 else 'right', # Handle zero correctly
                         color='black', fontsize=8) # Adjusted font size slightly

            plt.axvline(0, color='grey', linewidth=0.8) # Add a line at Z=0
            plt.grid(axis='x', linestyle='--', alpha=0.6)
            plt.tight_layout() # Adjust layout

            # Save the plot
            zscore_plot_filename = os.path.join(output_dir, 'Guild1_Zscore_Comparison_Plot.png')
            plt.savefig(zscore_plot_filename, dpi=300, bbox_inches='tight')
            print(f"Z-score comparison plot saved: {zscore_plot_filename}")
            plt.close()

        else:
            print("No valid data available for Z-score visualization after filtering.")

    except KeyError as e:
        print(f"Error: Missing expected column in 'comparison_df': {e}. Cannot plot Z-scores.")
    except Exception as e:
        print(f"Error generating or saving Z-score plot: {e}")
        traceback.print_exc() # Print detailed error information
    finally:
        plt.close() # Ensure plot is closed even if errors occur

elif 'comparison_df' not in locals() or comparison_df.empty:
    print("Skipping Z-score plot: 'comparison_df' not found or is empty. Ensure Analysis 2 ran successfully.")
elif 'output_dir' not in locals():
     print("Skipping Z-score plot: 'output_dir' variable not defined.")