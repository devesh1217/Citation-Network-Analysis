import os
import matplotlib.pyplot as plt
from datetime import datetime

def save_plot(plt, name, base_dir='plots'):
    """
    Save plot in a single organized directory.
    
    Args:
        plt: matplotlib.pyplot instance
        name: name of the plot
        base_dir: base directory for saving plots
    """
    # Create base directory if it doesn't exist
    os.makedirs(base_dir, exist_ok=True)
    
    # Save the plot
    filename = os.path.join(base_dir, f"{name}.png")
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")
