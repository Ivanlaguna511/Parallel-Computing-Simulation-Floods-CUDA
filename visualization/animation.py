import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

"""
Rainwater Flooding Visualization Script
Parallel Computing Course (3rd Year)
University of Valladolid
"""

def load_simulation_data(filename: str):
    """
    Parses the simulation output file.
    Format:
    Line 1: rows columns
    Line 2: number_of_matrices
    Following lines: Flattened float values for all matrices
    """
    try:
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Read matrix dimensions from the first line
        rows, cols = map(int, lines[0].split())
        
        # Read the number of recorded frames
        num_matrices = int(lines[1].strip())
        
        # Parse all numerical data into a flattened array
        raw_data = [list(map(float, l.split())) for l in lines[2:] if l.strip()]
        data_array = np.array(raw_data).flatten()

        # Validate data integrity
        expected_size = num_matrices * rows * cols
        if len(data_array) != expected_size:
            print(f"Error: Data mismatch. Expected {expected_size} values, found {len(data_array)}.")
            sys.exit(1)

        # Reshape data into a list of 2D matrices
        frames = [data_array[i * rows * cols : (i + 1) * rows * cols].reshape(rows, cols) 
                  for i in range(num_matrices)]
        
        print(f"Successfully loaded {num_matrices} frames ({rows}x{cols})")

        # The first frame is the ground terrain
        terrain = frames[0]
        # The remaining frames represent water evolution
        water_frames = frames[1:]
        
        return rows, cols, terrain, water_frames

    except FileNotFoundError:
        print(f"Error: File '{filename}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

def animate_simulation(rows: int, cols: int, terrain, water_frames):
    """
    Creates an interactive animation of the water flowing over the terrain.
    """
    # Adjusted figsize to (10, 10) for standard monitors
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal')

    # Draw the ground terrain (YlOrBr colormap for soil/mountains)
    # Extent is set to SCENARIO_SIZE (30km) as defined in the C code
    ax.imshow(terrain, cmap="YlOrBr_r", interpolation="nearest", extent=[0, 30, 0, 30])

    print("Click on the window or press any key to start the animation...")
    plt.waitforbuttonpress()

    # Define the water colormap (Blues)
    cmap_water = plt.cm.Blues

    # Replace 0 values with NaN so that the terrain is visible where there is no water
    processed_water = [np.where(m <= 1e-6, np.nan, m) for m in water_frames]

    # Calculate global max for color consistency
    vmax_water = np.nanmax(processed_water) if len(processed_water) > 0 else 1.0

    # Initialize the water image overlay with transparency (alpha=0.7)
    img_water = ax.imshow(processed_water[0], cmap=cmap_water, interpolation='nearest', 
                          extent=[0, 30, 0, 30], vmin=0, vmax=vmax_water, alpha=0.7)

    # UI Elements
    ax.set_title("Rainwater Flooding Simulation")
    cbar = plt.colorbar(img_water, ax=ax, shrink=0.8)
    cbar.set_label("Water Level (m)")

    def update(frame):
        """Update function for the animation."""
        img_water.set_array(processed_water[frame])
        ax.set_title(f"Simulation Time Step: {frame + 1}")
        return [img_water]

    # Create the animation object
    # interval: milliseconds between frames
    ani = animation.FuncAnimation(fig, update, frames=len(processed_water), 
                                   interval=150, blit=True, repeat=True)

    # Optional: Save as video (requires ffmpeg)
    # ani.save("flood_animation.mp4", writer="ffmpeg", fps=10)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python animation.py <simulation_output.txt>")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Load and process data
    r, c, terrain_data, frames_data = load_simulation_data(input_file)

    # Launch visualizer
    animate_simulation(r, c, terrain_data, frames_data)