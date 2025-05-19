import torch
import torch.nn as nn
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np # For numpy array manipulation if needed by drawing func

# --- 1. Настройка PYTHONPATH для импорта model_architecture ---
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
APP_DIR = os.path.join(PROJECT_ROOT, 'app')
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

try:
    from learn.torch_text.model_architecture import AdvancedTextClassifier
    print("Successfully imported AdvancedTextClassifier from app.learn.torch_text.model_architecture")
except ImportError as e:
    print(f"Error: Could not import AdvancedTextClassifier: {e}")
    print("Current sys.path:", sys.path)
    print("Please ensure the path to 'app' directory is correctly added to sys.path,")
    print("and that 'app/learn/torch_text/model_architecture.py' exists and is a valid Python module.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during import: {e}")
    exit()

# --- 2. Загрузка Гиперпараметров ---
params_path = os.path.join(PROJECT_ROOT, "model", "torch_text", "final_best_params.json")
if not os.path.exists(params_path):
    print(f"Error: Hyperparameter file not found at {params_path}")
    # Create a dummy best_params if file not found, so script can run for viz demo
    best_params = {
        'dropout': 0.5,
        'activation': 'relu',
        'use_batch_norm': False,
        # 'num_layers' and 'hidden_size_x' will be set by TARGET_HIDDEN_SIZES
    }
    print("Warning: final_best_params.json not found. Using default non-structural parameters.")
else:
    with open(params_path, 'r') as f:
        best_params = json.load(f)

# --- 3. Подготовка Конфигурации Модели ---
# Override structural parameters to match the target image for visualization.
# Non-structural parameters (dropout, activation, batch_norm) will be from best_params.
TARGET_INPUT_SIZE = 16
TARGET_HIDDEN_SIZES = [12, 10] # Two hidden layers with 12 and 10 neurons
TARGET_NUM_CLASSES = 1         # Single output neuron

# These will be used to instantiate the model
actual_input_size = TARGET_INPUT_SIZE
actual_hidden_layers = TARGET_HIDDEN_SIZES
actual_num_classes = TARGET_NUM_CLASSES

# Get other params from the loaded file, or use defaults if file was missing
dropout_config = best_params.get('dropout', 0.5)
activation_config = best_params.get('activation', 'relu')
use_batch_norm_config = best_params.get('use_batch_norm', False)

print(f"\nInstantiating model for visualization with:")
print(f"  input_size: {actual_input_size}")
print(f"  hidden_layers: {actual_hidden_layers}")
print(f"  dropout: {dropout_config}")
print(f"  activation: {activation_config}")
print(f"  use_batch_norm: {use_batch_norm_config}")
print(f"  num_classes: {actual_num_classes}")

# --- 4. Создание Экземпляра Модели ---
# This model will have the 16 -> 12 -> 10 -> 1 architecture
model_to_visualize = AdvancedTextClassifier(
    input_size=actual_input_size,
    hidden_layers=actual_hidden_layers,
    num_classes=actual_num_classes,
    dropout=dropout_config,
    activation=activation_config,
    use_batch_norm=use_batch_norm_config
)
model_to_visualize.eval()


# --- 5. Генерация Визуализации с помощью Matplotlib ---
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend, good for scripts
import matplotlib.pyplot as plt
import numpy as np # For numpy array manipulation if needed by drawing func

def draw_neural_net(ax, left, right, bottom, top, layer_sizes, layer_labels=None, node_radius_scale=1.0, line_width=0.5):
    '''
    Draw a neural network cartoon using matplotlib.
    :param ax: matplotlib.axes.Axes instance
    :param left: float, x-coordinate of the leftmost nodes (relative to ax 0-1 scale)
    :param right: float, x-coordinate of the rightmost nodes (relative to ax 0-1 scale)
    :param bottom: float, y-coordinate of the bottommost nodes (relative to ax 0-1 scale)
    :param top: float, y-coordinate of the topmost nodes (relative to ax 0-1 scale)
    :param layer_sizes: list of int, list containing the number of nodes in each layer
    :param layer_labels: list of str, list containing labels for each layer
    :param node_radius_scale: float, scaling factor for node radius
    :param line_width: float, width of the connection lines
    '''
    n_layers = len(layer_sizes)
    # Handle cases with no layers or single layer gracefully for spacing
    if n_layers == 0:
        return
    max_nodes_in_layer = float(max(layer_sizes) if layer_sizes else 1)
    if max_nodes_in_layer == 0: # Avoid division by zero if a layer has 0 nodes (should not happen with valid layer_sizes)
        max_nodes_in_layer = 1

    v_spacing_total_height = (top - bottom)
    v_spacing = v_spacing_total_height / max_nodes_in_layer if max_nodes_in_layer > 0 else v_spacing_total_height

    h_spacing = (right - left) / float(n_layers - 1) if n_layers > 1 else 0

    base_radius = (v_spacing / 2.5) * node_radius_scale
    if max_nodes_in_layer == 1:
         base_radius = (v_spacing_total_height / 5.0) * node_radius_scale # Use total height for better scaling of single nodes
    
    # Cap the node radius to prevent it from being excessively large or small
    # The radius is in data coordinates (0-1 range of the axis)
    min_abs_radius = 0.002 # Absolute minimum radius in data coords
    max_abs_radius_prop = 0.03 # Max radius as proportion of total height
    node_radius = max(min_abs_radius, min(base_radius, max_abs_radius_prop * v_spacing_total_height))
    if layer_sizes == [1]: # Special case for a single node network
        node_radius = 0.05 * node_radius_scale


    node_positions = []
    for n, layer_size in enumerate(layer_sizes):
        if layer_size == 0: # Skip layers with zero nodes
            node_positions.append([])
            continue
            
        current_layer_positions = []
        layer_y_center = (top + bottom) / 2.0
        
        # Calculate the y-positions for nodes in the current layer
        if layer_size == 1:
            ys = [layer_y_center]
        else:
            # Total height occupied by nodes in this layer if spaced by v_spacing
            # This v_spacing is based on max_nodes, so layers with fewer nodes will appear more spread out.
            # To make them compact and centered:
            effective_v_spacing_this_layer = v_spacing_total_height / layer_size if layer_size > 1 else v_spacing_total_height
            if effective_v_spacing_this_layer * (layer_size -1) > v_spacing_total_height * 0.95 : # If too spread, use v_spacing from max_nodes
                 effective_v_spacing_this_layer = v_spacing

            total_layer_node_span = (layer_size - 1) * effective_v_spacing_this_layer
            ys = [layer_y_center + (total_layer_node_span / 2.0) - m * effective_v_spacing_this_layer for m in range(layer_size)]


        x = left + n * h_spacing
        if n_layers == 1: # Center single layer
            x = (left + right) / 2.0

        for y_pos in ys:
            circle = plt.Circle((x, y_pos), node_radius, color='white', ec='black', zorder=4, lw=0.75) # slightly thinner edge
            ax.add_artist(circle)
            current_layer_positions.append((x, y_pos))
        node_positions.append(current_layer_positions)

    if n_layers > 1:
        for n in range(n_layers - 1):
            layer_nodes_a = node_positions[n]
            layer_nodes_b = node_positions[n+1]
            if not layer_nodes_a or not layer_nodes_b: # Skip if a layer was empty
                continue
            for x1, y1 in layer_nodes_a:
                for x2, y2 in layer_nodes_b:
                    line = plt.Line2D([x1, x2], [y1, y2], c='grey', lw=line_width, zorder=1, alpha=0.6) # Added alpha
                    ax.add_artist(line)

    if layer_labels:
        label_y_offset = 0.05 * (ax.get_ylim()[1] - ax.get_ylim()[0]) # Relative to axes height
        label_y_pos = ax.get_ylim()[0] + draw_bottom - label_y_offset # Place below the 'draw_bottom' line by an offset

        for n, label_text in enumerate(layer_labels):
            x = left + n * h_spacing
            if n_layers == 1:
                x = (left + right) / 2.0
            ax.text(x, label_y_pos, label_text, ha='center', va='top', fontsize=8, linespacing=0.9) # Reduced fontsize slightly

# Prepare data for drawing
layer_sizes_for_plot = [actual_input_size] + actual_hidden_layers + [actual_num_classes]

labels_for_plot = []
labels_for_plot.append(r"Input Layer $\in \mathbb{R}^{" + str(actual_input_size) + "}$")
for i, size in enumerate(actual_hidden_layers):
    labels_for_plot.append(r"Hidden Layer $\in \mathbb{R}^{" + str(size) + "}$")
labels_for_plot.append(r"Output Layer $\in \mathbb{R}^{" + str(actual_num_classes) + "}$")


# Create plot
fig_width_inches = 8
fig_height_inches = 8
fig = plt.figure(figsize=(fig_width_inches, fig_height_inches))
ax = fig.add_subplot(1, 1, 1) # Use add_subplot for clarity
ax.set_axis_off() # Turn off axis lines and labels

# IMPORTANT: Define the drawing area relative to the axes (which will be 0-1)
# These are the boundaries *within* the (0,1) x (0,1) axes where drawing will occur.
draw_left = 0.05
draw_right = 0.95
draw_bottom = 0.15 # Space for labels below this
draw_top = 0.95    # Network uses up to this y-coordinate

# Set axes limits to be [0,1] so our draw_left/right/bottom/top make sense as fractions
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)

custom_node_radius_scale = 0.7 # Fine-tune scale
custom_line_width = 0.35   # Fine-tune line width

draw_neural_net(ax, draw_left, draw_right, draw_bottom, draw_top,
                layer_sizes_for_plot, labels_for_plot,
                node_radius_scale=custom_node_radius_scale,
                line_width=custom_line_width)

# Ensure the output directory exists
output_dir = os.path.join(PROJECT_ROOT, "model", "torch_text")
os.makedirs(output_dir, exist_ok=True)
output_image_path = os.path.join(output_dir, "model_architecture.png")

try:
    # Save the figure
    plt.savefig(output_image_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    print(f"\nSuccess! Model architecture visualization saved to:")
    print(f"- PNG: {output_image_path}")
    if os.path.exists(output_image_path):
        print("PNG file verified.")
    else:
        print("Warning: PNG file was not created at expected location.")
except Exception as e:
    print(f"\nFailed to save Matplotlib graph: {e}")
finally:
    plt.close(fig) # Close the figure to free memory

# --- 6. (Optional) Print Textual Model Structure ---
print("\nTextual model structure (for the visualized architecture):")
print(model_to_visualize)