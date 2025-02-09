import matplotlib.pyplot as plt

# Function to smooth the curve
# Code adapted from Chollet (2021)
def smooth_curve(points, factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points: # an empty list is 'False'
            previous = smoothed_points[-1] # the last appended point
            smoothed_points.append(previous * factor + point * (1-factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

# Function to compare training and validation metrics across epochs
def plot_epochs(x, y, title, x_label, y_label):
    plt.clf()
    for key in y:
        plt.plot(x, y[key], label=key)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    return plt.gcf()

def plot_ensemble_history(history):
    # Smooth the accuracy curves
    smooth_pitch_accuracy = smooth_curve(history.history['pitch_output_accuracy'][5:])
    smooth_val_pitch_accuracy = smooth_curve(history.history['val_pitch_output_accuracy'][5:])
    smooth_vertical_accuracy = smooth_curve(history.history['vertical_output_accuracy'][5:])
    smooth_val_vertical_accuracy = smooth_curve(history.history['val_vertical_output_accuracy'][5:])
    smooth_horizontal_accuracy = smooth_curve(history.history['horizontal_output_accuracy'][5:])
    smooth_val_horizontal_accuracy = smooth_curve(history.history['val_horizontal_output_accuracy'][5:])

    # Create a grid of plots
    fig, axs = plt.subplots(3, 1, figsize=(10, 15))

    # Plot pitch accuracy
    axs[0].plot(range(1, len(smooth_pitch_accuracy) + 1), smooth_pitch_accuracy, label='Training')
    axs[0].plot(range(1, len(smooth_val_pitch_accuracy) + 1), smooth_val_pitch_accuracy, label='Validation')
    axs[0].set_title('Pitch type accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Plot vertical accuracy
    axs[1].plot(range(1, len(smooth_vertical_accuracy) + 1), smooth_vertical_accuracy, label='Training')
    axs[1].plot(range(1, len(smooth_val_vertical_accuracy) + 1), smooth_val_vertical_accuracy, label='Validation')
    axs[1].set_title('Vertical Accuracy')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()

    # Plot horizontal accuracy
    axs[2].plot(range(1, len(smooth_horizontal_accuracy) + 1), smooth_horizontal_accuracy, label='Training')
    axs[2].plot(range(1, len(smooth_val_horizontal_accuracy) + 1), smooth_val_horizontal_accuracy, label='Validation')
    axs[2].set_title('Horizontal Accuracy')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Accuracy')
    axs[2].legend()

    plt.tight_layout()
    plt.show()