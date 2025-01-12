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
    import matplotlib.pyplot as plt
    plt.clf()
    for key in y:
        plt.plot(x, y[key], label=key)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.show()