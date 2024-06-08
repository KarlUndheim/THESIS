import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

# Confusion matrices from results

BEST = """ [[86.4  2.9  8.   0.7  1.6  0.3  0.   0.2  0. ]
 [ 9.3 70.9 14.1  0.4  3.8  0.   1.3  0.1  0. ]
 [ 6.4  3.6 83.4  0.7  4.9  0.4  0.3  0.3  0. ]
 [ 3.2  1.1  2.  84.7  2.2  6.3  0.1  0.4  0. ]
 [ 7.3  8.1 17.3  1.  63.2  1.4  1.1  0.5  0.1]
 [ 7.5  0.   6.2 20.8  9.5 46.5  0.8  8.8  0. ]
 [ 6.5  0.   5.5  9.   9.  16.5 43.  10.5  0. ]
 [ 8.2  1.   7.2  7.5  8.   0.8  0.5 65.3  1.5]
 [ 0.  16.7 16.   6.7 48.   4.   1.3  7.3  0. ]] """

GEORAD = """ [[82.   2.6 10.9  0.8  2.7  0.6  0.3  0.   0. ]
 [ 9.7 67.8 17.1  0.1  3.9  0.   0.9  0.5  0. ]
 [ 8.4  5.1 79.6  0.7  5.2  0.2  0.3  0.5  0.1]
 [ 3.7  1.4  1.9 85.   2.9  5.1  0.   0.   0. ]
 [10.9  8.6 15.2  2.1 58.8  1.1  2.2  0.9  0.2]
 [ 9.2  0.   5.3 18.8  7.2 46.5  1.5 11.5  0. ]
 [ 8.   3.   5.  12.   7.  18.  37.5  9.5  0. ]
 [12.5  0.5  7.8  6.8  9.5  1.8  2.3 58.   1. ]
 [ 0.7 19.3  8.7  6.7 44.7  6.7  8.7  1.3  3.3]] """

GEORADVOX = """ [[84.6  2.6  9.2  0.8  1.6  0.8  0.3  0.2  0. ]
 [10.9 69.1 14.6  0.1  3.3  0.   1.4  0.6  0. ]
 [ 6.9  5.3 80.9  0.7  5.   0.1  0.3  0.7  0.1]
 [ 4.   1.   2.6 84.9  1.4  5.7  0.2  0.2  0. ]
 [ 7.2  9.6 15.6  2.3 60.9  1.2  2.   1.1  0.1]
 [ 7.8  0.   5.5 18.8  8.8 48.3  0.5 10.5  0. ]
 [12.   4.   5.  10.5  6.  15.5 40.   7.   0. ]
 [15.5  0.8  4.8  7.   7.5  1.5  1.7 59.7  1.5]
 [ 6.   9.3 16.   6.  45.3  7.3  6.7  2.   1.3]] """

GEORADBIN = """ [[83.7  3.3  9.7  0.7  2.1  0.1  0.2  0.2  0. ]
 [ 9.7 69.5 16.1  0.2  2.3  0.3  1.5  0.3  0.1]
 [ 8.1  2.7 82.3  0.7  4.9  0.4  0.5  0.1  0.2]
 [ 3.3  1.2  2.1 85.3  1.7  5.6  0.   0.5  0.3]
 [ 9.9  7.4 17.4  1.1 60.   1.3  1.9  0.1  0.9]
 [ 7.8  0.   5.3 18.8 10.8 46.5  2.   9.   0. ]
 [ 5.5  0.   4.5  9.5 10.  16.  43.  11.5  0. ]
 [ 9.8  0.5  7.5  7.5  9.8  0.8  0.8 61.3  2.2]
 [ 0.  20.7 12.   6.7 43.3  5.3  1.3  5.3  5.3]] """

display_labels=['PINE', 'SPRUCE', 'BIRCH', 'MAPLE', 'ASPEN', 'ROWAN', 'OAK', 'LIME', 'ALDER']
class_sizes = np.array([300, 150, 300, 100, 100, 40, 20, 40, 15])


# Convert from strings above to matrix
def string_to_matrix(matrix_str):
    lines = matrix_str.strip().split('\n')
    matrix = [line.strip(' []').split() for line in lines]
    return np.array(matrix, dtype=float)

# Calculate all metrics in case they should be needed
def calculate_metrics(conf_matrix, class_sizes):

    # Convert percentage matrix to count matrix
    conf_matrix_counts = (conf_matrix / 100) * class_sizes[:, None]

    users_accuracy = np.diag(conf_matrix_counts) / np.sum(conf_matrix_counts, axis=0)
    overall_accuracy = np.sum(np.diag(conf_matrix_counts)) / np.sum(conf_matrix_counts)
    
    total_sum = np.sum(conf_matrix_counts)
    pe = np.sum(np.sum(conf_matrix_counts, axis=0) * np.sum(conf_matrix_counts, axis=1)) / total_sum**2
    kappa = (overall_accuracy - pe) / (1 - pe)
    
    return users_accuracy, kappa, overall_accuracy



conf_matrix = string_to_matrix(GEORAD)
users_accuracy, kappa, overall_accuracy = calculate_metrics(conf_matrix, class_sizes)

# Print calculated metrics
print('')
print("User's Accuracy:", [f'{accuracy * 100:.1f}' for accuracy in users_accuracy])
print("")
print("Kappa Coefficient:", kappa)
print("Overall Accuracy:", overall_accuracy)

# plotting confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix,
                              display_labels=['PINE', 'SPRUCE', 'BIRCH', 'MAPLE', 'ASPEN', 'ROWAN', 'OAK', 'LIME', 'ALDER'])

fig, ax = plt.subplots()
disp.plot(cmap=plt.cm.YlOrRd, ax=ax, values_format='.1f') 
ax.set_title('GEO + RAD + BIN + VOX')

# Adjust label size
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(ax.get_yticklabels(), fontsize=8)

plt.show()
