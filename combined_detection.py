import numpy as np

def find_elbow(values, threshold=0.10):
    """ Elbow method function to asses the optimal number of components required for the GMM.
        An unsupervised approach which detects where the current change falls below xx% of the
        previuous change

    Args:
        values: Measured values, either AIC or BIC.
        threshold: Percentage threshold for the new change. Defaults to 0.10.

    Returns:
        n_optimal_components: optimal number of components
    """
    changes = np.diff(values)  # Compute first derivative (changes between steps)
    for i in range(1, len(changes)):
        # Check if the new change is less than threshold * previous change
        if abs(changes[i]) < threshold * abs(changes[i - 1]):
            return i + 1  # Return the index (components count starts from 1)
    return len(values)



def filter_occurence_indices(occurence_indices, min_space=100):
    """This function is used to filter out occurrences that are too-close to each other, imposing 
    a constraint of minimum distance between the end of an occurrence to the beginning of a new one.
    It is required for the algorithm in Q.4, where pulses might be all over the place.

    Args:
        occurence_indices_pulse (_type_): _description_
        min_space (int, optional): _description_. Defaults to 100.

    Returns:
        _type_: _description_
    """
    filtered = []
    filtered.append(occurence_indices[0])
    for i in range(1, len(occurence_indices)):
        idx_cur = occurence_indices[i]
        idx_prev = occurence_indices[i-1]

        if idx_cur - min_space > idx_prev:
            filtered.append(idx_cur)

    return filtered


def get_connected_components(indices):
    """A function that given a list of occurrence indices, calculates the connected components-
    these are the ranges of *consecutive* occurrence indices.

    Args:
        indices: A list of occurrence indices

    Returns:
        num_components: Number of connected componenets
        components: A list containing all lists of connected componenets
    """
    if not list(indices):  # Handle empty list
        return 0, []

    # Sort the indices to ensure they're in order
    indices = sorted(indices)

    # Initialize variables
    components = []  # List to store (start, end) of each component
    start = indices[0]  # Start of the current component

    for i in range(1, len(indices)):
        # Check if current index is not consecutive
        if indices[i] != indices[i - 1] + 1:
            # Store the previous component (start, end)
            components.append((start, indices[i - 1]))
            # Start a new component
            start = indices[i]

    # Append the last component
    components.append((start, indices[-1]))

    # Number of components
    num_components = len(components)

    return num_components, components




def merge_connected_components(components1, components2):
    """Unused method"""

    # Combine the two lists
    all_components = components1 + components2
    # Sort components by their start index
    all_components.sort(key=lambda x: x[0])
    
    # Merged list of components
    merged = []
    
    # Initialize the first component
    current_start, current_end = all_components[0]
    
    # Iterate through all components
    for start, end in all_components[1:]:
        if start <= current_end + 1:  # Check for overlap or touching
            current_end = max(current_end, end)  # Extend the current component
        else:
            merged.append((current_start, current_end))  # Add the previous component
            current_start, current_end = start, end  # Start a new component
    
    # Add the last component
    merged.append((current_start, current_end))
    
    return merged
