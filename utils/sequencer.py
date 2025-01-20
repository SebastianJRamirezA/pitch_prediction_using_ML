import pandas as pd
import numpy as np

class Sequencer:
    def __init__(self,
                 data: pd.core.frame.DataFrame,
                 max_length: int,
                 n_features: int,
                 n_pitch_types: int,
                 n_vertical_locs: int,
                 n_horizontal_locs: int) -> None:
        self.data = data
        self.max_length = max_length
        self.n_features = n_features
        self.n_pitch_types = n_pitch_types
        self.n_vertical_locs = n_vertical_locs
        self.n_horizontal_locs = n_horizontal_locs

    def _make_mappings(self, group):
        # Convert group to numpy array
        data_arr = group.values
        # Create dictionary keys for 1-indexed feature vectors
        dict_keys = [i for i in range(1, len(group) + 1)]
        # Extract features and labels from the data array
        features = data_arr[:, 1: -self.n_pitch_types - self.n_vertical_locs - self.n_horizontal_locs]
        pitch_labels = data_arr[:, -self.n_pitch_types - self.n_vertical_locs - self.n_horizontal_locs: -self.n_vertical_locs - self.n_horizontal_locs]
        vertical_labels = data_arr[:, -self.n_vertical_locs - self.n_horizontal_locs: -self.n_horizontal_locs]
        horizontal_labels = data_arr[:, -self.n_horizontal_locs:]
        # Create mappings from indices to features and labels
        index_to_pitch = dict(zip(dict_keys, features))
        index_to_pitch[0] = np.zeros(features.shape[1], dtype='int64')
        index_to_pitch_label = dict(zip(dict_keys, pitch_labels))
        index_to_vertical_label = dict(zip(dict_keys, vertical_labels))
        index_to_horizontal_label = dict(zip(dict_keys, horizontal_labels))
        return index_to_pitch, index_to_pitch_label, index_to_vertical_label, index_to_horizontal_label

    def _ascending_subsequences(self, increasing_list: list):
        # Generate all ascending subsequences from the list
        subsequences = []
        n = len(increasing_list)
        for i in range(n):
            for j in range(i+1, n+1):
                subsequences.append(increasing_list[i:j])
        return subsequences

    def _build_sequences(self, subsequences):
        """
        Convert nested list of subsequences into a (n, m) matrix where m is the desired length (number of timesteps) allowed per sequence and n is determined dynamically by that restriction.
        Subsequences shorter than the desired length are padded with zeroes; longer subsequences are broken up and padded accordingly.
        Args
            subsequences: output of ascending_subsequences; a list of all possible subsequences of the indices of our feature vectors.
        Returns
            processed_sublists: (n, m) matrix representing the sequences we have computed for our pitch feature vectors.
        """
        length = self.max_length
        processed_sublists = []
        for sublist in subsequences:
            sublist_length = len(sublist)
            if sublist_length < length:  # If too short, pad with zeros
                sublist.extend([0] * (length - sublist_length))
                processed_sublists.append(sublist)
            else:
                # If too long, split into multiple sublists and pad with zeros
                num_sublists = (sublist_length + length - 1) // length
                pad_length = num_sublists * length - sublist_length
                sublist = np.pad(sublist, (0, pad_length),
                                 mode='constant', constant_values=0)
                processed_sublists.extend(np.split(sublist, num_sublists))
        return np.array(processed_sublists)

    def _populate_vectors(self, group):
        """
        After building the skeleton of our sequences using scalars for legibility and simplicity, use the feature vector mappings to populate the full 3D feature and label matrices.
        This function operates on one 'group' at a time, where each group is a unique atbat from a pitcher's history.
        This function is mapped onto our 2D DataFrame of pitches in a pandas groupby.
        Args
            group: A group resulting from a pandas groupby operation. Each group is a unique plate appearance.
        Returns
            all_features: (s, t, f) numpy array where s = the number of samples, t = the number of time steps per sample, and f = the number of features of every sample
        """
        # Sort group by pitch number
        group = group.sort_values('pitch_number', ascending=True)
        # Create mappings from indices to features and labels
        index_to_pitch, index_to_pitch_label, index_to_vertical_label, index_to_horizontal_label = self._make_mappings(group)
        # Generate all ascending subsequences of indices
        subsequences = self._ascending_subsequences(
            list(index_to_pitch_label.keys()))
        # Add zero vectors for padding
        index_to_pitch_label[0] = np.zeros(self.n_pitch_types, dtype='int64')
        index_to_vertical_label[0] = np.zeros(self.n_vertical_locs, dtype='int64')
        index_to_horizontal_label[0] = np.zeros(self.n_horizontal_locs, dtype='int64')
        # Build sequences from subsequences
        processed_sublists = self._build_sequences(subsequences)
        # Create feature and label arrays for each sequence
        all_features = np.array([[index_to_pitch[i] for i in sublist]
                                for sublist in processed_sublists])
        all_pitch_labels = np.array(
            [index_to_pitch_label[max(sublist)] for sublist in processed_sublists]
        )
        all_vertical_labels = np.array(
            [index_to_vertical_label[max(sublist)] for sublist in processed_sublists]
        )
        all_horizontal_labels = np.array(
            [index_to_horizontal_label[max(sublist)] for sublist in processed_sublists]
        )
        return all_features, all_pitch_labels, all_vertical_labels, all_horizontal_labels

    def make_sequences(self):
        # Convert data to float64
        data = self.data.astype(np.float64)
        # Group data by plate_app_id and apply _populate_vectors to each group
        sequences = data.groupby('plate_app_id').apply(self._populate_vectors)
        # Concatenate results from all groups
        X = np.concatenate([result[0] for result in sequences])
        y_pitch = np.concatenate([result[1] for result in sequences])
        y_vertical = np.concatenate([result[2] for result in sequences])
        y_horizontal = np.concatenate([result[3] for result in sequences])

        return X, y_pitch, y_vertical, y_horizontal
