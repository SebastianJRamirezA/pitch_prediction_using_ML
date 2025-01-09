import pandas as pd
import numpy as np

class Sequencer:
    def __init__(self,
                 data: pd.core.frame.DataFrame,
                 max_length: int,
                 n_features: int,
                 n_pitch_types: int) -> None:
        self.data = data
        self.max_length = max_length
        self.n_features = n_features
        self.n_pitch_types = n_pitch_types

    def _make_mappings(self, group):
        data_arr = group.values
        # 1-indexing the feature vectors lets us reserve 0 for padding
        dict_keys = [i for i in range(1, len(group) + 1)]
        features = data_arr[:, 1: -self.n_pitch_types]
        labels = data_arr[:, -self.n_pitch_types:]
        index_to_pitch = dict(zip(dict_keys, features))
        index_to_pitch[0] = np.zeros(features.shape[1], dtype='int64')
        index_to_label = dict(zip(dict_keys, labels))
        return index_to_pitch, index_to_label

    def _ascending_subsequences(self, increasing_list: list):
        """
        Break a list of increasing numbers into all possible increasing subsequences, including one-numbered subsequences.
        For a list of len == n, n(n+1)/2 sublists will be generated.
        Args
            increasing_list: A list of inceasing numbers. Each number is the index of a feature vector.
        Returns
            subsequences: A list of the n(n+1)/2 sublists.
        """
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
            if sublist_length < length:  # If too short
                sublist.extend([0] * (length - sublist_length))
                processed_sublists.append(sublist)
            else:
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
        group = group.sort_values('pitch_number', ascending=True)
        index_to_pitch, index_to_label = self._make_mappings(group)
        subsequences = self._ascending_subsequences(
            list(index_to_label.keys()))
        # add this now to prevent messing up the key length
        index_to_label[0] = np.zeros(self.n_pitch_types, dtype='int64')
        processed_sublists = self._build_sequences(subsequences)
        all_features = np.array([[index_to_pitch[i] for i in sublist]
                                for sublist in processed_sublists])
        all_labels = np.array(
            [index_to_label[max(sublist)] for sublist in processed_sublists]
        )
        return all_features, all_labels

    def make_sequences(self):
        data = self.data.astype(np.float64)

        test_split_ratio = 0.8
        val_split_ratio = 0.9

        train_val_size = int(len(data) * test_split_ratio)
        train_val_data, test_data = data[:train_val_size], data[train_val_size:]
        train_size = int(len(train_val_data) * val_split_ratio)

        train_data, val_data = train_val_data[:
                                              train_size], train_val_data[train_size:]
        train_sequences = train_data.groupby(
            'plate_app_id').apply(self._populate_vectors)
        X_train = np.concatenate([result[0] for result in train_sequences])
        y_train = np.concatenate([result[1] for result in train_sequences])

        val_sequences = val_data.groupby(
            'plate_app_id').apply(self._populate_vectors)
        X_val = np.concatenate([result[0] for result in val_sequences])
        y_val = np.concatenate([result[1] for result in val_sequences])

        test_sequences = test_data.groupby(
            'plate_app_id').apply(self._populate_vectors)
        X_test = np.concatenate([result[0] for result in test_sequences])
        y_test = np.concatenate([result[1] for result in test_sequences])

        return X_train, X_val, X_test, y_train, y_val, y_test
