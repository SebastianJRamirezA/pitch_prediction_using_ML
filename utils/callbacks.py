import tensorflow as tf
import numpy as np

class FreezeOutputCallback(tf.keras.callbacks.Callback):
    def __init__(self, patience=5):
        super(FreezeOutputCallback, self).__init__()
        self.patience = patience
        self.best_val_accuracies = {}
        self.wait = {}
        self.frozen_outputs = set()

    def on_train_begin(self, logs=None):
        self.best_val_accuracies = {
            'pitch': -np.Inf,
            'vertical': -np.Inf,
            'horizontal': -np.Inf
        }
        self.wait = {
            'pitch': 0,
            'vertical': 0,
            'horizontal': 0
        }
        self.frozen_outputs = set()

    def on_epoch_end(self, epoch, logs=None):
        for output in self.best_val_accuracies.keys():
            if output in self.frozen_outputs:
                continue
            val_acc = logs.get(f'val_{output}_output_accuracy')
            if val_acc is not None:
                if val_acc > self.best_val_accuracies[output]:
                    self.best_val_accuracies[output] = val_acc
                    self.wait[output] = 0
                else:
                    self.wait[output] += 1
                    if self.wait[output] >= self.patience:
                        self.freeze_output(output)
                        self.frozen_outputs.add(output)
                        if len(self.frozen_outputs) == len(self.best_val_accuracies):
                            self.model.stop_training = True
                            print(f"\nAll outputs frozen. Stopping training at epoch {epoch + 1}.")
                            break

    def freeze_output(self, output_name):
        for layer in self.model.layers:
            if layer.name.startswith(output_name):
                layer.trainable = False
        print(f"\nFreezing output: {output_name}")