"""
Run:  python confusion_matrix_from_folder.py
"""

import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ⚙️ 1. basic params -----------------------------------------------------------
MODEL_PATH   = "/Users/martin/Desktop/LIAT/circuit_detector/models/v12.keras"       # your saved model
TEST_DIR     = "/Users/martin/Desktop/LIAT/circuit_detector/dataset/circuit_dataset"                 # folder shown above
IMG_SIZE     = (128, 128)             # change if your model expects e.g. (299, 299)
BATCH_SIZE   = 32                     # tweak for your GPU / CPU
AUTOTUNE     = tf.data.AUTOTUNE
# -----------------------------------------------------------------------------

print("loading model …")
model = tf.keras.models.load_model(MODEL_PATH)

# 2. build the tf.data pipeline ------------------------------------------------
print("building test dataset …")
test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    image_size=IMG_SIZE,
    shuffle=False,             # DO NOT shuffle – we need original order to line up labels
    batch_size=BATCH_SIZE,
    color_mode = "grayscale" # 👈 forces shape (batch, h, w, 1)
)

class_names = test_ds.class_names          # e.g. ['classA', 'classB', 'classC']
n_classes   = len(class_names)

AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.prefetch(AUTOTUNE)

# 4. collect labels & predictions ---------------------------------------------
print("running inference …")
y_true = np.concatenate([y.numpy() for _, y in test_ds])
y_pred_probs = model.predict(test_ds, verbose=0)
if n_classes == 2 and y_pred_probs.shape[-1] == 1:      # binary sigmoid head
    y_pred = (y_pred_probs.ravel() > 0.5).astype(int)
else:                                                   # softmax head
    y_pred = np.argmax(y_pred_probs, axis=1)

# 5. confusion matrix ----------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)
print("confusion matrix:\n", cm)

# 6. visual --------------------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.heatmap(cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()
