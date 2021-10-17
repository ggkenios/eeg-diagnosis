from patient_validation import dic
from common import CLASS_LIST

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt


array = [
        [dic["t0_p0"], dic["t0_p1"], dic["t0_p2"]],
        [dic["t1_p0"], dic["t1_p1"], dic["t1_p2"]],
        [dic["t2_p0"], dic["t2_p1"], dic["t2_p2"]],
        ]

df_cm = pd.DataFrame(
    array,
    index=CLASS_LIST,
    columns=CLASS_LIST
)

plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)

'''
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.7, 1])
plt.legend(loc='best')

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc='best')
'''