from common import CLASS_LIST

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = [
        [19, 0, 0],
        [1, 24, 0],
        [0, 0, 6],
        ]
df_cm = pd.DataFrame(array,
                     index=CLASS_LIST,
                     columns=CLASS_LIST)
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