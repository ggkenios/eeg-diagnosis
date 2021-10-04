from common import CLASS_LIST

import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt
array = [
        [19, 0, 0],
        [1, 24, 0],
        [1, 1, 4],
        ]
df_cm = pd.DataFrame(array,
                     index=CLASS_LIST,
                     columns=CLASS_LIST)
plt.figure(figsize=(10, 7))
sn.heatmap(df_cm, annot=True)
