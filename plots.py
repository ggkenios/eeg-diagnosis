import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from common import CLASS_LIST
from common.patient_validation import dic

array = [
    [dic["t0_p0"], dic["t0_p1"], dic["t0_p2"]],
    [dic["t1_p0"], dic["t1_p1"], dic["t1_p2"]],
    [dic["t2_p0"], dic["t2_p1"], dic["t2_p2"]],
]

df_cm = pd.DataFrame(
    array,
    index=CLASS_LIST,
    columns=CLASS_LIST,
)

sn.set(font_scale=1.4)
sn.heatmap(df_cm, annot=True)
plt.show()
