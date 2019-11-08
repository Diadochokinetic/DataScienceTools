import pandas as pd
from dstools.preprocessing.NumScaler import NumScaler

#x1 should be scaled and x2 not
df = pd.DataFrame({'x1':[1,2,3,4],'x2':['a','b','c','d']})
df_transformed = NumScaler().fit_transform(df)
print(df_transformed)