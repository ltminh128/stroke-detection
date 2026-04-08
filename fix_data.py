import pandas as pd
df = pd.read_csv('landmarks.csv')
cols = [c for c in df.columns if any(x in c for x in ['wrist','arm','shoulder'])]
df.drop(columns=cols).to_csv('landmarks_face_only.csv', index=False)
print('Done! Saved landmarks_face_only.csv')