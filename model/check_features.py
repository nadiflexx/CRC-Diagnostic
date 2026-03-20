import pandas as pd, os
p = os.path.join(os.path.dirname(__file__), '..', 'Data', 'processed', 'dataset_clinico_tumoral.csv')
print('CSV:', p)
df = pd.read_csv(p)
cols = [c for c in df.columns if c not in ['Patient_ID','Diagnosis']]
perfect_sep = []
matches_label = []
for c in cols:
    a = df[df['Diagnosis']==0][c]
    b = df[df['Diagnosis']==1][c]
    if df[c].equals(df['Diagnosis']):
        matches_label.append(c)
    if a.max() < b.min() or b.max() < a.min():
        perfect_sep.append((c, float(a.min()), float(a.max()), float(b.min()), float(b.max())))

print('matches_label:', matches_label)
print('perfectly separated features (<=10 shown):')
for item in perfect_sep[:10]:
    print(item)
print('total features:', len(cols))
print('total perfectly separated:', len(perfect_sep))
