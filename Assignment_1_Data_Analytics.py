import numpy as np
import pandas as pd
import plotly.express as px
from scipy.spatial.distance import mahalanobis
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
n_patients = 200


data = pd.DataFrame({
    'Patient_ID': np.arange(1, n_patients + 1),
    'Pain': np.random.randint(0, 10, n_patients),
    'Urgency': np.random.randint(0, 10, n_patients),
    'Frequency': np.random.randint(0, 10, n_patients),
    'Treated': np.random.choice([0, 1], size=n_patients, p=[0.5, 0.5]),
    'Treatment_Time': np.random.randint(1, 12, n_patients)  
})


scaler = StandardScaler()
data[['Pain', 'Urgency', 'Frequency']] = scaler.fit_transform(data[['Pain', 'Urgency', 'Frequency']])


def compute_mahalanobis_matrix(df):
    cov_matrix = np.cov(df[['Pain', 'Urgency', 'Frequency']].T)
    inv_cov_matrix = np.linalg.inv(cov_matrix)
    distance_matrix = np.zeros((len(df), len(df)))
    
    for i, row1 in df.iterrows():
        for j, row2 in df.iterrows():
            distance_matrix[i, j] = mahalanobis(row1[['Pain', 'Urgency', 'Frequency']], 
                                                row2[['Pain', 'Urgency', 'Frequency']], 
                                                inv_cov_matrix)
    return distance_matrix


treated = data[data['Treated'] == 1].reset_index(drop=True)
untreated = data[data['Treated'] == 0].reset_index(drop=True)


distance_matrix = compute_mahalanobis_matrix(pd.concat([treated, untreated], ignore_index=True))


row_ind, col_ind = linear_sum_assignment(distance_matrix[:len(treated), len(treated):])


matched_pairs = pd.DataFrame({
    'Treated_ID': treated.iloc[row_ind]['Patient_ID'].values,
    'Untreated_ID': untreated.iloc[col_ind]['Patient_ID'].values
})

print("Matched Pairs:")
print(matched_pairs.head())


mean_pain = data.groupby('Treated')['Pain'].mean().reset_index()


wide_mean_pain = mean_pain.pivot(columns='Treated', values='Pain').reset_index(drop=True)


fig = px.bar(wide_mean_pain, title="Average Pain Score in Treated vs Untreated Patients",
             labels={0: "Untreated", 1: "Treated"},
             barmode='group')

fig.show()

