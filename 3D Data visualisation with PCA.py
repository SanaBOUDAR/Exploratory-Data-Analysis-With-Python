import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Uploading Data
data = pd.read_excel('DATA.xlsx', engine='openpyxl')

# Separating categorical and numerical columns
colonnes_numeriques = ['MT_ACCORD_BoxCox', 'Revenu_estime_BoxCox','DUR_P_BoxCox','Age_BoxCox']  
colonnes_categorielles = ['m2_Wilaya', 'CATEGORIE', 't24_Profession', 'CODE', 't18_Genre', 't23_EtatCivil', 'I_CLASS']

# Separating categorical and numerical data
data_numerique = data[colonnes_numeriques]
data_categorique = data[colonnes_categorielles]

# Saving the targer column (class)
classes = data['I_CLASS']

# Standardisation of numerical columns
scaler_standard = StandardScaler()
data_numerique_standard = pd.DataFrame(scaler_standard.fit_transform(data_numerique), columns=colonnes_numeriques)

# Normalisation of numerical columns
scaler_normal = MinMaxScaler()
data_numerique_scaled = pd.DataFrame(scaler_normal.fit_transform(data_numerique_standard), columns=colonnes_numeriques)

# One-hot encoding of categorical values
data_categorique_encoded = pd.get_dummies(data[colonnes_categorielles])

# Reconstrcuting the dataset
data_preprocessed = pd.concat([data_numerique_scaled, data_categorique_encoded], axis=1)

# PCA
pca = PCA(n_components=3)
principal_components = pca.fit_transform(data_preprocessed)

# Dataframe for principal components
pca_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2', 'PC3'])

# Adding the target column
pca_df = pd.concat([pca_df, classes.reset_index(drop=True)], axis=1)

# Mapping class names (if needed)
class_mapping = {0: 'Classe 0', 1: 'Classe 1', 2: 'Classe 2', 3: 'Classe 3'}
pca_df['Class_Name'] = pca_df['I_CLASS'].map(class_mapping)

# Visualisation
fig = plt.figure(figsize=(6, 4))  # Size of the plot
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(pca_df['PC1'], pca_df['PC2'], pca_df['PC3'], c=pca_df['I_CLASS'], cmap='viridis', alpha=0.7)

# Add a legend
legend1 = ax.legend(*scatter.legend_elements(), title="I_CLASS", bbox_to_anchor=(1.05, 1), loc='upper left')
ax.add_artist(legend1)

# Définir les étiquettes
ax.set_title('Class visualisation with PCA')
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_zlabel('PC3')

plt.tight_layout()
plt.show()
