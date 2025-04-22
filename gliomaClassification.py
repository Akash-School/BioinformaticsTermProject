
from sklearn.metrics import mean_squared_error, accuracy_score,confusion_matrix, classification_report
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier 
from sklearn.linear_model import LinearRegression, LogisticRegression
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
import numpy as np

glioma_dataset = fetch_ucirepo(id=759)
#data preprocessing
X = glioma_dataset.data.features
X = X.drop(['Age_at_diagnosis', 'Race'], axis=1) 
y = glioma_dataset.data.targets 

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()  
 
print(categorical_cols)

print(glioma_dataset.metadata) 

print(glioma_dataset.variables) 

X_train, X_test, Y_train, Y_test = train_test_split(X,y, test_size = 0.3, random_state=42)

#model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, Y_train)

y_pred = rf_model.predict(X_test)
print(y_pred)

print(f"Accuracy: {accuracy_score(Y_test, y_pred):.2f}")



perm_importance = permutation_importance(rf_model, X_test, Y_test, n_repeats=10)
sorted_features = perm_importance.importances_mean.argsort()
top_features = 10
gene_features = [col for col in X.columns if col not in ['Race', 'Age_at_diagnosis']]

#confusion matrix plot - shows the true positives, true negatives, false postives, false negatives
cm = confusion_matrix(Y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                             display_labels=['LGG', 'HGG'])
disp.plot(cmap='Blues')
plt.title('Glioma Classification Performance')
plt.show()


#permutation importance plot - showing which genes affect the glioma
plt.figure(figsize=(10,6))
plt.barh(np.array(gene_features)[sorted_features][-top_features:],
        perm_importance.importances_mean[sorted_features][-top_features:])
plt.xlabel("Permutation Importance")
plt.title("Top Genetic Drivers of Glioma Classification")
plt.tight_layout()
plt.show()