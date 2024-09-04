import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
from imblearn.under_sampling import RandomUnderSampler

# Step 1: Load the dataset
file_path =r'C:\Users\Admin\Desktop\CreditCard Detection\Credit_Card_Applications.csv'
df = pd.read_csv(file_path)

# Step 2: Basic Data Exploration and Visualization
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# Original Class Distribution
sns.countplot(x='Class', data=df, ax=axs[0, 0])
axs[0, 0].set_title('Class Distribution Before Under-Sampling')

# Step 3: Data Preprocessing
# Features and Labels
X = df.drop(columns=['Class'])  # Features
y = df['Class']  # Target variable

# Handling imbalanced data using under-sampling
rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X, y)

# Resampled Class Distribution
sns.countplot(x=y_resampled, ax=axs[0, 1])
axs[0, 1].set_title('Class Distribution After Under-Sampling')

# Scaling the features
scaler = StandardScaler()
X_resampled = scaler.fit_transform(X_resampled)

# Step 4: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

# Step 5: Build and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Make predictions and evaluate the model
y_pred = model.predict(X_test)

# Confusion Matrix Visualization
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, ax=axs[1, 0])
axs[1, 0].set_title('Confusion Matrix')
axs[1, 0].set_xlabel('Predicted')
axs[1, 0].set_ylabel('Actual')

# ROC Curve Visualization
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

axs[1, 1].plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
axs[1, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axs[1, 1].set_xlim([0.0, 1.0])
axs[1, 1].set_ylim([0.0, 1.05])
axs[1, 1].set_xlabel('False Positive Rate')
axs[1, 1].set_ylabel('True Positive Rate')
axs[1, 1].set_title('Receiver Operating Characteristic (ROC) Curve')
axs[1, 1].legend(loc="lower right")

plt.tight_layout()  # Adjust the layout so everything fits without overlapping
plt.show()
