#!/usr/bin/env python
# coding: utf-8

# #  Import Libraries 

# In[1]:


# Libraries for data manipulation and analysis
# --------------------------------------------
import numpy as np
import pandas as pd
from collections import Counter


# In[2]:


# Visualization libraries
# -----------------------
import matplotlib.pyplot as plt
import seaborn as sns 


# In[3]:


# Machine Learning Models
# -----------------------
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.model_selection import train_test_split 
from imblearn.over_sampling import SMOTE


# In[4]:


# Model Evaluation Metrics
# ------------------------
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report, 
    ConfusionMatrixDisplay, RocCurveDisplay
)


# In[5]:


# Random Seed
# -----------
SEED = 42


# Configure Pandas to display all rows and columns
# ------------------------------------------------
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)


# Suppress unnecessary warnings
# -----------------------------
import warnings
warnings.simplefilter("ignore")


# Import time module to measure detection time for each model
# ------------------------------------------------------------
import time


# # Load and Preview Data

# In[6]:


# Load the dataset into a Pandas DataFrame
# ----------------------------------------
smarthome_dataset = pd.read_csv('smarthome_data.csv')


# Display dataset summary
# -----------------------
smarthome_dataset.info()


# In[7]:


# Display the first five rows of the dataset
# ------------------------------------------
smarthome_dataset.head()


# # Clean Data

# In[8]:


# Check for null values in the dataset
# ------------------------------------
if smarthome_dataset.isna().values.any():
    print("Dataset contains null values:")
    print(smarthome_dataset.isna().sum().loc[lambda x: x > 0])  # Display only columns with null values
else:
    print("Dataset does not contain null values.")


# In[9]:


# Check for infinity values in numeric columns
# --------------------------------------------

# Extract only numeric columns
numeric_columns = smarthome_dataset.select_dtypes(include=np.number)

inf_counts = np.isinf(numeric_columns).sum()

if inf_counts.any():
    print("Dataset contains infinity values.")
    print(inf_counts[inf_counts > 0])  # Display only columns with infinity values
else:
    print("Dataset does not contain infinity values.")


# In[10]:


# Replace infinity values with NaN
# --------------------------------
smarthome_dataset.replace([np.inf, -np.inf], np.nan, inplace=True)


# In[11]:


# Drop rows with any NaN values 
# -----------------------------
smarthome_dataset.dropna(inplace=True)


# In[12]:


# Identify single-valued columns
# ------------------------------
single_valued_columns = [col for col in smarthome_dataset.columns if smarthome_dataset[col].nunique() == 1]

# Display the count and names of single-valued columns
print(f'There are {len(single_valued_columns)} single-valued columns:')
print(single_valued_columns)


# In[13]:


# Convert flag number columns from float to integer
# -------------------------------------------------
flag_columns = [
    'fin_flag_number', 'syn_flag_number', 'rst_flag_number',
    'psh_flag_number', 'ack_flag_number', 'ece_flag_number', 'cwr_flag_number'
]

smarthome_dataset[flag_columns] = smarthome_dataset[flag_columns].round().astype(int)


# In[14]:


# Consolidate categories
# ----------------------

# Show categories
print("Number of Categories: ", smarthome_dataset['Label'].nunique())
print(smarthome_dataset['Label'].unique())

# Define a label mapping dictionary to categorize different attack types
label_mapping_dict = {
    'BENIGN': 'Benign',
    
    # Brute-force attacks
    'DICTIONARYBRUTEFORCE': 'BruteForce',
    
    # Distributed Denial-of-Service (DDoS) Attacks
    'DDOS-ACK_FRAGMENTATION': 'DDoS',
    'DDOS-HTTP_FLOOD': 'DDoS',
    'DDOS-ICMP_FLOOD': 'DDoS',
    'DDOS-ICMP_FRAGMENTATION': 'DDoS',
    'DDOS-PSHACK_FLOOD': 'DDoS',
    'DDOS-RSTFINFLOOD': 'DDoS',
    'DDOS-SYN_FLOOD': 'DDoS',
    'DDOS-SLOWLORIS': 'DDoS',
    'DDOS-SYNONYMOUSIP_FLOOD': 'DDoS',
    'DDOS-TCP_FLOOD': 'DDoS',
    'DDOS-UDP_FLOOD': 'DDoS',
    'DDOS-UDP_FRAGMENTATION': 'DDoS',
    
    # Denial-of-Service (DoS) Attacks
    'DOS-HTTP_FLOOD': 'DoS',
    'DOS-SYN_FLOOD': 'DoS',
    'DOS-TCP_FLOOD': 'DoS',
    'DOS-UDP_FLOOD': 'DoS',

    # Mirai Attacks
    'MIRAI-GREETH_FLOOD': 'Mirai',
    'MIRAI-GREIP_FLOOD': 'Mirai',
    'MIRAI-UDPPLAIN': 'Mirai',

    # Reconnaissance attacks
    'RECON-HOSTDISCOVERY': 'Recon',
    'RECON-OSSCAN': 'Recon',
    'RECON-PINGSWEEP': 'Recon',
    'RECON-PORTSCAN': 'Recon',
    'VULNERABILITYSCAN': 'Recon',
    
    # Spoofing attacks
    'DNS_SPOOFING': 'Spoofing',
    'MITM-ARPSPOOFING': 'Spoofing',

    # Web-based attacks
    'BACKDOOR_MALWARE': 'Web-Based',
    'BROWSERHIJACKING': 'Web-Based',
    'COMMANDINJECTION': 'Web-Based',
    'SQLINJECTION': 'Web-Based',
    'UPLOADING_ATTACK': 'Web-Based',
    'XSS': 'Web-Based'
}

# Map labels to categories
smarthome_dataset['Category'] = smarthome_dataset['Label'].replace(label_mapping_dict)

print("\nNumber of Categories after Mapping:", smarthome_dataset['Category'].nunique())
print(smarthome_dataset['Category'].unique())
print("\nCategory Distribution:\n", smarthome_dataset['Category'].value_counts())


# In[15]:


# Identify duplicate rows
# -----------------------
duplicate_rows = smarthome_dataset[smarthome_dataset.duplicated()]
print(f"Number of duplicate rows (excluding first occurrences): {duplicate_rows.shape[0]}")


# In[16]:


# Remove duplicate rows while keeping the first occurrence
# --------------------------------------------------------
smarthome_dataset.drop_duplicates(inplace=True)

# Display the shape of the updated dataset
print(f"Dataset shape after removing duplicates: {smarthome_dataset.shape}\n")

# Count occurrences of each unique value in the category column
print(smarthome_dataset['Category'].value_counts())


# # Explore Data

# In[17]:


# View first five rows of cleaned dataset
# ---------------------------------------
smarthome_dataset.head()


# In[18]:


# Show concise summary of dataset
# -------------------------------
smarthome_dataset.info()


# In[19]:


# Display the count of unique values for each column in the dataset  
# -----------------------------------------------------------------
for column in smarthome_dataset.columns:  
    print(column, smarthome_dataset[column].nunique())  


# ## Visualize Data

# In[20]:


# Get a sorted list of unique categories
# --------------------------------------
categories = sorted(list(smarthome_dataset['Category'].unique()))
print(categories)


# In[21]:


# Count and display the number of unique protocol types across categories
# -----------------------------------------------------------------------
print(smarthome_dataset.groupby(['Category', 'Protocol Type']).size())

plt.figure(figsize=(11,5))
sns.countplot(
    data=smarthome_dataset, 
    x='Category', 
    order=categories, 
    hue='Protocol Type'
)
plt.yscale('log')
plt.show()


# In[22]:


# Display the average header length for each category
# ---------------------------------------------------
print(smarthome_dataset.groupby('Category')['Header_Length'].mean())

plt.figure(figsize=(10,5))
sns.barplot(
    data=smarthome_dataset, 
    x='Category', 
    y='Header_Length', 
    order=categories
)
plt.show()


# In[23]:


# Show the average flow rate for each category
# --------------------------------------------
mean_flow_rate_per_category = smarthome_dataset.groupby('Category')['Rate'].mean().sort_values(ascending=False)
print(mean_flow_rate_per_category)

plt.figure(figsize=(7,5))
sns.pointplot(
    data=smarthome_dataset, 
    x='Rate', 
    y='Category', 
    order=mean_flow_rate_per_category.index, 
    color='#BF0A30'
)
plt.grid(axis='y')
plt.xlabel('Flow Rate')
plt.show()


# In[24]:


# Display the average packet size of each category
# --------------------------------------
print(smarthome_dataset.groupby('Category')['Tot size'].mean())

plt.figure(figsize=(10,5))
sns.barplot(
    data=smarthome_dataset,
    x='Category',
    y='Tot size',
    order=categories
)
plt.ylabel('Total Size')
plt.show()


# In[25]:


# Show the distribution of Inter-Arrival Time for each category
# -------------------------------------------------------------
plt.figure(figsize=(9,5))
sns.boxplot(
    data=smarthome_dataset, 
    x='IAT', 
    y='Category', 
    showfliers=False  # Hide outliers for a cleaner plot
)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.xlabel('Inter-Arrival Time')
plt.show()


# In[26]:


# Show the distribution of 'ack_flag_number' within each category
# ---------------------------------------------------------------
ack_flag_count = smarthome_dataset.groupby('Category')['ack_flag_number'].value_counts().unstack()
ack_flag_count = ack_flag_count[ack_flag_count.columns[::-1]]  # Reverse Columns Order
print(ack_flag_count, '\n')

# Calculate the percentage of each unique 'ack_flag_number' per category
ack_flag = smarthome_dataset.groupby('Category')['ack_flag_number'].value_counts(normalize=True).unstack() * 100
ack_flag.columns = ['Not Set', 'Set']
ack_flag = ack_flag[ack_flag.columns[::-1]]

print(ack_flag, '\n')

# Plot the stacked bar chart
ack_flag.plot(
    kind='bar', 
    stacked=True, 
    edgecolor='black', 
    figsize=(10, 5.5)
)
plt.xticks(rotation=0)
plt.legend(title="ACK Flag", loc="upper right")  # Ensure the legend appears with a title
plt.ylabel('Percentage (%)')
plt.show()


# # Transform Labels

# ## For Binary Classification

# In[27]:


# Map 'Benign' to 0 and all other categories to 1
# -----------------------------------------------
smarthome_dataset['Binary Category Code'] = smarthome_dataset['Category'].apply(lambda x: 0 if x == 'Benign' else 1)

# Display the count of the two classes
print(smarthome_dataset['Binary Category Code'].value_counts(ascending=True))


# In[28]:


# Show binary classification categories
# -------------------------------------
binary_code_count = smarthome_dataset['Binary Category Code'].value_counts(ascending=True)
binary_code_count.index = ['Benign', 'Anomaly']
print(binary_code_count)

binary_code_count.plot(
    kind='bar', 
    color=['#004F98', '#E0115F']
)
plt.xticks(rotation=0)
plt.show()


#  ## For Multi-class Classification

# In[29]:


# Assign number to each category
# ------------------------------
category_to_index_mapping = {category: index for index, category in enumerate(categories)}

category_to_index_mapping


# In[30]:


# Map categories to unique numbers
# --------------------------------
smarthome_dataset['MultiClass Category Code'] = smarthome_dataset['Category'].map(category_to_index_mapping)

# Display the count of each unique numerical label in the dataset
print(smarthome_dataset['MultiClass Category Code'].value_counts())


# In[31]:


# Show Categories
# ---------------
print(smarthome_dataset['Category'].value_counts())

plt.figure(figsize=(10,6))
sns.countplot(
    data=smarthome_dataset, 
    x='Category', 
    order=categories, 
    palette='hsv'
)
plt.yscale('log')
plt.show()


# ### Separation of Features and Labels

# In[32]:


# Show column information
# -----------------------
smarthome_dataset.info()


# In[33]:


# Drop target columns to get only features
# ----------------------------------------
numeric_features = list(smarthome_dataset.columns.drop(['Label', 'Category',
                                                        'Binary Category Code',
                                                        'MultiClass Category Code']))

# Features
X = smarthome_dataset[numeric_features].values

# Target variable for binary classification
binary_y = smarthome_dataset['Binary Category Code'].values

# Target variable for multi-class classification
multiclass_y = smarthome_dataset['MultiClass Category Code'].values


# # Train and Evaluate Model

# In[34]:


# Instantiate models
# ------------------
models = [
    RandomForestClassifier(random_state=SEED),
    DecisionTreeClassifier(random_state=SEED),
    KNeighborsClassifier(),
    LogisticRegression(random_state=SEED),
    LinearSVC(random_state=SEED)
]


# ## Binary Classification

# In[35]:


# Split data into 70% training set and 30% test set
# -------------------------------------------------
binary_X_train, binary_X_test, binary_y_train, binary_y_test = train_test_split(X, binary_y,
                                                                                test_size=0.3,
                                                                                stratify=binary_y,
                                                                                random_state=SEED)


# In[36]:


# Apply SMOTE to balance the classes in the training set
# ------------------------------------------------------

# Check class distribution before SMOTE
print("Before SMOTE:", Counter(binary_y_train))

# Generate synthetic samples for the minority class
smote = SMOTE(random_state=SEED)
binary_X_train_resampled, binary_y_train_resampled = smote.fit_resample(binary_X_train, binary_y_train)

# Check class distribution after SMOTE
print("After SMOTE:", Counter(binary_y_train_resampled))

# View binary class distribution in the training set before and after SMOTE
# Create DataFrames for before and after SMOTE
before_df_binary = pd.DataFrame({'Class': binary_y_train})
after_df_binary = pd.DataFrame({'Class': binary_y_train_resampled})

# Plot vertically stacked countplots
plt.figure(figsize=(5, 8))

# Before SMOTE
plt.subplot(2, 1, 1)
sns.countplot(
    data=before_df_binary,
    x='Class', 
    palette=['#004F98', '#E0115F']
)
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], 
           labels=['Benign', 'Anomaly'])

# After SMOTE
plt.subplot(2, 1, 2)
sns.countplot(
    data=after_df_binary,
    x='Class',
    palette=['#004F98', '#E0115F']
)
plt.title('Class Distribution After SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1],
           labels=['Benign', 'Anomaly'])

plt.tight_layout()
plt.show()


# In[37]:


# Dictionary to store evaluation results
# --------------------------------------
binary_results = {
    "Model": [],
    "Accuracy": [],
    "Precision": [],
    "True Positive Rate": [],
    "True Negative Rate": [],
    "F1 Score": [],
    "ROC AUC Score": [],
    "Detection Time (s)": []
}


for model in models:
    model_name = model.__class__.__name__
    
    # Train the model
    model.fit(binary_X_train_resampled, binary_y_train_resampled) 
    
    # Measure detection time
    start_time = time.time()
    model_pred = model.predict(binary_X_test)               
    end_time = time.time()
    
    detection_time = round(end_time - start_time, 3)  # Time in seconds, rounded to 3 decimal places             
    
    # Compute confusion matrix
    cm = confusion_matrix(binary_y_test, model_pred)

    # Compute metrics (rounded to 3 decimal places)
    accuracy = round(accuracy_score(binary_y_test, model_pred), 3)
    precision = round(precision_score(binary_y_test, model_pred), 3)
    tpr = round(recall_score(binary_y_test, model_pred), 3)
    tnr = round(cm[0, 0] / cm[0].sum(), 3)
    f1score = round(f1_score(binary_y_test, model_pred), 3)
    roc_auc = round(roc_auc_score(binary_y_test, model_pred), 3)
    
    # Store results in the dictionary
    binary_results["Model"].append(model_name)
    binary_results["Accuracy"].append(accuracy)
    binary_results["Precision"].append(precision)
    binary_results["True Positive Rate"].append(tpr)
    binary_results["True Negative Rate"].append(tnr)
    binary_results["F1 Score"].append(f1score)
    binary_results["ROC AUC Score"].append(roc_auc)
    binary_results["Detection Time (s)"].append(detection_time)

    # Print results
    print(f"\nBinary Classification Evaluation Results for {model_name}: ")
    print("----------------------------------------------------------------------")
        
    metrics = ["Accuracy", "Precision", "True Positive Rate", "True Negative Rate", 
               "F1 Score", "ROC AUC Score", "Detection Time (s)"]
    values = [accuracy, precision, tpr, tnr, f1score, roc_auc, detection_time]
    for metric, value in zip(metrics, values):
        print(f"{metric}: {value}")

    # Print Confusion Matrix
    print("\nConfusion Matrix:")
    print(cm)
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=binary_code_count.index)
    cm_display.plot(cmap='plasma')
    plt.show()

    # Print Classification Report
    print("\nClassification Report:")
    print(classification_report(binary_y_test, model_pred, target_names=binary_code_count.index))

    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1], 'r--')  # Diagonal reference line
    RocCurveDisplay.from_predictions(binary_y_test, model_pred, name=model_name, ax=ax)
    plt.show()

    print("=====================================================================================")


# In[38]:


# Convert results dictionary to a DataFrame
# -----------------------------------------
binary_results_df = pd.DataFrame(binary_results)
print("\nFinal Summary of Model Performance:")
print(binary_results_df)


# In[39]:


# Show accuracy scores
# --------------------
plt.figure(figsize=(7,4))
sns.barplot(
    data=binary_results_df.sort_values(by='Accuracy', ascending=False),
    x='Accuracy', 
    y='Model', 
    palette='PuRd_r'
)
plt.xlabel("Accuracy Score")
plt.show()


# In[40]:


# Show True Positive Rates
# ------------------------
plt.figure(figsize=(7,3))
sns.barplot(
    data=binary_results_df.sort_values(by='True Positive Rate', ascending=False),
    x='True Positive Rate', 
    y='Model', 
    width=0.6, 
    palette='hsv_r'
)

plt.show()


# In[41]:


# Show True Negative Rates
# ------------------------
plt.figure(figsize=(7,3))
sns.barplot(
    data=binary_results_df.sort_values(by='True Negative Rate', ascending=False),
    x='True Negative Rate', 
    y='Model', 
    width=0.6, 
    palette='hsv_r'
)
plt.show()


# In[42]:


# Show all ROC_AUC curves in one chart
# ------------------------------------
ax = plt.gca()
ax.plot([0, 1], [0, 1], 'r--')

for model in models:
    model_name = model.__class__.__name__

    model.fit(binary_X_train_resampled, binary_y_train_resampled) 
    model_pred = model.predict(binary_X_test)  

    RocCurveDisplay.from_predictions(binary_y_test, model_pred, name=model_name, ax=ax)

plt.show()


# In[43]:


# Show Detection Times
# ---------------------
plt.figure(figsize=(7,4))
sns.barplot(
    data=binary_results_df.sort_values(by='Detection Time (s)'),
    x='Detection Time (s)', 
    y='Model', 
    width=0.6, 
    palette='OrRd'
)
plt.xscale('log')
plt.show()


# ## Multi-class Classification

# In[44]:


# Split data into 70% training set and 30% test set
# -------------------------------------------------
multiclass_X_train, multiclass_X_test, multiclass_y_train, multiclass_y_test = train_test_split(X, multiclass_y,
                                                                                                test_size=0.3,
                                                                                                stratify=multiclass_y,
                                                                                                random_state=SEED)


# In[45]:


# Apply SMOTE to balance the classes in the training set
# ------------------------------------------------------

# Check class distribution before applying SMOTE
print("Before SMOTE:", Counter(multiclass_y_train))

# Initialize SMOTE for oversampling the minority classes
smote = SMOTE(random_state=SEED)
multiclass_X_train_resampled, multiclass_y_train_resampled = smote.fit_resample(multiclass_X_train, multiclass_y_train)

# Check class distribution after applying SMOTE
print("After SMOTE:", Counter(multiclass_y_train_resampled))

# View multiclass class distribution in the training set before and after SMOTE
# Create DataFrames for before and after SMOTE
before_df_multiclass = pd.DataFrame({'Class': multiclass_y_train})
after_df_multiclass = pd.DataFrame({'Class': multiclass_y_train_resampled})

# Plot vertically stacked countplots
plt.figure(figsize=(9, 12))

# Before SMOTE
plt.subplot(2, 1, 1)
sns.countplot(
    data=before_df_multiclass,
    x='Class',
    palette='hsv'
)
plt.title('Class Distribution Before SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=list(category_to_index_mapping.values()),
           labels=list(category_to_index_mapping.keys()))

# After SMOTE
plt.subplot(2, 1, 2)
sns.countplot(
    data=after_df_multiclass,
    x='Class',
    palette='hsv'
)
plt.title('Class Distribution After SMOTE')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(ticks=list(category_to_index_mapping.values()),
           labels=list(category_to_index_mapping.keys()))

plt.tight_layout()
plt.show()


# In[46]:


# Dictionary to store evaluation results
# --------------------------------------
multiclass_results = {
    "Model": [],
    "Accuracy": [],
    "Detection Time (s)": []
}


for model in models:
    model_name = model.__class__.__name__
    
    # Train the model
    model.fit(multiclass_X_train_resampled, multiclass_y_train_resampled) 
    
    # Measure detection time
    start_time = time.time()
    model_pred = model.predict(multiclass_X_test)               
    end_time = time.time()
    
    detection_time = round(end_time - start_time, 3)  # Time in seconds, rounded to 3 decimal places             
    
    # Compute accuracy
    accuracy = round(accuracy_score(multiclass_y_test, model_pred), 3)

    # Store results
    multiclass_results["Model"].append(model_name)
    multiclass_results["Accuracy"].append(accuracy)
    multiclass_results["Detection Time (s)"].append(detection_time)
    
    # Print results
    print(f"\nMulti-class Classification Evaluation Results for {model_name}: ")
    print("---------------------------------------------------------------------------")
    
    print(f"Accuracy: {accuracy}")
    print(f"Detection Time (s): {detection_time}")

    # Print Confusion Matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(multiclass_y_test, model_pred)
    print(cm)
    
    fig, ax = plt.subplots(figsize=(13, 7))
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=categories)
    cm_display.plot(ax=ax, cmap='plasma')
    fig.tight_layout() 
    plt.show()

    # Print Classification Report
    print("\nClassification Report:")
    print(classification_report(multiclass_y_test, model_pred, target_names=categories))

    print("=====================================================================================")


# In[47]:


# Convert results dictionary to a DataFrame
# -----------------------------------------
multiclass_results_df = pd.DataFrame(multiclass_results)
print("\nFinal Summary of Model Performance:")
print(multiclass_results_df)


# In[48]:


# Show accuracy scores
# --------------------
plt.figure(figsize=(7,4))
sns.barplot(
    data=multiclass_results_df.sort_values(by='Accuracy', ascending=False),
    x='Accuracy', 
    y='Model', 
    palette='RdPu_r'
)
plt.xlabel("Accuracy Score")
plt.show()


# In[49]:


# Show Detection Times
# ---------------------
plt.figure(figsize=(7,4))
sns.barplot(
    data=multiclass_results_df.sort_values(by='Detection Time (s)'),
    x='Detection Time (s)', 
    y='Model', 
    width=0.6, 
    palette='OrRd'
)
plt.xscale('log')
plt.show()


# In[50]:


# Show accuracy metrics for binary and multi-class models
# -------------------------------------------------------
accuracy_metrics = {
    'Model': binary_results["Model"],
    'Binary Accuracy': binary_results["Accuracy"],
    'Multi-class Accuracy': multiclass_results["Accuracy"]
}

accuracy_df = pd.DataFrame(accuracy_metrics)
accuracy_df


# In[51]:


# Reshape the DataFrame from wide to long format
# ----------------------------------------------
accuracy_df_long = accuracy_df.melt(
    id_vars=["Model"], var_name="Accuracy Type", value_name="Accuracy Score"
)

accuracy_df_long


# In[52]:


# Compare accuracy scores for the two tasks
# -----------------------------------------
fig, ax = plt.subplots(figsize=(11, 5))

sns.barplot(
    data=accuracy_df_long, 
    x='Model', 
    y='Accuracy Score', 
    width=0.5, 
    hue='Accuracy Type', 
    palette=['#FFA52C', '#007FFF']
)

plt.show()

