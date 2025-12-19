# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import numpy as np


# %%
try:
    df = pd.read_csv('D:\programmes\Major Project\dataset_traffic_accident_prediction1.csv')
    print(f"Initial shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
except FileNotFoundError:
    print("Error: dataset_traffic_accident_prediction1.csv not found. Please ensure the file path is correct.")
    exit()

# %%
# --- 1. Initial Data Inspection & Renaming (Optional but good practice) ---
# Let's clean up column names for easier access (e.g., remove spaces)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace('__', '_')
print("\nCleaned column names:")
print(df.columns.tolist())

# Identify columns with missing values
print("\nMissing values before preprocessing:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# %%
# --- 2. Handling Missing Values ---

# Drop rows where the 'Accident' (target) is missing. Can't predict what isn't there!
initial_rows = df.shape[0]
df.dropna(subset=['Accident'], inplace=True)
rows_after_target_drop = df.shape[0]
if initial_rows - rows_after_target_drop > 0:
    print(f"\nDropped {initial_rows - rows_after_target_drop} rows due to missing 'Accident' values.")

# Define numerical and categorical features for imputation
numerical_cols = ['Traffic_Density', 'Speed_Limit', 'Number_of_Vehicles', 'Driver_Age', 'Driver_Experience','Driver_Alcohol']
categorical_cols = [
    'Weather', 'Road_Type', 'Time_of_Day', 'Accident_Severity',
    'Road_Condition', 'Vehicle_Type', 'Road_Light_Condition'
]

# Impute numerical features with the median
print("\nImputing numerical features with median...")
for col in numerical_cols:
    if df[col].isnull().any():
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"  - Filled missing values in '{col}' with median: {median_val}")

# Impute categorical features with the mode
print("\nImputing categorical features with mode...")
for col in categorical_cols:
    if df[col].isnull().any():
        mode_val = df[col].mode()[0] # .mode() can return multiple if ties, take the first
        df[col].fillna(mode_val, inplace=True)
        print(f"  - Filled missing values in '{col}' with mode: '{mode_val}'")

# Verify no more missing values (except potentially for 'Accident_Severity' if it was a target and not a feature)
print("\nMissing values after imputation:")
print(df.isnull().sum()[df.isnull().sum() > 0])
if df.isnull().sum().sum() == 0:
    print("All missing values handled! Data's looking cleaner already, Boss.")
else:
    print("Still some stragglers, Boss. Double-check the imputation logic.")

# %%

# --- 3. Encoding Categorical Features ---

print("\nEncoding categorical features...")

# Ordinal Encoding for Accident_Severity (assuming an order: Low < Moderate < High)
# Define the order explicitly to ensure correct mapping
severity_order = ['Low', 'Moderate', 'High']
# Check if all unique values in Accident_Severity are in our defined order
if not set(df['Accident_Severity'].unique()).issubset(set(severity_order)):
    print("Warning: 'Accident_Severity' contains values not in defined order. Adjust 'severity_order' if needed.")
    print(f"Unique values found: {df['Accident_Severity'].unique()}")
    # If there are new values, you might need to decide how to handle them.
    # For now, we'll proceed with the defined order, but this is a heads-up.

ordinal_encoder = OrdinalEncoder(categories=[severity_order], handle_unknown='use_encoded_value', unknown_value=-1)
df['Accident_Severity_Encoded'] = ordinal_encoder.fit_transform(df[['Accident_Severity']])
print(f"  - 'Accident_Severity' encoded to numerical: {df['Accident_Severity_Encoded'].unique()}")

# One-Hot Encoding for other nominal categorical features
# Exclude 'Accident_Severity' as it's now encoded
nominal_cols_for_ohe = [col for col in categorical_cols if col != 'Accident_Severity']
    
# Initialize OneHotEncoder
# handle_unknown='ignore' prevents errors if new categories appear in test set
# sparse_output=False returns a dense array
one_hot_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

# Fit and transform the nominal columns
encoded_features = one_hot_encoder.fit_transform(df[nominal_cols_for_ohe])

# Create a DataFrame from the encoded features with proper column names
encoded_df = pd.DataFrame(encoded_features, columns=one_hot_encoder.get_feature_names_out(nominal_cols_for_ohe), index=df.index)
print(f"  - Nominal features one-hot encoded. New columns: {encoded_df.columns.tolist()[:5]}...")

# Drop original nominal columns and concatenate the new encoded ones
df = pd.concat([df.drop(columns=nominal_cols_for_ohe), encoded_df], axis=1)
print(f"  - Original nominal columns dropped. Current shape: {df.shape}")

# Drop the original 'Accident_Severity' column as we have the encoded version
df.drop(columns=['Accident_Severity'], inplace=True)
print("  - Original 'Accident_Severity' column dropped.")

# %%
# --- 4. Feature Scaling (Standardization) ---

print("\nScaling numerical features using StandardScaler...")
# Identify all numerical columns that need scaling (excluding the target and already encoded ordinal)
# We'll use the original numerical_cols list, as they are now clean.
# 'Accident_Severity_Encoded' is also numerical but its scale is small and meaningful,
# so we might choose not to scale it, or scale it along with others.
# For simplicity, let's scale all numerical features including the newly encoded ordinal.
features_to_scale = numerical_cols + ['Accident_Severity_Encoded']

scaler = StandardScaler()
df[features_to_scale] = scaler.fit_transform(df[features_to_scale])
print("  - Numerical features standardized. They're all playing nice now.")

# %%
# --- 5. Feature Engineering (A little extra spice!) ---
print("\nAdding some engineered features...")

# Example: Age-Experience Ratio (handle division by zero if Driver_Age can be 0)
# Assuming Driver_Age is always > 0 after imputation, but good to be safe.
df['Driver_Age_Experience_Ratio'] = df['Driver_Experience'] / df['Driver_Age']
# Handle potential inf values if Driver_Age was 0 and Driver_Experience was not
df['Driver_Age_Experience_Ratio'].replace([np.inf, -np.inf], np.nan, inplace=True)
df['Driver_Age_Experience_Ratio'].fillna(df['Driver_Age_Experience_Ratio'].median(), inplace=True) # Impute any new NaNs

# Example: Is_Peak_Hour (simple example, adjust logic based on actual peak times)
# Assuming 'Morning', 'Afternoon', 'Evening', 'Night' are the Time_of_Day categories
# This will work with the one-hot encoded columns
if 'Time_of_Day_Morning' in df.columns and 'Time_of_Day_Evening' in df.columns:
    df['Is_Peak_Hour'] = ((df['Time_of_Day_Morning'] == 1) | (df['Time_of_Day_Evening'] == 1)).astype(int)
    print("  - 'Is_Peak_Hour' feature created.")
else:
    print("  - Could not create 'Is_Peak_Hour' feature (Time_of_Day columns not found as expected).")

# %%
# --- 6. Final Data Inspection ---
print("\nFinal dataset head after polishing:")
print(df.head())
print(f"\nFinal dataset shape: {df.shape}")
print("\nFinal missing values check:")
print(df.isnull().sum().sum()) # Should be 0

# Separate features (X) and target (y)
X = df.drop('Accident', axis=1)
y = df['Accident']

# --- 7. Data Splitting (Stratified for 'Accident' target) ---
print("\nSplitting data into training, validation, and test sets (stratified)...")
# First split: training + validation vs. test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)

# Second split: training vs. validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=(0.15 / 0.85), random_state=42, stratify=y_train_val
)

print(f"  - Training set shape: {X_train.shape}, Target shape: {y_train.shape}")
print(f"  - Validation set shape: {X_val.shape}, Target shape: {y_val.shape}")
print(f"  - Test set shape: {X_test.shape}, Target shape: {y_test.shape}")

print("\nDistribution of 'Accident' in splits:")
print("Train:", y_train.value_counts(normalize=True))
print("Validation:", y_val.value_counts(normalize=True))
print("Test:", y_test.value_counts(normalize=True))

print("\nDataset polishing complete, Boss! Ready for model training. Go get 'em!")

# %%
df

# %%
# Assuming df is your polished DataFrame
output_filename = 'polished_dataset.csv'
df.to_csv(output_filename, index=False)
print(f"Successfully saved the polished dataset to {output_filename}")


# %%
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# Load your polished dataset (assuming it's saved as 'polished_dataset.csv')
df = pd.read_csv('polished_dataset.csv') 

# --- Upsampling to 1200 records ---
TARGET_RECORDS = 1200
current_records = len(df)
additional_records_needed = TARGET_RECORDS - current_records

print(f"Current records: {current_records}")
print(f"Additional records needed: {additional_records_needed}")

# Separate features and target
X = df.drop('Accident', axis=1)
y = df['Accident']

# Apply SMOTE only if we need more records
if additional_records_needed > 0:
    # Calculate sampling_strategy to get exactly 1200 records
    # SMOTE needs the minority class count, so we'll calculate based on current class distribution
    minority_class = y.value_counts().idxmin()
    minority_count = y.value_counts().min()
    majority_count = y.value_counts().max()
    
    # We want to keep the majority class as is and upsample minority
    sampling_strategy = {
        minority_class: minority_count + additional_records_needed
    }
    
    smote = SMOTE(
        sampling_strategy=sampling_strategy,
        random_state=42,
        k_neighbors=min(5, minority_count - 1)  # Ensure k_neighbors <= minority samples
    )
    
    X_resampled, y_resampled = smote.fit_resample(X, y)
    
    # Combine back into DataFrame
    df_upsampled = pd.concat([
        pd.DataFrame(X_resampled, columns=X.columns),
        pd.Series(y_resampled, name='Accident')
    ], axis=1)
    
    print(f"\nAfter SMOTE upsampling: {len(df_upsampled)} records")
    print("Class distribution after upsampling:")
    print(df_upsampled['Accident'].value_counts())
else:
    print("Dataset already meets target size")
    df_upsampled = df.copy()

# --- Save to CSV ---
output_filename = 'upsampled_dataset_1200_records.csv'
df_upsampled.to_csv(output_filename, index=False)
print(f"\nSuccessfully saved upsampled dataset to {output_filename}")



