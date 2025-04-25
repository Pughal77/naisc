import pandas as pd
import sys

# --- Installation Check ---
try:
    import pyarrow  # Or fastparquet if you prefer
except ImportError:
    print("The 'pyarrow' library is required but not installed.")
    print("Please install it using: pip install pyarrow")
    # Alternatively, you might need: pip install pandas[parquet]
    # or pip install fastparquet
    sys.exit(1)

# --- Configuration ---
METADATA_FILE_PATH = '../dataset/metadata.parquet'  # Corrected path

# --- Load Data ---
try:
    print(f"Loading metadata from: {METADATA_FILE_PATH}")
    df_metadata = pd.read_parquet(METADATA_FILE_PATH)
    print("Metadata loaded successfully.")
except FileNotFoundError:
    print(f"Error: Metadata file not found at {METADATA_FILE_PATH}")
    print("Please ensure the file path is correct.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred while loading the parquet file: {e}")
    sys.exit(1)


# --- Initial Exploratory Data Analysis (EDA) ---

print("\n--- First 5 Rows (Head) ---")
print(df_metadata.head())

print("\n--- DataFrame Info ---")
df_metadata.info()  # Prints info to stdout, no need for print()

print("\n--- Descriptive Statistics (Numerical Columns) ---")
# Include all types for a broader overview initially
print(df_metadata.describe(include='all'))

print("\n--- Value Counts for Object/Categorical Columns ---")
# Limit to columns with a reasonable number of unique values for clarity
MAX_UNIQUE_VALUES_FOR_COUNTS = 20
for col in df_metadata.select_dtypes(include=['object', 'category']).columns:
    if df_metadata[col].nunique() <= MAX_UNIQUE_VALUES_FOR_COUNTS:
        print(f"\nValue Counts for column: '{col}'")
        print(df_metadata[col].value_counts())
    else:
        print(
            f"\nSkipping value counts for '{col}' (>{MAX_UNIQUE_VALUES_FOR_COUNTS} unique values).")


print("\n\nEDA script finished.")

# --- Next Steps ---
# You can add more specific analysis below, for example:
# - Distribution plots for numerical features (histograms, boxplots)
# - Correlation analysis
# - Handling missing values (if any)
# - Feature engineering
# ... etc.
