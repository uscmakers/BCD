import pandas as pd
import sys # To exit gracefully on error
import os # To interact with the operating system (check directory)
import glob # To find files matching a pattern

# --- Configuration ---
# Directory containing the CSV files to process
target_directory = 'eeg_samples/'
label_column = 'Label'

# --- Check if target directory exists ---
if not os.path.isdir(target_directory):
    print(f"Error: Directory not found at {target_directory}")
    sys.exit(1)

# --- Find CSV files ---
csv_files = glob.glob(os.path.join(target_directory, '*.csv'))

if not csv_files:
    print(f"No CSV files found in {target_directory}")
    sys.exit(0)

print(f"Found {len(csv_files)} CSV files to process in {target_directory}")

# --- Process each CSV file ---
total_rows_removed = 0
files_processed = 0
files_modified = 0

for file_path in csv_files:
    print(f"\n--- Processing file: {file_path} ---")
    try:
        # --- Load Data ---
        try:
            df = pd.read_csv(file_path)
            print(f"Successfully loaded {os.path.basename(file_path)}")
        except FileNotFoundError:
            # This shouldn't happen with glob, but good practice
            print(f"Error: File not found at {file_path}")
            continue # Skip to the next file
        except Exception as e:
            print(f"Error reading CSV file {os.path.basename(file_path)}: {e}")
            continue # Skip to the next file

        # --- Deduplication Logic ---

        # Identify columns to compare (all except the label column)
        signal_columns = [col for col in df.columns if col != label_column]

        # Check for empty DataFrame or insufficient columns
        if df.empty:
            print("DataFrame is empty. No changes made.")
            files_processed += 1
            continue # Skip to the next file
        elif not signal_columns:
            print(f"No columns found other than '{label_column}'. No changes made.")
            files_processed += 1
            continue # Skip to the next file
        else:
            # print(f"Comparing columns for duplication: {signal_columns}") # Can be verbose

            # Create a boolean mask: Keep row if any signal column is different from the previous row.
            mask_keep = df[signal_columns].ne(df[signal_columns].shift()).any(axis=1)

            # Filter the DataFrame
            df_deduplicated = df[mask_keep].copy() # Use .copy() to avoid SettingWithCopyWarning

            # --- Report and Save ---
            original_rows = len(df)
            deduplicated_rows = len(df_deduplicated)
            rows_removed = original_rows - deduplicated_rows

            print(f"Original rows: {original_rows}")
            print(f"Rows after deduplication: {deduplicated_rows}")
            print(f"Rows removed: {rows_removed}")

            if rows_removed > 0:
                try:
                    # Save the deduplicated data back to the original file
                    df_deduplicated.to_csv(file_path, index=False)
                    print(f"Deduplicated data saved back to {file_path}")
                    total_rows_removed += rows_removed
                    files_modified += 1
                except Exception as e:
                    print(f"Error writing deduplicated data back to {file_path}: {e}")
            else:
                print("No duplicate consecutive signal rows found. File remains unchanged.")

            files_processed += 1

    except Exception as e:
        # Catch any unexpected error during the processing of a single file
        print(f"An unexpected error occurred while processing {file_path}: {e}")
        # Optionally add more specific error handling or logging here

# --- Final Summary ---
print("\n--- Deduplication Summary ---")
print(f"Processed {files_processed} files.")
print(f"Modified {files_modified} files.")
print(f"Total rows removed across all files: {total_rows_removed}")
print("-----------------------------")
