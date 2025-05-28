import json
import csv
import os

def convert_json_to_csv(json_folder, csv_folder):
    """
    Converts all JSON files from a specified folder to CSV format
    and saves them in another specified folder.

    Args:
        json_folder (str): The path to the folder containing JSON files.
        csv_folder (str): The path to the folder where CSV files will be saved.
    """
    # Ensure the JSON input folder exists
    if not os.path.exists(json_folder):
        print(f"Error: JSON input folder '{json_folder}' not found.")
        return

    # Create the CSV output folder if it doesn't exist
    os.makedirs(csv_folder, exist_ok=True)
    print(f"CSV output folder '{csv_folder}' ensured to exist.")

    # Get a list of all files in the JSON folder
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]

    if not json_files:
        print(f"No JSON files found in '{json_folder}'.")
        return

    print(f"Found {len(json_files)} JSON files to process in '{json_folder}'.")

    # Process each JSON file
    for json_file_name in json_files:
        json_file_path = os.path.join(json_folder, json_file_name)
        csv_file_name = os.path.splitext(json_file_name)[0] + '.csv'
        csv_file_path = os.path.join(csv_folder, csv_file_name)

        print(f"\nProcessing '{json_file_name}'...")

        try:
            # Read the JSON data from the file
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            # Ensure data is a list of dictionaries for consistent processing
            if not isinstance(data, list):
                # If it's a single dictionary, wrap it in a list
                data = [data]

            if not data:
                print(f"Warning: JSON file '{json_file_name}' is empty or contains no data.")
                continue

            # Extract headers (keys) from the first dictionary in the list
            # This assumes all objects in the JSON list have the same keys.
            # For more complex JSON structures, you might need a recursive function
            # to flatten the data and collect all unique headers.
            headers = []
            for item in data:
                if isinstance(item, dict):
                    for key in item.keys():
                        if key not in headers:
                            headers.append(key)
                # Handle cases where items in the list might not be dictionaries
                # or if nested structures need flattening.
                # For simplicity, this script assumes flat dictionaries.
            
            if not headers:
                print(f"Warning: No headers could be extracted from '{json_file_name}'. Skipping.")
                continue

            # Write the data to a CSV file
            with open(csv_file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=headers)

                # Write the header row
                writer.writeheader()

                # Write the data rows
                for row in data:
                    # Ensure all values are strings to prevent CSV writing issues
                    # For nested JSON, you might need to flatten dicts/lists before writing
                    flat_row = {k: str(v) if not isinstance(v, (dict, list)) else json.dumps(v) for k, v in row.items()}
                    writer.writerow(flat_row)

            print(f"Successfully converted '{json_file_name}' to '{csv_file_name}'.")

        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from '{json_file_name}'. Please check its format.")
        except FileNotFoundError:
            print(f"Error: File '{json_file_path}' not found (should not happen if os.listdir works).")
        except Exception as e:
            print(f"An unexpected error occurred while processing '{json_file_name}': {e}")

# --- Configuration ---
# Define the input and output folder paths
# Make sure these folders exist relative to where you run the script,
# or provide absolute paths.
JSON_FOLDER = 'JSON'
CSV_FOLDER = 'CSV'

# --- Run the conversion ---
if __name__ == "__main__":
    convert_json_to_csv(JSON_FOLDER, CSV_FOLDER)