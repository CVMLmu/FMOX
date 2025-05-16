import json
import pandas as pd
from collections import defaultdict


def fmox_obj_size_count(fmox_json_path, obj_size_distribution_path):
    column_names = ["Main Dataset", "Subsequence", "Object Size Distributions"]

    df = pd.DataFrame(columns=column_names)

    with open(fmox_json_path, 'r') as json_file:  # Load the JSON data from the file
        data = json.load(json_file)

    # Iterate through the databases
    for database in data["databases"]:
        # Iterate through the sub-datasets
        for sub_dataset in database["sub_datasets"]:
            all_obj_size = []

            # note: some sequences does not have annotations (e.g. fmov2 swaying) so skip them.
            if len(sub_dataset["images"]) != 0:
                # Initialize a dictionary to count category occurrences
                category_count = defaultdict(int)

                # Iterate through the images
                for image in sub_dataset["images"]:
                    # Iterate through the annotations
                    index = 0
                    for annotation in image["annotations"]:
                        category_name = annotation["size_category"]

                        current_obj_wh = annotation["object_wh"]
                        current_obj_size = current_obj_wh[0] * current_obj_wh[1]
                        all_obj_size.append(current_obj_size)

                        # Increment the count for this category
                        category_count[category_name] += 1

                obj_size_distribution = {}
                for category, count in category_count.items():
                    obj_size_distribution[category] = count

                # print(database["dataset_name"], sub_dataset["subdb_name"], obj_size_distribution)

                row_data = {}  # Store the row values in a dictionary
                row_data["Main Dataset"] = database["dataset_name"]
                row_data["Subsequence"] = sub_dataset["subdb_name"]
                row_data["Object Size Distributions"] = obj_size_distribution

                new_row = pd.DataFrame([row_data])  # Create a DataFrame from the dictionary for the new row
                df = pd.concat([df, new_row], ignore_index=True)  # Concatenate the new row to the existing DataFrame
                print("Object Size Distributions Samples:", df.head())

    # Save the DataFrame to a CSV file - Set index=False to avoid saving the index as a column
    df.to_csv(obj_size_distribution_path, index=False)

# fmox_json_path = " "
# json_obj_size_count(fmox_json_path)

