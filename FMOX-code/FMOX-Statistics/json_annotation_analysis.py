import json
import pandas as pd
from collections import defaultdict

def combine_and_save_jsons(json1,json2):
    try:
        # data1 = json.loads(json1)
        # data2 = json.loads(json2)
        with open(json1, 'r') as file:
            data1 = json.load(file)

        with open(json2, 'r') as file:
            data2 = json.load(file)

    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return

    # Combine the databases
    combined_databases = data1['databases'] + data2['databases']
    # Create a new combined structure
    combined_data = {"databases": combined_databases}

    with open('./json_anns/fmo_all4_annotations.json', 'w') as json_file:
        json.dump(combined_data, json_file, indent=4)


def json_to_csv():
    column_names = ["Main Dataset", "Subsequence",
                    "Total Frame Number", "FMO Exists Frame Number",
                    "Average Object Size", "Object Size Levels"]

    df = pd.DataFrame(columns=column_names)

    # Load the JSON data from the file
    with open('./json_anns/fmo_all4_annotations.json', 'r') as json_file:
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

                average_obj_size = int(sum(all_obj_size) / len(all_obj_size))

                print("Category Counts:")
                for category, count in category_count.items():
                    print(f"{category}: {count}")

                # get the total FMO exist frame number
                fmo_exists_frame_number = sub_dataset["images"][(len(sub_dataset["images"])-1)]["img_index"] - sub_dataset["images"][0]["img_index"] + 1

                row_data = {} # Store the row values in a dictionary
                row_data["Main Dataset"] = database["dataset_name"]
                row_data["Subsequence"] = sub_dataset["subdb_name"]
                row_data["Total Frame Number"] = sub_dataset["total_frame_num"]
                row_data["FMO Exists Frame Number"] = fmo_exists_frame_number
                row_data["Average Object Size"] = average_obj_size
                row_data["Object Size Levels"] = dict(sorted(category_count.items()))

                new_row = pd.DataFrame([row_data])  # Create a DataFrame from the dictionary for the new row
                df = pd.concat([df, new_row], ignore_index=True)  # Concatenate the new row to the existing DataFrame

    # Save the DataFrame to a CSV file - Set index=False to avoid saving the index as a column
    df.to_csv('./json_anns/json_annotation_analysis_output.csv', index=False)
    # Save the DataFrame to an Excel file  df.to_excel('output.xlsx', index=False)


