import json
import pandas as pd
from collections import defaultdict


def json_to_csv():
    column_names = ["Main Dataset", "Subsequence",
                    "Total Frame Number", "FMO Exists Frame Number",
                    "Average Object Size", "Object Size Levels"]

    df = pd.DataFrame(columns=column_names)

    # Load the JSON data from the file
    with open('../FMOX-Jsons/FMOX_fall_and_tbd3d.json', 'r') as json_file:
        data = json.load(json_file)

    # Iterate through the databases
    for database in data["databases"]:

        # Iterate through the sub-datasets
        for sub_dataset in database["sub_datasets"]:
            print(sub_dataset["subdb_name"])
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

                keep = {}
                for category, count in category_count.items():
                    keep[category] = count

                print(keep, "\n")


json_to_csv()