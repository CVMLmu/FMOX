import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

file_path = './json_anns/json_annotation_analysis_output.csv'
data = pd.read_csv(file_path)

print(data.head())

def cvs_viz1():
    # Bar plot of "Average Object Size" by "Main Dataset"
    plt.figure(figsize=(10, 6))
    sns.barplot(data=data, x='Main Dataset', y='Average Object Size', ci=None)
    plt.title('Average Object Size by Main Dataset')
    plt.xlabel('Main Dataset')
    plt.ylabel('Average Object Size')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def cvs_viz2():
    # Scatter plot of "Total Frame Number" vs "FMO Exists Frame Number"
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x='Total Frame Number', y='FMO Exists Frame Number', hue='Main Dataset')
    plt.title('Total Frame Number vs FMO Exists Frame Number')
    plt.xlabel('Total Frame Number')
    plt.ylabel('FMO Exists Frame Number')
    plt.legend(title='Main Dataset')
    plt.tight_layout()
    plt.show()


def cvs_viz3():
    # Box plot of "Average Object Size" by "Object Size Levels"
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data, x='Object Size Levels', y='Average Object Size')
    plt.title('Average Object Size by Object Size Levels')
    plt.xlabel('Object Size Levels')
    plt.ylabel('Average Object Size')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def visualize_object_size_levels():
    # Initialize a list to store the processed data
    processed_data = []

    # Process the "Object Size Levels" column
    for index, row in data.iterrows():
        main_dataset = row['Main Dataset']
        subsequence = row['Subsequence']
        levels_dict = ast.literal_eval(row['Object Size Levels'])

        for size, count in levels_dict.items():
            processed_data.append({
                'Main Dataset': main_dataset,
                'Subsequence': subsequence,
                'Size Level': size,
                'Count': count
            })

    # Create a DataFrame from the processed data
    processed_df = pd.DataFrame(processed_data)

    # Pivot the DataFrame to get counts for each size level by main dataset and subsequence
    pivot_df = processed_df.pivot_table(index=['Main Dataset', 'Subsequence'],
                                        columns='Size Level',
                                        values='Count',
                                        fill_value=0)

    pivot_df.plot(kind='bar', stacked=True, figsize=(12, 8))
    plt.title('Object Size Levels by Main Dataset and Subsequence')
    plt.xlabel('Main Dataset and Subsequence')
    plt.ylabel('Count')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Size Level')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.show()
