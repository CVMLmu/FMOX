import os
import json
from .loaders_helpers import *
from .reporters import *
import numpy as np


object_size_labels = {"extremely_tiny":((1,1), (8,8)),
					  "tiny":((8,8),(16,16)),
					  "small":((16,16),(32,32)),
					  "medium":((32,32),(96,96)),
					  "large":((96,96),(100000, 10000))} # assumed that object could not be larger than 100000x100000

# if w1  <= w <  w2  and  h1 <=  h <  h2  assign label else try others
def get_obj_size_category1(obj_width,obj_height):
	obj_size_category = ""
	for label_name, label_value in object_size_labels.items():
		start_point, end_point = label_value
		w1, h1 = start_point
		w2, h2 = end_point

		if  w1  <= obj_width <  w2  and  h1 <=  obj_height <  h2:
			obj_size_category = str(label_name)

		elif obj_width > obj_height:
			print(f"Width is greater: {obj_width}")
			if w1 <= obj_width < w2:
				obj_size_category = str(label_name)

		elif obj_height > obj_width:
			print(f"Height is greater: {obj_height}")
			if h1 <= obj_height < h2:
				obj_size_category = str(label_name)
		else:
			continue

	return obj_size_category


# according to object area - e.g. the area of a 32 x 32 pixel object is 1024 square pixels.
def get_obj_size_category2(obj_width, obj_height):
	obj_size_category = ""
	for label_name, label_value in object_size_labels.items():
		start_point, end_point = label_value
		w1, h1 = start_point
		w2, h2 = end_point

		if w1*h1 <= obj_width*obj_height < w2*h2:
			obj_size_category = str(label_name)
		else:
			continue
	return obj_size_category


def create_json_for_three_dataset():
	falling_path = './fmo_data_extracted_files/Falling_Object'
	tbd_path = './fmo_data_extracted_files/TbD'
	tbd3d_path = './fmo_data_extracted_files/TbD-3D'

	# Initialize the main data structure
	data = {
		"databases": []
	}

	files = get_falling_dataset(falling_path)
	data1 = evaluate_on(data, files)
	files = get_tbd3d_dataset(tbd3d_path)
	data2 = evaluate_on(data1, files)
	# files = get_tbd_dataset(tbd_path)   # fall_coin ping_wall not inside the annotation ....
	# data3 = evaluate_on(data2, files)

	# Save the data to a JSON file
	with open('./json_anns/three_fmo_data_annotations.json', 'w') as json_file:
		json.dump(data2, json_file, indent=4)  # indent for pretty printing

	
def evaluate_on(data, files, callback=None):
	dataset_name = os.path.split(os.path.split(os.path.split(files[0])[0])[0])[-1]

	print("\nDataset name", dataset_name)
	db_entry = {
		"dataset_name": dataset_name,
		"version": "1.0",
		"description": f"{dataset_name} containing bounding box and object size annotations.",
		"sub_datasets": []
	}

	medn = 50
	for kkf, ff in enumerate(files):
		sub_dataset_name = os.path.basename(os.path.normpath(ff))
		total_frame_num = len(os.listdir(ff))

		# Count total files excluding specific .txt files
		# total_frame_num = len([f for f in os.listdir(ff) if f not in ['gt.txt', 'gtr.txt'] and not f.endswith('.txt')])
		total_frame_num = len([f for f in os.listdir(ff) if f.endswith('.png')])
		# 'TbD-3D' has gt.txt, gtr.txt and TbD has gt.txt some videos also

		print("sub dataset path", ff, "total frame number", total_frame_num)
		gtp = GroundTruthProcessor(ff,kkf,medn)

		# Loop to fill in the sub-datasets
		sub_dataset_entry = {
			"subdb_name": sub_dataset_name,
			"total_frame_num": total_frame_num,
			"images": []
		}

		img_index = 0
		for kk in range(gtp.nfrms):
			gt_traj, radius, bbox = gtp.get_trajgt(kk)
			I, B, original_I, current_image_name = gtp.get_img(kk)
			gt_hs = gtp.get_hs(kk)

			image_file_name = current_image_name
			annotations = []
			x_min, y_min, x_max, y_max = 0,0,0,0  # keep the place

			bbox = extend_bbox_uniform(bbox,radius,I.shape)
			bbox_tight = bbox_fmo(extend_bbox_uniform(bbox.copy(),10,I.shape),gt_hs,B)
			obj_dim = [0,0]
			for timei in range(gt_hs.shape[3]):
				bbox_temp = bbox_detect_hs(crop_only(gt_hs[:,:,:,timei],bbox_tight), crop_only(B,bbox_tight))
				if len(bbox_temp) == 0:
					bbox_temp = bbox_tight
				obj_dim_temp = bbox_temp[2:] - bbox_temp[:2]
				obj_dim[0] = max(obj_dim[0],obj_dim_temp[0])
				obj_dim[1] = max(obj_dim[1],obj_dim_temp[1])

				# from crop_only Is[bbox[0]:bbox[2], bbox[1]:bbox[3]] -> ROI = image[y:y+h, x:x+w]
				y_min = int(bbox_tight[0])
				x_min = int(bbox_tight[1])
				x_max = int(bbox_tight[3])
				y_max = int(bbox_tight[2])

			obj_width = int(x_max) - int(x_min)
			obj_height = int(y_max) - int(y_min)

			# Fill the annotations
			annotations.append({
				"bbox_xyxy": [int(x_min), int(y_min), int(x_max), int(y_max)],
				"object_wh": (obj_width,obj_height),
				"size_category": get_obj_size_category2(obj_width, obj_height)
			})

			# Create the image entry
			img_index += 1
			image_entry = {
				"img_index": img_index,
				"image_file_name": image_file_name,
				"annotations": annotations
			}

			# Add the image entry to the sub-dataset
			sub_dataset_entry["images"].append(image_entry)

			# cv2.rectangle(original_I, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)
			# # cv2.imwrite('./output_image'+ str(kk) + '.jpg', original_I)
			# cv2.imshow('OpenCV Image', original_I)
			# cv2.waitKey(0)
			# cv2.destroyAllWindows()

		# Add the sub-dataset entry to the database
		db_entry["sub_datasets"].append(sub_dataset_entry)

	# Add the database entry to the main data structure
	data["databases"].append(db_entry)

	return data


# if __name__ == "__main__":
# 	create_json_for_three_dataset()

