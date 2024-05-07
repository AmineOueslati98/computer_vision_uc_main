"""
JSON annotation file doesn't have the bounding box coordinates for some images available in the dataset.
this script patch the JSON file to include the missing bounding box coordinates from the XML annotation file.
"""

import json
import xmltodict
import os


def xml_file_to_dict(xml_filename):
    """
    Convert XML annotation file to a dictionary.

    Args:
    xml_filename (str): Path to the XML file.

    Returns:
    dict: Dictionary containing annotation data.
    """
    try:
        with open(xml_filename) as xml_file:
            data_dict = xmltodict.parse(xml_file.read())

        objects = data_dict["annotation"]["object"]
        
        # In case of multiple bboxes
        if isinstance(objects, list):
            bboxs = [
                [int(coordinate) for coordinate in o["bndbox"].values()] for o in objects
            ]
            category_len = len(bboxs)
        else:
            bboxs = [int(coordinate) for coordinate in objects["bndbox"].values()]
            category_len = 1

        result = {
            "file_name": data_dict["annotation"]["filename"],
            "height": data_dict["annotation"]["size"]["height"],
            "width": data_dict["annotation"]["size"]["width"],
            "id": int(data_dict["annotation"]["filename"].split(".")[0]),
            "bbox": bboxs,
            "category_id": [0] * category_len,
        }

        return result
    
    except Exception as e:
        print(f"Error processing XML file '{xml_filename}': {e}")
        return None

def patch_json_from_xml(json_file, xml_annot_folder):
    """
    Patch JSON file with data from XML annotation files.

    Args:
    json_file (str): Path to the JSON file.
    xml_annot_folder (str): Path to the folder containing XML annotation files.

    Returns:
    None
    """
    try:
        with open(json_file) as f:
            data = json.load(f)

        # Extract file names in the json file
        file_names = [os.path.splitext(image["file_name"])[0] for image in data["images"]]

        # List of XML files
        xml_file_names = [
            os.path.splitext(xml_file)[0] for xml_file in os.listdir(xml_annot_folder)
        ]

        xml_files_not_included = [
            xml_file for xml_file in xml_file_names if xml_file not in file_names
        ]

        # Use the XML files from the folder to extend the json
        for xml_file_name in xml_files_not_included:
            xml_file_path = os.path.join(xml_annot_folder, xml_file_name + ".xml")
            new_entry = xml_file_to_dict(xml_file_path)
            data["images"].append(new_entry)

        json_filename = os.path.basename(json_file)
        output_json_file = os.path.join(os.path.dirname(json_file), 'updated_' + json_filename)

        with open(output_json_file, "w") as outfile:
            json.dump(data, outfile, indent=4)
    except Exception as e:
        print(f"Error patching JSON from XML: {e}")


if __name__ == "__main__":

    # patch train dataset
    train_json = "dataset-1280/train/annotations_json/train.json"
    train_xml = "dataset-1280/train/annotations_xml/"
    patch_json_from_xml(train_json, train_xml)

    # patch val dataset 
    train_json = "dataset-1280/val/annotations_json/val.json"
    train_xml = "dataset-1280/val/annotations_xml/"
    patch_json_from_xml(train_json, train_xml)

    # patch eval dataset
    train_json = "dataset-1280/test/annotations_json/test.json"
    train_xml = "dataset-1280/test/annotations_xml/"
    patch_json_from_xml(train_json, train_xml)