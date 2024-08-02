def save_annotations_to_yolo_format(annotations, image_width, image_height, output_file):
    # Define the class mapping
    class_mapping = {'cross': 0}  # Add more classes if needed

    # Prepare the output data
    yolo_annotations = []
    for ann in annotations:
        class_id = class_mapping[ann['label']]
        x_center = (ann['x'] + ann['width'] / 2) / image_width
        y_center = (ann['y'] + ann['height'] / 2) / image_height
        width = ann['width'] / image_width
        height = ann['height'] / image_height
        yolo_annotations.append(f"{class_id} {x_center} {y_center} {width} {height}")

    # Write to the file
    with open(output_file, 'w') as f:
        for line in yolo_annotations:
            f.write(line + '\n')

    print(f"Annotations saved to {output_file}")


# Example usage
annotations = [
    {'x': 299, 'y': 124, 'width': 130, 'height': 86, 'label': 'cross'},
    {'x': 224, 'y': 267, 'width': 124, 'height': 80, 'label': 'cross'},
    {'x': 443, 'y': 176, 'width': 138, 'height': 101, 'label': 'cross'},
    {'x': 306, 'y': 371, 'width': 120, 'height': 70, 'label': 'cross'}
]

image_width = 640  # Replace with actual image width
image_height = 480  # Replace with actual image height
output_file = 'annotations.txt'  # Change to your desired output file name

save_annotations_to_yolo_format(annotations, image_width, image_height, output_file)
