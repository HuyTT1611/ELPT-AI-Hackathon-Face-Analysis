import onnxruntime
import numpy as np
from PIL import Image
import torch
import glob
import os
import csv
import time
from PIL import Image
from ultralytics import YOLO
torch.set_printoptions(precision=10)
import json
from tools.utils import set_seed
from config import argument_parser

set_seed(605)


def main(args):
    attribute=['Baby','Kid','Teenager','20-30s','40-50s','Senior','Negroid','Caucasian','Mongoloid','unmasked','masked','mid-light','light','mid-dark','dark','Fear','Disgust','Surprise','Anger','Sadness','Neutral','Happiness','Female','Male']
    print(onnxruntime.get_device())
    onnx_model_path = args.onnx_model_path
    ort_session = onnxruntime.InferenceSession(onnx_model_path)
    input_name = ort_session.get_inputs()[0].name
    output_name = ort_session.get_outputs()[0].name

    model_yolo = YOLO(args.yolo_dir)

    # json_file_path = '/data/phuongdd/Computer_vision/Multi_attr_cls/face_attributes/public_test_and_submission_guidelines/file_name_to_image_id.json'
    # with open(json_file_path, 'r') as file:
    #         json_data = json.load(file)
            
    # image_list = list(json_data.keys())


    imgs_path = args.root_path
    data = []

    first_line = ['file_name', 'bbox', 'image_id', 'race', 'age', 'emotion', 'gender', 'skintone', 'masked']
    for i,filename in enumerate(os.listdir(imgs_path)):
        print(f"------------------Image {filename} --------------------")
        image = Image.open(os.path.join(imgs_path, filename)).convert('RGB')
        copy_image = image.copy()

        start1=time.time()
        results = model_yolo(image)
        image_width, image_height = image.size
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x,y, width, height = box.xywh[0]
                x=x-width/2
                y=y-height/2
                cropped_image = copy_image.crop((int(x), int(y), int(x + width), int(y + height)))
                
                # input_image = cropped_image.resize((192, 256))  # Resize to the model's input shape
                input_image = cropped_image.resize((224, 224))  # Resize to the model's input shape
                
                input_image = np.array(input_image).astype(np.float32)  # Convert to numpy array
                input_image = input_image.astype(np.float32) / 255

                input_image = np.transpose(input_image, (2, 0, 1))
                input_image = input_image[np.newaxis, :]  # Add a batch dimension


                results_onnx = ort_session.run([output_name], {input_name: input_image})
                output = results_onnx[0]
                output = torch.tensor(output)
                valid_probs = torch.sigmoid(output)

                print('---------------------Time inference--------- === ', time.time()-start1)
                for valid_prob in valid_probs:
                    new_row = []
                    new_row.append(filename)
                    new_row.append([float(x.cpu()),float(y.cpu()), float(width.cpu()), float(height.cpu())])
                    new_row.append(int(i+1))
                    _, max_index_race = torch.max(valid_prob[6:9], dim=0)
                    _, max_index_age = torch.max(valid_prob[0:6], dim=0)
                    _, max_index_emotion = torch.max(valid_prob[15:22], dim=0)
                    _, max_index_gender = torch.max(valid_prob[22:24], dim=0)
                    _, max_index_skintone = torch.max(valid_prob[11:15], dim=0)
                    _, max_index_masked = torch.max(valid_prob[9:11], dim=0)
                    new_row.append(attribute[max_index_race+6])
                    new_row.append(attribute[max_index_age])
                    new_row.append(attribute[max_index_emotion+15])
                    new_row.append(attribute[max_index_gender+22])
                    new_row.append(attribute[max_index_skintone+11])
                    new_row.append(attribute[max_index_masked+9])
                    print(new_row)
                    data.append(new_row)

    data = [first_line] + data
    answer_csv_file = "./answer.csv"
    with open(answer_csv_file, mode='w', newline='') as answer_file:
        answer_writer = csv.writer(answer_file)
        for row in data:
            answer_writer.writerow(row)
if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()

    main(args)
