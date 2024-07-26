import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from segment_anything import sam_model_registry
from segment_anything import SamAutomaticMaskGenerator


def sam_masks(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)
    c_mask=[]
    for ann in sorted_anns:
        m = ann['segmentation']
        img = np.ones((m.shape[0], m.shape[1], 3))
        color_mask = np.random.random((1, 3)).tolist()[0]
        for i in range(3):
            img[:,:,i] = color_mask[i]
        ax.imshow(np.dstack((img, m*0.8)))
        c_mask.append(img)
    return c_mask


if __name__ == '__main__':

    print('CUDA available -> ', torch.cuda.is_available())
    print('CUDA GPU number -> ', torch.cuda.device_count())
    print('GPU -> ', torch.cuda.get_device_name())
    USED_D = torch.device('cuda:0')

    image_file = "/mobileye/RPT/users/kfirs/kfir_project/text2sql/MSC_Project/ortho_image_res2500.png"

    #When loading an image with openCV, it is in bgr by default
    loaded_img = cv2.imread(image_file)

    #Now we get the R,G,B image
    image_rgb = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB)

    MODEL = "/opt/data/sam_vit_h_4b8939.pth"

    sam = sam_model_registry["vit_h"](checkpoint = MODEL)

    #Cast your model to a specific device (cuda or cpu)
    sam.to(device = USED_D)

    mask_generator = SamAutomaticMaskGenerator(sam)
    result = mask_generator.generate(image_rgb)

    fig = plt.figure(figsize=(np.shape(image_rgb)[1] / 72, np.shape(image_rgb)[0] / 72))
    fig.add_axes([0, 0, 1, 1])
    plt.imshow(image_rgb)
    color_mask = sam_masks(result)
    plt.axis('off')
    plt.savefig("/mobileye/RPT/users/kfirs/kfir_project/text2sql/MSC_Project/test_result.jpg")