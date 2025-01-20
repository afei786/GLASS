import cv2
import onnxruntime as ort
import numpy as np
from PIL import Image

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

if __name__ == '__main__':
    # read image
    image = Image.open("/home/fei/data/new_images_seg/cxjzq11/test/cxjzq1_A-17_1918_1819_A-17_2024110600010.jpg").convert("RGB")
    image = image.resize((288, 288))
    image = np.array(image).astype(np.float32) / 255.0

    image = (image - mean) / std
    image = image.transpose((2, 0, 1))
    image_batch = np.expand_dims(image, axis=0).astype(np.float32)
    image_batch = np.repeat(image_batch, 8, axis=0)

    # load model
    r50_session = ort.InferenceSession("results/models/backbone_0/mvtec_cxjzq1/glass_simplified.onnx",
                                       providers=['CPUAExecutionProvider'])
    output = r50_session.run(None, {"input": image_batch})[0]
    print(output.shape)

    # segmentation output
    output = np.expand_dims(output, axis=1)[0]
    output = output.transpose((1, 2, 0))
    output = cv2.resize(output, (288, 288), interpolation=cv2.INTER_LINEAR)
    output = cv2.GaussianBlur(output, (33, 33), 4)

    # binary or heatmap
    # ret, mask = cv2.threshold(output, 0.5, 255, cv2.THRESH_BINARY)
    mask = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)
    mask = (mask * 255).astype('uint8')
    mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
    cv2.imwrite("01.png", mask)
