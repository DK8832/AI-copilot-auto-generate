import cv2
import numpy as np

proto = 'eccv16\colorization_deploy_v2.prototxt'
weight = 'eccv16\colorization_release_v2.caffemodel'

net = cv2.dnn.readNetFromCaffe(proto, weight)

pts_in_hull = np.load('eccv16\pts_in_hull.npy')
pts_in_hull = pts_in_hull.transpose().reshape(2, 313, 1, 1).astype(np.float32)

net.getLayer(net.getLayerId('class8_ab')).blobs = [pts_in_hull]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [np.full((1, 313), 2.606, np.float32)]

img = cv2.imread('image\Hepburn.jpg')
h, w, c = img.shape

img_input = img.copy()
img_input = img_input.astype('float32') / 255.0

img_lab = cv2.cvtColor(img_input, cv2.COLOR_BGR2LAB)
img_l = img_lab[:, :, :1]
    
blob = cv2.dnn.blobFromImage(img_l, size=(224, 224), mean=[50, 50, 50])

net.setInput(blob)
output = net.forward()

output = output.squeeze().transpose(1, 2, 0)
output_resized = cv2.resize(output, dsize=(w, h))

output_lab = np.concatenate([img_l, output_resized], axis=2)
output_bgr = cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)
output_bgr = output_bgr * 255
output_bgr = output_bgr.astype('uint8')
output_bgr = np.clip(output_bgr, 0, 255)

cv2.imshow('img', output_bgr)
cv2.waitKey(0)

