import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('eccv16\shape_predictor_5_face_landmarks.dat')

sticker_img = cv2.imread('sticker\glasses.png', cv2.IMREAD_UNCHANGED)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if ret == False:
        break

    dets = detector(frame)

    for det in dets:
        shape = predictor(frame, det)
        # for i, point in enumerate(shape.parts()):
        #     cv2.circle(frame, center = (point.x, point.y), radius = 2, color = (0, 0, 255), thickness=-1)
        #     cv2.putText(frame, text = str(i), org=(point.x, point.y), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.8, color=(0, 0, 255), thickness=2)
        try:
            # x1 = det.left()
            # y1 = det.top()
            # x2 = det.right()
            # y2 = det.bottom()

            # cv2.rectangle(frame, pt1 = (x1, y1), pt2 = (x2, y2), color=(0, 0, 255), thickness=2)
        
            glass_x1 = shape.parts()[2].x - 20
            glass_x2 = shape.parts()[0].x + 20

            glass_w = glass_x2 - glass_x1
            h, w, c = sticker_img.shape
            
            glass_h = int(glass_w * h / w)
            glass_center = int((shape.parts()[2].y + shape.parts()[0].y) / 2)
            glass_y1 = int(glass_center - glass_h / 2)
            glass_y2 = int(glass_center + glass_h / 2)

            overlay_img = sticker_img.copy()
            overlay_img = cv2.resize(overlay_img, dsize=(glass_w, glass_h))
            overlay_alpha = overlay_img[:, :, 3:4] / 255.0
            background_alpha = 1.0 - overlay_alpha

            frame[glass_y1:glass_y2, glass_x1:glass_x2] = overlay_img[:, :, :3] * overlay_alpha + frame[glass_y1:glass_y2, glass_x1:glass_x2] * background_alpha
        except:
            pass

    cv2.imshow('img', frame)
    if cv2.waitKey(5) == ord('q'):
        break