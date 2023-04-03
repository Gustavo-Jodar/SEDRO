import cv2
import imutils

FILE = "../../video.MP4"
cap = cv2.VideoCapture(FILE)

win_size = (128, 256)
block_size = (32, 32)
block_stride = (8, 8)
cell_size = (8, 8)
nbins = 9

hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, nbins)
hog.setSVMDetector

while True:
    ret, image = cap.read()
    #frame = cv2.resize(frame, (600, 500))
        
    # Initializing the HOG person detector
    # Reading the Image
    #image = cv2.imread("pedestres.jpg")

    # Resizing the Image
    image = imutils.resize(image,width=min(400, image.shape[1]))

    # Detecting all the regions in the image that has a pedestrians inside it
    (regions, _) = hog.detectMultiScale(image, winStride=(8, 8),padding=(16, 16),scale=1.02)

    # Drawing the regions in the Image
    for (x, y, w, h) in regions:
        cv2.rectangle(image, (x, y),(x + w, y + h),(0, 0, 255), 2)

    # Showing the output Image
    cv2.imshow("img", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()