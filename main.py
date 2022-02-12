# DETECTING MOVEMENT FROM CAMERA AND WHEN ITS FIND ONE ITS SAVES THE PHOTO IN 'images' DIRECTORY WITH THE DATE AND TIME
import os, cv2, datetime

def main():
    camera = cv2.VideoCapture(0)
    sensitivity = 20
    timeNow = ""

    prev_frame = None  # for compare between 2 images and find different pixels
    while True:
        success, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        if prev_frame is None:
            prev_frame = gray
            continue

        diff = cv2.absdiff(gray, prev_frame)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

        prev_frame = gray

        contours, __ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        print(len(contours))

        if len(contours) < sensitivity:
            updatedframe = cv2.putText(frame, 'MOVEMENTS: NOT FOUND', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                       (0, 255, 0), 1, cv2.LINE_AA)
        else:
            updatedframe = cv2.putText(frame, 'MOVEMENTS: FOUND', (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),
                                       1, cv2.LINE_AA)
            timeNow = (str(datetime.datetime.now()).replace(":", "-")).split(".")[
                0]  # get current time for save the image with this name
            path = os.path.join(os.getcwd(), "Camera", timeNow + ".jpg")
            cv2.imwrite(path, frame)  # save photo

        cv2.imshow("CAMERA", updatedframe)
        cv2.imshow("MOVEMENTS", thresh)
        if cv2.waitKey(50) == ord('q'):
            break

    cv2.destroyAllWindows()
    camera.release()

if __name__ == "__main__":
    main()
