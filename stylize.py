import sys
import cv2
import os

def main():
    if len(sys.argv) != 4:
        print("Usage: python stylize.py <path_to_image> <model_name.t7> <path_to_output>")
        sys.exit(1)

    image_path = sys.argv[1]
    model_path = os.path.join("models", sys.argv[2])
    output_path = sys.argv[3]

    if not os.path.exists(image_path):
        print(f"Error: image not found: {image_path}")
        sys.exit(1)

    if not os.path.exists(model_path):
        print(f"Error: model not found: {model_path}")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print("Error: could not load the image.")
        sys.exit(1)

    net = cv2.dnn.readNetFromTorch(model_path)

    h, w = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1.0, (w, h),
                                 (42, 54, 167),
                                 # (103.939, 116.779, 123.680),
                                 swapRB=False, crop=False)

    net.setInput(blob)
    output = net.forward()

    output = output.reshape((3, output.shape[2], output.shape[3]))
    output[0] += 42
    output[1] += 54
    output[2] += 167
    # output[0] += 103.939
    # output[1] += 116.779
    # output[2] += 123.680
    output = output.transpose(1, 2, 0)
    result = cv2.convertScaleAbs(output)

    cv2.imwrite(output_path, result)
    print(f"Image successufully redacted and saved as {output_path}")

if __name__ == "__main__":
    main()
