import glob
import cv2

desired_dimension = 512
skipped_images = 0
kept_images = 0

for filepath in glob.iglob('images/*.jpg'):
    img = cv2.imread(filepath)
    if (img.shape[0] >= desired_dimension and img.shape[1] >= desired_dimension):
        kept_images += 1
        res_img = cv2.resize(img, dsize=(512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'preprocessed_images/{str(kept_images)}.jpg', res_img)
    else:
        skipped_images +=1

print('Kept Images: ' + str(kept_images))
print('Skipped Images: ' + str(skipped_images))