import cv2
import numpy as np

img_orig = cv2.imread('./image/bike.png', cv2.IMREAD_COLOR)  # 원본
img_piece = cv2.imread('./cut/cropped_4.png', cv2.IMREAD_COLOR)  # 잘린 변형된 이미지
#
first_piece = cv2.imread('./cut/cropped_1.png', cv2.IMREAD_COLOR)  # 다른 이미지를 합칠 첫 번째 이미지

# SIFT 필터
sift = cv2.SIFT_create()
keypoints_orig, descriptors_orig = sift.detectAndCompute(img_orig, None)
keypoints_piece, descriptors_piece = sift.detectAndCompute(img_piece, None)
#
keypoints_1st, descriptors_1st = sift.detectAndCompute(first_piece, None)

bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)  # bf?
matches = bf.match(descriptors_orig, descriptors_piece)
matches = sorted(matches, key=lambda x:x.distance)
print('len(matches):', len(matches))
# 
matches_1st = bf.match(descriptors_orig, descriptors_1st)
matches_1st = sorted(matches_1st, key=lambda x:x.distance)


# calculating transformation
src_points = []
dst_points = []
for match in matches:
    src_points.append(keypoints_orig[match.queryIdx].pt)
    dst_points.append(keypoints_piece[match.trainIdx].pt)
matching, _ = cv2.findHomography(np.array(dst_points), np.array(src_points), cv2.RANSAC)

src_points_1st = []
dst_points_1st = []
for m1 in matches_1st:
    src_points_1st.append(keypoints_orig[m1.queryIdx].pt)
    dst_points_1st.append(keypoints_1st[m1.trainIdx].pt)
matching_1st, _ = cv2.findHomography(np.array(dst_points_1st), np.array(src_points_1st), cv2.RANSAC)

# apply transformation to the cropped piece (이미지 복원)
restored_piece = cv2.warpPerspective(img_piece, matching, (img_orig.shape[1], img_orig.shape[0]))
cv2.imwrite('./sift/restored_piece.jpg', restored_piece)

restored_1st_piece = cv2.warpPerspective(first_piece, matching_1st, (img_orig.shape[1], img_orig.shape[0]))
cv2.imwrite('./sift/temp.jpg', restored_1st_piece)  # 이미지 조각을 합칠 이미지

# combine image(test)
temp_img = cv2.imread('./sift/temp.jpg')
temp_img += restored_piece
cv2.imwrite('./sift/combined_temp.jpg', temp_img)

matched_img = cv2.drawMatches(img_orig, keypoints_orig, img_piece, keypoints_piece, matches[:50], img_piece, flags=2)
cv2.imwrite('./sift/matched_images.jpg', matched_img)