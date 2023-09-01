import cv2
import numpy as np
import sys
import os

if len(sys.argv) != 5:
    print("Usage: python3 merge_image.py input_filename_prefix column_num row_num output_filename")
    sys.exit(1)

input_prefix = sys.argv[1]
M, N = int(sys.argv[2]), int(sys.argv[3])
output_name = sys.argv[4]

with open('./cut/img2cut.txt', 'r') as f:
    image_name = f.read().strip()
img_orig = cv2.imread('./image/' + image_name, cv2.IMREAD_COLOR)

# SIFT
sift = cv2.SIFT_create()
keypoints_orig, descriptors_orig = sift.detectAndCompute(img_orig, None)

merged_images_list = []
for m in range(M):
    row_images_list = []
    for n in range(N):
        piece_filename = f'./cut/{input_prefix}_{m*N + n + 1}.png'
        
        if os.path.exists(piece_filename):
            img_piece = cv2.imread(piece_filename, cv2.IMREAD_COLOR)
            keypoints_piece, descriptors_piece = sift.detectAndCompute(img_piece, None)
            
            bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)
            matches = bf.match(descriptors_orig, descriptors_piece)
            matches = sorted(matches, key=lambda x:x.distance)
            
            if len(matches) < 4:
                continue

            # 이미지 변환 계산
            src_points = []
            dst_points = []
            for match in matches:
                src_points.append(keypoints_orig[match.queryIdx].pt)
                dst_points.append(keypoints_piece[match.trainIdx].pt)
            matching, _ = cv2.findHomography(np.array(dst_points), np.array(src_points), cv2.RANSAC)

            # 잘린 이미지 복원
            restored_piece = cv2.warpPerspective(img_piece, matching, (img_orig.shape[1], img_orig.shape[0]))
            
            row_images_list.append(restored_piece)

        else:
            print(f"'{piece_filename}'이란 파일이 존재하지 않습니다.")
            sys.exit(1)
    merged_row = np.zeros_like(row_images_list[0])
    for p in row_images_list:
        merged_row += p
    merged_images_list.append(merged_row)

merged_img = np.zeros_like(merged_images_list[0])
for row in merged_images_list:
    merged_img += row

cv2.imwrite('./merge/' + output_name + '.jpg', merged_img)
