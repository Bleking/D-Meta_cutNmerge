import cv2
import random
import sys

if len(sys.argv) != 5:
    print("Usage: python3 cut_image.py image_file_name column_num row_num prefix_output_filename")
    sys.exit(1)

image_name = sys.argv[1]
M, N = int(sys.argv[2]), int(sys.argv[3])
output_name = sys.argv[4]

img_orig = cv2.imread('./image/' + image_name, cv2.IMREAD_COLOR)

# 자를 이미지 파일 명을 텍스트 파일에 저장해서 merge_image.py가 읽을 수 있도록 하기
with open('./cut/img2cut.txt', 'w') as f:
    f.write(image_name)

height, width, _ = img_orig.shape
height_piece = height // M
width_piece = width // N

# 잘린 이미지 무작위 순서대로 이름 붙이기
random_filenames = random.sample(range(1, (M * N) + 1), (M * N))

# 이미지 크기가 나눠 떨어지지 않으면 자를 수 없는 조건 추가하기
for m in range(M):
    for n in range(N):
        top = m * height_piece
        bottom = (m + 1) * height_piece
        left = n * width_piece
        right = (n + 1) * width_piece

        crop = img_orig[top:bottom, left:right]

        if random.random() < 0.5:
            crop = cv2.flip(crop, 1)  # 좌우 반전
        if random.random() < 0.5:
            crop = cv2.flip(crop, 0)  # 상하 반전
        if random.random() < 0.5:
            crop = cv2.rotate(crop, cv2.ROTATE_90_COUNTERCLOCKWISE)  # 반시계 방향 90도 회전
        
        random_filename = random_filenames.pop()
        cv2.imwrite('./cut/' + output_name + f'_{random_filename}.png', crop)  # cv2.imwrite('./cut/' + output_name + f'_{m}{n}.png', crop)
