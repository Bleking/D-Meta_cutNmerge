import cv2
import sys
import os

if len(sys.argv) != 5:
    print("Usage: python3 merging.py input_filename_prefix column_num row_num output_filename")
    sys.exit(1)

input_prefix = sys.argv[1]
M, N = int(sys.argv[2]), int(sys.argv[3])
output_name = sys.argv[4]

merged_images_list = []
for m in range(M):
    row_images_list = []
    for n in range(N):
        piece_filename = f'./cut/{input_prefix}_{m*N + n + 1}.png'
        if os.path.exists(piece_filename):
            # img_piece = piece_filename  # 파일이 제대로 입력됐는지 확인 용
            img_piece = cv2.imread(piece_filename, cv2.IMREAD_COLOR)
            
            # img_piece가 회전되면 세로가 길테니 세로가 길면 회전시켜야 함
            h, w, _ = img_piece.shape
            if h > w:
                img_piece = cv2.rotate(img_piece, cv2.ROTATE_90_CLOCKWISE)
            
            row_images_list.append(img_piece)
        else:
            print(f"'{piece_filename}'이란 파일이 존재하지 않습니다.")
            sys.exit(1)
    
    # merged_images_list.append(row_images_list)  # 파일이 제대로 입력됐는지 확인 용
    merged_images_list.append(cv2.hconcat(row_images_list))

# print(merged_images_list)  # 파일이 제대로 입력됐는지 확인 용
merged_img = cv2.vconcat(merged_images_list)
cv2.imwrite('./merge/' + output_name + '.png', merged_img)


# import cv2
# import numpy as np
# import os

# M, N = 3, 3
# output_name = 'merged'

# merged_height = M * height_piece
# merged_width = N * width_piece

# merged_image = np.zeros((merged_height, merged_width, 3), dtype=np.uint8)

# for m in range(M):
#     for n in range(N):
#         random_filename = random_filenames.pop()
#         piece = cv2.imread(f'./cut/{output_name}_{random_filename}.png')

#         # If pieces have varying sizes, you need to resize them to the same size
#         piece_resized = cv2.resize(piece, (width_piece, height_piece))

#         top = m * height_piece
#         bottom = (m + 1) * height_piece
#         left = n * width_piece
#         right = (n + 1) * width_piece

#         merged_image[top:bottom, left:right] = piece_resized

# cv2.imwrite(f'./cut/{output_name}.png', merged_image)
