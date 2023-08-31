# D-Meta_task
디메타 과제 면접

## 코드 실행하기
### 방법 1
1. 리눅스 환경에 접속하여 `cd dmeta` 커맨드를 입력하여 'dmeta' 경로로 이동합니다.
2. 이미지 cropping을 하기 위해 `python cut_image.py ${자를 이미지명.파일타입} ${M} ${N} {sub image명}` 명령어를 터미널에 입력하면 잘린 이미지 조각(sub image)들이 'meta/cut/' 경로에 형성됩니다
   - 예를들어, 'bike.png'라는 이미지를 2x2 크기로 자르고 싶고, 각 잘라진 sub image의 이름을 'crop'으로 하고 싶을 경우, 다음과 같이 입력하면 됩니다.
   
   `python cut_image.py bike.png 2 2 crop`
   
3. 이 잘린 이미지들을 합병하기 위해 `python merge_image.py ${sub image명} ${M} ${N} ${merged image명}` 명령어를 터미널에 입력하면 'dmeta/merge/' 경로에 원본 이미지와 유사한 합병된 이미지가 저장됩니다.
   - 예를들어 sub image들의 이름을 'crop'으로 이름붙였고, 병합된 이미지 파일명을 'result'로 하고 싶으면 아래와 같이 입력하면 됩니다. 단, M, N값은 이미지를 자를 때 입력한 값과 동일해야 합니다.
   
   `python merge_image.py crop 2 2 result`

### 방법 2
1. 리눅스 환경에 접속하여 `cd dmeta` 커맨드를 입력하여 'dmeta' 경로로 이동합니다.
2. 커맨드 창에 `chmod +x cutNmerge.sh`를 입력합니다.
3. 커맨드 창에 `./cutNmerge.sh`를 입력하여 "cut_image.py"와 "merge_image.py"를 동시에 실행할 수 있습니다.
  
## 진행 방식
### cut_image.py
- Crop된 sub image들의 위치 정보를 알 수 없도록 1에서 sub image의 총 개수 사이에서 무작위로 이름이 붙여지도록 하였습니다. (예를들어 2x2로 등분하면, sub image의 번호가 무작위로 1부터 4까지 부여됩니다.)
- Crop할 이미지 파일명이 merge_image.py에서도 사용될 수 있도록, 해당 파일명을 저장하는 "img2cut.txt" 파일을 생성하도록 하였습니다.

### merge_image.py
- 회전 및 이동에 불변하는 특징 검출기(feature detector)인 SIFT를 사용하여 원본 이미지 및 sub image들에서 keypoint와 descriptor값을 뽑아내고, 이들을 이용해 변형된 sub image들을 복구한 뒤에 merge하는 방식을 택했습니다.
- 원본 이미지와 비교하여 변형된 sub image들을 복원할 수 있도록 "img2cut.txt" 파일을 불러와 "cut_image.py"에서 사용한 이미지를 불러올 수 있도록 하였습니다.

### 마주했던 문제점 및 해결 시도
1. 2x2 크기로 분할할 때는 괜찮았지만, 3x3 이상으로 할 때는 이미지에 따라 SIFT 매치 개수가 4미만인 sub image가 있어 병합을 할 수 없는 오류
   -> 이러한 경우는 continue로 그냥 넘기기로 함
2. 3x3 이상으로 분할하고 나서 다시 병합하면 SIFT 매치 개수와 관계없이 색상이 왜곡되는 현상 발생(color distortion)
   -> 각 복원된 sub image마다 평균 값(mean)만큼 빼서 0.1에서 1 사이의 값으로 나눠서 해결하려 했지만, 오히려 잘 되던 2x2 합병에도 색상 왜곡 현상이 생기는 듯 하여 그냥 복원된 이미지 그대로 병합하도록 함. 이미지가 어떻게 잘렸느냐에 따라 3x3 병합 결과도 원본과 거의 유사하게 나올 때가 있음.
