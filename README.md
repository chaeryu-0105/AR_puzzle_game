# 🧩 AR Puzzle Game (Augmented Reality Puzzle Game)

AR 마커 기반으로 실시간 퍼즐 맞추기를 즐길 수 있는 Python + OpenCV 기반의 증강현실 퍼즐 게임입니다.  
카메라로 ArUco 마커를 인식해 평면상에 퍼즐 조각을 드래그 앤 드롭하여 배치하며, 각 스테이지를 시간 내에 클리어해야 합니다.

## 🎮 게임 규칙 및 기능

- **AR 퍼즐 조각 조작**: 마우스로 조각을 집고, 마커 위에 정확히 배치하세요.
- **스테이지 타이머**: 제한시간 내에 퍼즐을 모두 완성해야 다음 스테이지로 넘어갈 수 있습니다.
- **총 4개 스테이지**: 점점 어려워지는 난이도 (조각 수 증가).
- **사운드 효과**: BGM 및 조작 효과음 지원.

##필수 조건:

- ArUco 마커 4개가 인쇄된 종이 필요 (ID: 0, 1, 2, 3)

- music/ 디렉토리에 bgm.mp3, pickup.mp3, drop.mp3, fail.mp3, success.mp3 포함

- puzzle_stage/ 폴더에 퍼즐 이미지 파일 (puzzle_stage_0.jpg, puzzle_stage_1.jpg ...) 필요

##조작법
```c
좌클릭: 퍼즐 조각을 집고 놓기
우클릭: 이미 배치된 조각 제거
ESC 키 : 게임 종료
```

##디렉토리 구조
```c

├── main.py
├── music/
│   ├── bgm.mp3
│   ├── pickup.mp3
│   ├── drop.mp3
│   ├── fail.mp3
│   └── success.mp3
├── puzzle_stage/
│   ├── puzzle_stage_0.jpg
│   ├── puzzle_stage_1.jpg
│   └── puzzle_stage_2.jpg
│   └── puzzle_stage_3.jpg
```


##라이선스
본 프로젝트는 개인/학습용으로 제작되었습니다.
사용된 외부 라이브러리는 각각의 라이선스를 따릅니다:
```c
OpenCV (Apache 2.0)
numpy (BSD)
playsound (MIT)
```