import cv2
import cv2.aruco as aruco
import numpy as np
import os
import random
import time
import threading
from playsound import playsound

stages = [
    {"cols": 3, "rows": 2, "time_limit": 30},
    {"cols": 4, "rows": 3, "time_limit": 60},
    {"cols": 6, "rows": 4, "time_limit": 75},
    {"cols": 8, "rows": 6, "time_limit": 120},
]
current_stage = 0
start_time = time.time()
time_limit = stages[0]["time_limit"]
game_failed = False
stage_scores = []


puzzle_pieces = []
correct_positions = {}
overlay_background = None
shuffled_positions = None
clickable_regions = []
placed = {}
hold_piece = None
is_holding = False
mouse_x, mouse_y = 0, 0
free_placed = {}
force_complete = False

ar_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
params = aruco.DetectorParameters()

def play_sound_async(path):
    def loop_play():
        while True:
            playsound(path)
    if "bgm" in path:
        threading.Thread(target=loop_play, daemon=True).start()
    else:
        threading.Thread(target=playsound, args=(path,), daemon=True).start()

play_sound_async("music/bgm.mp3")


def world_to_image(H, x, y):
    pt = np.array([[x], [y], [1]], dtype=np.float32)
    img_pt = H @ pt
    img_pt /= img_pt[2]
    return int(img_pt[0].item()), int(img_pt[1].item())

def image_to_world(H, x, y):
    pt = np.array([[x], [y], [1]], dtype=np.float32)
    inv_H = np.linalg.inv(H)
    world_pt = inv_H @ pt
    world_pt /= world_pt[2]
    return float(world_pt[0].item()), float(world_pt[1].item())

def find_marker_corners(corners, ids):
    id_index = {id[0]: i for i, id in enumerate(ids)}
    if all(k in id_index for k in [0, 1, 2, 3]):
        return [
            corners[id_index[0]][0][3],
            corners[id_index[1]][0][0],
            corners[id_index[3]][0][1],
            corners[id_index[2]][0][2],
        ]
    return None


def load_stage_image(stage_index, cols, rows):
    fname = f"puzzle_stage/puzzle_stage_{stage_index}.jpg"
    if not os.path.exists(fname):
        raise FileNotFoundError(f" {fname} 이미지 파일을 찾을 수 없습니다.")
    img_array = np.fromfile(fname, dtype=np.uint8)
    full_img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return cv2.resize(full_img, (cols * 100, rows * 100))

def draw_piece_2d(canvas, piece, pos):
    x, y = pos
    h, w = piece.shape[:2]
    if x + w > canvas.shape[1] or y + h > canvas.shape[0] or x < 0 or y < 0:
        return
    canvas[y:y+h, x:x+w] = piece

def draw_piece_on_plane(frame, H, piece_img, cell_pos, cell_w, cell_h, piece_w, piece_h):
    x, y = cell_pos
    px = x * cell_w
    py = y * cell_h

    src_pts = np.array([[0, 0], [piece_img.shape[1], 0], [piece_img.shape[1], piece_img.shape[0]], [0, piece_img.shape[0]]], dtype=np.float32)
    dst_pts = np.array([
        world_to_image(H, px, py),
        world_to_image(H, px + cell_w * (piece_img.shape[1] / piece_w), py),
        world_to_image(H, px + cell_w * (piece_img.shape[1] / piece_w), py + cell_h * (piece_img.shape[0] / piece_h)),
        world_to_image(H, px, py + cell_h * (piece_img.shape[0] / piece_h))
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(piece_img, matrix, (frame.shape[1], frame.shape[0]))
    mask = cv2.warpPerspective(np.ones_like(piece_img, dtype=np.uint8) * 255, matrix, (frame.shape[1], frame.shape[0]))
    gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    frame[gray_mask > 0] = warped[gray_mask > 0]


def load_next_stage():
    global puzzle_pieces, correct_positions, overlay_background, shuffled_positions
    global clickable_regions, placed, hold_piece, is_holding, free_placed
    global cols, rows, cell_w, cell_h, piece_w, piece_h
    global start_time, time_limit, game_failed

    cols, rows = stages[current_stage]["cols"], stages[current_stage]["rows"]
    cell_w = 1.0 / cols
    cell_h = 1.0 / rows
    time_limit = stages[current_stage]["time_limit"]
#   start_time = time.time()
    game_failed = False

    full_img = load_stage_image(current_stage, cols, rows)
    piece_w, piece_h = full_img.shape[1] // cols, full_img.shape[0] // rows
    puzzle_pieces = [
        full_img[y*piece_h:(y+1)*piece_h, x*piece_w:(x+1)*piece_w]
        for y in range(rows) for x in range(cols)
    ]

    correct_positions = {(i % cols, i // cols): i for i in range(len(puzzle_pieces))}
    overlay_background = np.zeros_like(full_img)
    for y in range(rows):
        for x in range(cols):
            piece = puzzle_pieces[y * cols + x]
            blended = cv2.addWeighted(piece, 0.3, np.ones_like(piece, dtype=np.uint8) * 255, 0.7, 0)
            overlay_background[y*piece_h:(y+1)*piece_h, x*piece_w:(x+1)*piece_w] = blended

    shuffled_positions = None
    clickable_regions = []
    placed = {}
    hold_piece = None
    is_holding = False
    free_placed = {}

def check_answer():
    global stage_scores
    global current_stage, game_failed, force_complete
    if force_complete:
        stage_scores.append(max(0, time_limit - int(time.time() - start_time)))
        force_complete = False
        current_stage += 1
        if current_stage >= len(stages):
            return True
        load_next_stage()
        return False
    if game_failed:
        return False
    if len(placed) != len(correct_positions):
        return False
    if all(correct_positions.get(pos) == i for i, pos in placed.items()):
        stage_scores.append(max(0, time_limit - int(time.time() - start_time)))
        play_sound_async("music/success.mp3")
        current_stage += 1
        if current_stage >= len(stages):
            return True
        load_next_stage()
        return False
    return False

def on_mouse(event, x, y, flags, param):
    global hold_piece, is_holding, mouse_x, mouse_y, force_complete
    mouse_x, mouse_y = x, y
    H = param.get("H")

    if event == cv2.EVENT_LBUTTONDOWN:
        if x > param['width'] - 100 and y < 50:
            force_complete = True
            return

        if not is_holding:
            for i, x1, y1, x2, y2 in clickable_regions:
                if i not in placed and i not in free_placed and x1 <= x <= x2 and y1 <= y <= y2:
                    hold_piece = i
                    is_holding = True
                    play_sound_async("music/pickup.mp3")
                    break
            for i, (fx, fy) in free_placed.items():
                if fx <= x <= fx + piece_w and fy <= y <= fy + piece_h:
                    hold_piece = i
                    is_holding = True
                    del free_placed[i]
                    play_sound_async("music/pickup.mp3")
                    break
        else:
            if H is not None:
                wx, wy = image_to_world(H, x, y)
                gx, gy = int(wx // cell_w), int(wy // cell_h)
                if 0 <= gx < cols and 0 <= gy < rows and (gx, gy) not in placed.values():
                    placed[hold_piece] = (gx, gy)
                    hold_piece = None
                    is_holding = False
                    play_sound_async("music/drop.mp3")
                    return
            free_placed[hold_piece] = (x - piece_w // 2, y - piece_h // 2)
            hold_piece = None
            is_holding = False
            play_sound_async("music/drop.mp3")

    elif event == cv2.EVENT_RBUTTONDOWN and H is not None:
        wx, wy = image_to_world(H, x, y)
        gx, gy = int(wx // cell_w), int(wy // cell_h)
        for i, (px, py) in placed.items():
            if (gx, gy) == (px, py):
                del placed[i]
                break

load_next_stage()

start_screen = np.ones((600, 900, 3), dtype=np.uint8) * 255
cv2.putText(start_screen, "AR Puzzle Game", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 255), 5)
cv2.putText(start_screen, "Press any key to start", (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 3)
cv2.imshow("AR Puzzle", start_screen)
cv2.waitKey(0)

start_time = time.time()
cap = cv2.VideoCapture(1)
cv2.namedWindow("AR Puzzle")
param_dict = {"H": None, "width": 0}
cv2.setMouseCallback("AR Puzzle", on_mouse, param_dict)

while True:
    ret, cam = cap.read()
    if not ret:
        break


    height, width = cam.shape[:2]
    param_dict["width"] = width
    canvas = np.ones((height, width + 500, 3), dtype=np.uint8) * 255
    canvas[0:height, 0:width] = cam

    if shuffled_positions is None and piece_w > 0 and piece_h > 0:
        right_margin = canvas.shape[1] - width - 20
        bottom_margin = canvas.shape[0] - 10
        cols_per_row = max(1, right_margin // piece_w)
        rows_per_col = max(1, bottom_margin // piece_h)
        total_slots = cols_per_row * rows_per_col

        indices = list(range(total_slots))
        random.shuffle(indices)

        shuffled_positions = []
        for i in range(len(puzzle_pieces)):
            idx = indices[i % total_slots]
            grid_x = idx % cols_per_row
            grid_y = idx // cols_per_row
            x = width + 10 + grid_x * piece_w
            y = 10 + grid_y * piece_h
            shuffled_positions.append((x, y))
        clickable_regions = [(i, x, y, x + piece_w, y + piece_h) for i, (x, y) in enumerate(shuffled_positions)]

    gray = cv2.cvtColor(cam, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco.detectMarkers(gray, ar_dict, parameters=params)

    if ids is not None and len(ids) >= 4:
        aruco.drawDetectedMarkers(canvas, corners, ids)
        marker_corners = find_marker_corners(corners, ids)
        if marker_corners:
            world_pts = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float32)
            image_pts = np.array(marker_corners, dtype=np.float32)
            H, _ = cv2.findHomography(world_pts, image_pts)
            if np.linalg.cond(H) < 1e4:
                param_dict["H"] = H
                draw_piece_on_plane(canvas, H, overlay_background, (0, 0), cell_w, cell_h, piece_w, piece_h)
                for i, (gx, gy) in placed.items():
                    draw_piece_on_plane(canvas, H, puzzle_pieces[i], (gx, gy), cell_w, cell_h, piece_w, piece_h)
                if check_answer():
                    total_score = sum(stage_scores)
                    canvas[:,:,:] = 255
                    cv2.putText(canvas, "All Stages Complete!", (200, height // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 128, 0), 4)
                    cv2.putText(canvas, f"Your Score: {total_score}", (250, height // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                    cv2.imshow("AR Puzzle", canvas)
                    cv2.waitKey(0)
                    break

    if shuffled_positions is not None:
        for i, pos in enumerate(shuffled_positions):
            if i not in placed and i not in free_placed and (i != hold_piece or not is_holding):
                draw_piece_2d(canvas, puzzle_pieces[i], pos)

    for i, pos in free_placed.items():
        draw_piece_2d(canvas, puzzle_pieces[i], pos)

    if hold_piece is not None and is_holding:
        draw_piece_2d(canvas, puzzle_pieces[hold_piece], (mouse_x - piece_w // 2, mouse_y - piece_h // 2))

    elapsed = int(time.time() - start_time)
    remaining = max(0, time_limit - elapsed)
    cv2.putText(canvas, f"Time Left: {remaining}s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 2)

    if remaining <= 0 and not game_failed:
        game_failed = True
        play_sound_async("music/fail.mp3")

    if game_failed:
        # 실패 메시지 출력
        canvas[:, :] = 255  # 화면 전체 흰색으로
        cv2.putText(canvas, "TIME OVER!", (200, height // 2 - 30), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
        cv2.putText(canvas, "Press any key to restart", (180, height // 2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (50, 50, 50), 3)
        cv2.imshow("AR Puzzle", canvas)
        cv2.waitKey(0)

        # 스테이지 초기화 및 시작화면 복귀
        current_stage = 0
        stage_scores = []
        load_next_stage()

        # 시작화면 다시 보여주기
        start_screen = np.ones((600, 900, 3), dtype=np.uint8) * 255
        cv2.putText(start_screen, "AR Puzzle Game", (150, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 128, 255), 5)
        cv2.putText(start_screen, "Press any key to start", (180, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (100, 100, 100), 3)
        cv2.imshow("AR Puzzle", start_screen)
        cv2.waitKey(0)

        start_time = time.time()
        game_failed = False
        continue  # 다음 프레임으로 루프 재시작



    cv2.imshow("AR Puzzle", canvas)
    key = cv2.waitKey(1)
    if key == ord('s'):
        force_complete = True
        play_sound_async("music/success.mp3")
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
