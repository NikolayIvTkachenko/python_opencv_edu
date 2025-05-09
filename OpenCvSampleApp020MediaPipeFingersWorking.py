import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1270)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

tip_ids = [4, 8, 12, 16, 20]
base_ids = [0, 5, 9, 13, 17]
joint_ids = [3, 6, 10, 14, 18]

# Пороговые значения для определения согнутых пальцев
thumb_bend_threshold = 40
finger_bend_threshold = 50

# Функция вычисления угла между двумя векторами
def get_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Функция определения согнут ли палец
def is_finger_bent(base, joint, tip, is_thumb=False):
    v1 = [joint.x - base.x, joint.y - base.y, joint.z - base.z]
    v2 = [tip.x - joint.x, tip.y - joint.y, tip.z - joint.z]
    angle = get_angle(v1, v2)
    if is_thumb:
        return angle >= thumb_bend_threshold
    else:
        return angle >= finger_bend_threshold

# Переменные для работы с кнопкой
is_in_code_entry_mode = False  # Текущий режим (ввод кода или стартовый)
timer_started = False  # Таймер удержания пальца
timer_start_time = None  # Время начала удержания

# Координаты и размеры кнопки
rect_x = 10
rect_y = 10
rect_width = 300  # Увеличенный размер кнопки
rect_height = 150

while True:
    reed_ok, frame = cap.read()
    if not reed_ok:
        continue
    frame = cv2.flip(frame, 1)

    # Конвертируем изображение в RGB для работы с Mediapipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    index_finger_tip = None
    index_finger_straight = False

    if results.multi_hand_landmarks:
        # Рисуем найденные руки и обрабатываем координаты
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            straight_fingers = 0  # Количество прямых пальцев

            for finger_index, tip_id in enumerate(tip_ids):
                base_id = base_ids[finger_index]
                joint_id = joint_ids[finger_index]
                is_thumb = (finger_index == 0)
                is_bent = is_finger_bent(landmarks[base_id], landmarks[joint_id], landmarks[tip_id], is_thumb)
                cx, cy = int(landmarks[tip_id].x * frame.shape[1]), int(landmarks[tip_id].y * frame.shape[0])
                if is_bent:
                    cv2.circle(frame, (cx, cy), 7, (0, 0, 255), cv2.FILLED)
                else:
                    cv2.circle(frame, (cx, cy), 7, (0, 255, 0), cv2.FILLED)
                    straight_fingers += 1

                if finger_index == 1:  # Проверяем указательный палец
                    index_finger_tip = (cx, cy)
                    index_finger_straight = not is_bent

            # Выводим количество прямых пальцев
            cv2.putText(frame, f'{straight_fingers}', (frame.shape[1] - 100, 50), cv2.FONT_HERSHEY_COMPLEX, 2,
                        (255, 255, 255), 3)

    # Устанавливаем текст кнопки в зависимости от текущего режима
    if is_in_code_entry_mode:
        rect_label = 'ВЫХОД'
    else:
        rect_label = 'Ввести код'

    # Определяем цвет кнопки в зависимости от таймера
    if timer_started:
        elapsed_time = time.time() - timer_start_time
        # Ограничиваем цветовой интервал на 2 секундах
        color_fraction = min(elapsed_time / 2, 1)
        # Вычисляем текущий цвет
        start_color = np.array([255, 0, 0], dtype=np.float32)  # Синий
        end_color = np.array([0, 255, 0], dtype=np.float32)    # Зеленый
        current_color = start_color + (end_color - start_color) * color_fraction
        current_color = tuple(current_color.astype(int).tolist())

        # Если прошло 2 секунды, переключаем режим
        if elapsed_time >= 2:
            is_in_code_entry_mode = not is_in_code_entry_mode
            timer_started = False
            timer_start_time = None
    else:
        current_color = (255, 0, 0)  # Синий цвет по умолчанию

    # Рисуем кнопку с закругленными углами
    button_rect = np.array([
        [rect_x + 10, rect_y],
        [rect_x + rect_width - 10, rect_y],
        [rect_x + rect_width, rect_y + 10],
        [rect_x + rect_width, rect_y + rect_height - 10],
        [rect_x + rect_width - 10, rect_y + rect_height],
        [rect_x + 10, rect_y + rect_height],
        [rect_x, rect_y + rect_height - 10],
        [rect_x, rect_y + 10]
    ], np.int32)

    cv2.fillPoly(frame, [button_rect], current_color)
    cv2.polylines(frame, [button_rect], True, (255, 255, 255), 2)

    # Отображаем текст на кнопке
    font = cv2.FONT_HERSHEY_COMPLEX
    text_size = cv2.getTextSize(rect_label, font, 1, 2)[0]
    text_x = rect_x + (rect_width - text_size[0]) // 2
    text_y = rect_y + (rect_height + text_size[1]) // 2
    cv2.putText(frame, rect_label, (text_x, text_y), font, 1, (255, 255, 255), 2)

    # Проверяем, находится ли указательный палец в пределах кнопки
    if index_finger_tip is not None and index_finger_straight:
        ix, iy = index_finger_tip
        if rect_x <= ix <= rect_x + rect_width and rect_y <= iy <= rect_y + rect_height:
            # Если палец внутри кнопки, запускаем таймер
            if not timer_started:
                timer_started = True
                timer_start_time = time.time()
        else:
            # Если палец вышел из зоны кнопки, сбрасываем таймер
            timer_started = False
            timer_start_time = None
    else:
        # Если палец не прямой или не найден, сбрасываем таймер
        timer_started = False
        timer_start_time = None

    cv2.imshow('Fingers', frame)

    if cv2.waitKey(10) == 27:  # Нажатие ESC для выхода
        break

cap.release()
cv2.destroyAllWindows()