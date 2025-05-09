# Полный пример системы авторизации на основе PIN-кода
# В данном проекте разработана система авторизации с использованием жестов рук, основанная на анализе положения
# и движения пальцев с помощью технологий компьютерного зрения, описанных ранее в этом курсе.
# В качестве основного инструмента для обработки видео и анализа жестов используется библиотека
# MediaPipe, которая позволяет точно отслеживать положение ключевых точек на руке пользователя.
#
# Как это работает
# Обнаружение руки и пальцев: В реальном времени камера отслеживает положение руки и каждого пальца. Используя MediaPipe
# Hands, система определяет ключевые точки на руке, такие как кончики пальцев, суставы и основание пальцев.
#
# Обработка жестов: Для каждого пальца рассчитывается угол между суставами, что позволяет определить, насколько
# сильно он согнут. Например, система может различать, согнут ли палец или находится ли он в прямом положении.
# Это используется для создания жестов, которые будут служить в качестве «кодов» или команд.
#
# Механизм ввода: Пользователь может вводить цифры, сложив или распрямляя пальцы, что эквивалентно вводу данных
# на сенсорной панели. Когда на экране появляется кнопка для ввода кода, система регистрирует количество распрямленных пальцев и запоминает этот ввод как часть кодовой последовательности.
#
# Режимы взаимодействия: В системе предусмотрены различные режимы работы, такие как "ввод кода" и "ожидание",
# которые активируются при помощи интерактивной кнопки. В случае неверного ввода кода появляется сообщение об ошибке, а при правильном вводе — подтверждение.
#
# Сферы применения
# Интерактивные терминалы: Технология может быть использована в бесконтактных терминалах для авторизации пользователей
# без необходимости физического ввода пароля или использования биометрических данных, таких как отпечатки пальцев.
#
# Компьютерные игры: Использование жестов для управления персонажами или навигации по интерфейсу игры.
# Жесты могут быть использованы как способ аутентификации перед доступом к игровому контенту или для использования особых
# игровых возможностей.
#
# Индустриальные приложения: В сферах, где руки должны оставаться чистыми или без перчаток, например,
# в медицине или производственных линиях, эта система может использоваться для контроля доступа в определенные
# зоны без необходимости касания физических кнопок.
#
# Системы умного дома: Встраивание жестовой аутентификации в системы управления умным домом позволяет
# пользователю безошибочно взаимодействовать с различными устройствами, такими как освещение, двери или системы безопасности.
#
# Преимущества системы
# Бесконтактность: Устройство не требует физического контакта с экраном или устройством, что снижает риск
# распространения инфекций и повышает удобство использования.
# Интуитивность: Система простая в использовании и позволяет людям вводить код с помощью естественных движений.
# Технологичность: Использование жестов в качестве метода авторизации делает систему более передовой и уникальной.
# Безопасность: Благодаря уникальным жестам, сгенерированным индивидуальными движениями пользователя, можно повысить
# безопасность и уменьшить вероятность взлома.

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

# Прочитаем один кадр, чтобы получить размеры фрейма
reed_ok, frame = cap.read()
if not reed_ok:
    print("Failed to read frame from camera")
    cap.release()
    exit(1)

frame = cv2.flip(frame, 1)
frame_height, frame_width = frame.shape[:2]

tip_ids = [4, 8, 12, 16, 20]
base_ids = [0, 5, 9, 13, 17]
joint_ids = [3, 6, 10, 14, 18]

# Пороговые значения углов для пальцев
thumb_bend_threshold = 30
finger_bend_threshold = 40


def get_angle(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)
    dot_product = np.dot(v1, v2)
    norm_v1 = np.linalg.norm(v1)
    norm_v2 = np.linalg.norm(v2)
    cosine_angle = dot_product / (norm_v1 * norm_v2)
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)


def is_finger_bent(base, joint, tip, is_thumb=False):
    v1 = [joint.x - base.x, joint.y - base.y, joint.z - base.z]
    v2 = [tip.x - joint.x, tip.y - joint.y, tip.z - joint.z]
    angle = get_angle(v1, v2)
    if is_thumb:
        return angle >= thumb_bend_threshold
    else:
        return angle >= finger_bend_threshold


# Инициализация переменных для функциональности кнопок
is_in_code_entry_mode = False
timer_started = False
timer_start_time = None
last_hand_detected_time = time.time()

# Свойства кнопки "Ввести код"/"ВЫХОД"
rect_width = 230
rect_height = 100
rect_x = frame_width - rect_width - 10  # Перемещаем в правый верхний угол
rect_y = 10

# Свойства кнопки стереть
delete_button_width = 100
delete_button_height = 100
delete_button_x = rect_x + (rect_width - delete_button_width) // 2  # Центрируем под основной кнопкой
delete_button_y = rect_y + rect_height + 10
delete_timer_started = False
delete_timer_start_time = None
is_delete_button_active = False

# Переменные для режима ввода кода
code_digits = []
current_digit_index = 0  # от 0 до 3
finger_count_timer_started = False
finger_count_timer_start_time = None
current_finger_count = None

correct_code = [1, 2, 3, 4]  # Укажите правильный код
display_message = None  # Сообщение для отображения ("КОД ВЕРНЫЙ" или "Доступ запрещен")
message_color = None  # Цвет сообщения (зеленый или красный)
message_timer_start = None  # Время начала отображения сообщения

while True:
    reed_ok, frame = cap.read()
    if not reed_ok:
        continue
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    index_finger_tip = None
    index_finger_straight = False
    straight_fingers = 0  # Инициализируем количество разогнутых пальцев

    if results.multi_hand_landmarks:
        last_hand_detected_time = time.time()  # Обновляем время, когда рука была обнаружена
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = hand_landmarks.landmark

            # Считаем количество разогнутых пальцев
            straight_fingers = 0

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

                # Проверяем указательный палец
                if finger_index == 1:
                    index_finger_tip = (cx, cy)
                    index_finger_straight = not is_bent
    else:
        if is_in_code_entry_mode and time.time() - last_hand_detected_time > 10:
            # Если прошло более 10 секунд, выходим в стартовый режим
            print("No hand detected for 10 seconds. Exiting code entry mode.")
            is_in_code_entry_mode = False
            code_digits = []
            current_digit_index = 0
            finger_count_timer_started = False
            finger_count_timer_start_time = None
            current_finger_count = None

    # Отображаем количество пальцев в нижнем правом углу
    cv2.putText(frame, f'{straight_fingers}', (frame.shape[1] - 100, frame.shape[0] - 50), cv2.FONT_HERSHEY_COMPLEX, 2,
                (255, 255, 255), 3)

    # Устанавливаем метку кнопки в зависимости от режима
    if is_in_code_entry_mode:
        rect_label = 'ВЫХОД'
    else:
        rect_label = 'Ввести код'

    # Определяем цвет кнопки на основе таймера
    if timer_started:
        elapsed_time = time.time() - timer_start_time
        color_fraction = min(elapsed_time / 2, 1)
        start_color = np.array([255, 0, 0], dtype=np.float32)  # Синий
        end_color = np.array([0, 255, 0], dtype=np.float32)  # Зеленый
        current_color = start_color + (end_color - start_color) * color_fraction
        current_color = tuple(current_color.astype(int).tolist())

        if elapsed_time >= 2:
            is_in_code_entry_mode = not is_in_code_entry_mode
            timer_started = False
            timer_start_time = None
            if not is_in_code_entry_mode:
                code_digits = []
                current_digit_index = 0
                finger_count_timer_started = False
                finger_count_timer_start_time = None
                current_finger_count = None
    else:
        current_color = (255, 0, 0)  # Синий

    # Рисуем кнопку с округленными углами
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

    # Отображаем метку на кнопке
    font = cv2.FONT_HERSHEY_COMPLEX
    text_size = cv2.getTextSize(rect_label, font, 1, 2)[0]
    text_x = rect_x + (rect_width - text_size[0]) // 2
    text_y = rect_y + (rect_height + text_size[1]) // 2
    cv2.putText(frame, rect_label, (text_x, text_y), font, 1, (255, 255, 255), 2)

    # Проверяем, находится ли указательный палец на кнопке
    if index_finger_tip is not None and index_finger_straight:
        ix, iy = index_finger_tip
        if rect_x <= ix <= rect_x + rect_width and rect_y <= iy <= rect_y + rect_height:
            if not timer_started:
                timer_started = True
                timer_start_time = time.time()
        else:
            timer_started = False
            timer_start_time = None
    else:
        timer_started = False
        timer_start_time = None

    # Если в режиме ввода кода, рисуем окружности и обрабатываем ввод
    if is_in_code_entry_mode:
        # Рисуем 4 окружности наверху
        circle_radius = 30
        circle_spacing = 20
        circles_x_positions = []
        circles_y_position = 150  # Настройте по необходимости
        frame_width = frame.shape[1]
        total_width = 4 * circle_radius * 2 + 3 * circle_spacing
        start_x = (frame_width - total_width) // 2 + circle_radius

        for i in range(4):
            x = start_x + i * (circle_radius * 2 + circle_spacing)
            circles_x_positions.append(x)

        # Логика ввода цифр
        if straight_fingers > 0 and not is_delete_button_active:
            if current_finger_count is None or current_finger_count != straight_fingers:
                current_finger_count = straight_fingers
                finger_count_timer_started = True
                finger_count_timer_start_time = time.time()
            else:
                if finger_count_timer_started:
                    elapsed_time = time.time() - finger_count_timer_start_time
                    if elapsed_time >= 2.0:
                        code_digits.append(current_finger_count)
                        current_digit_index += 1
                        finger_count_timer_started = False
                        finger_count_timer_start_time = None
                        current_finger_count = None
                        if len(code_digits) >= 4:
                            print("Code entered:", code_digits)
                            if code_digits == correct_code:
                                display_message = "КОД ВЕРНЫЙ"
                                message_color = (0, 255, 0)  # Зеленый
                            else:
                                display_message = "Доступ запрещен"
                                message_color = (0, 0, 255)  # Красный
                            code_digits = []
                            current_digit_index = 0
                            message_timer_start = time.time()  # Запускаем таймер для сообщения
                            code_digits = []  # Очищаем код
                            current_digit_index = 0
                            is_in_code_entry_mode = False  # Выходим из режима ввода
        else:
            current_finger_count = None
            finger_count_timer_started = False
            finger_count_timer_start_time = None

        # Рисуем окружности
        for i, x in enumerate(circles_x_positions):
            center = (int(x), circles_y_position)
            if i < len(code_digits):
                cv2.circle(frame, center, circle_radius, (0, 255, 0), -1)
                digit = code_digits[i]
                font = cv2.FONT_HERSHEY_SIMPLEX
                text_size = cv2.getTextSize(str(digit), font, 1, 2)[0]
                text_x = int(x) - text_size[0] // 2
                text_y = circles_y_position + text_size[1] // 2
                cv2.putText(frame, str(digit), (text_x, text_y), font, 1, (255, 255, 255), 2)
            elif i == current_digit_index:
                cv2.circle(frame, center, circle_radius, (255, 0, 0), 2)
                if finger_count_timer_started:
                    elapsed_time = time.time() - finger_count_timer_start_time
                    fraction = min(elapsed_time / 2.0, 1.0)
                    angle = int(360 * fraction)
                    axes = (circle_radius, circle_radius)
                    cv2.ellipse(frame, center, axes, -90, 0, angle, (0, 255, 0), 5)
                    cv2.ellipse(frame, center, axes, -90, angle, 360, (255, 0, 0), 2)
                else:
                    cv2.circle(frame, center, circle_radius, (255, 0, 0), 2)
            else:
                cv2.circle(frame, center, circle_radius, (255, 0, 0), 2)

        # Рисуем кнопку стереть под основной кнопкой
        delete_button_rect = np.array([
            [delete_button_x + 10, delete_button_y],
            [delete_button_x + delete_button_width - 10, delete_button_y],
            [delete_button_x + delete_button_width, delete_button_y + 10],
            [delete_button_x + delete_button_width, delete_button_y + delete_button_height - 10],
            [delete_button_x + delete_button_width - 10, delete_button_y + delete_button_height],
            [delete_button_x + 10, delete_button_y + delete_button_height],
            [delete_button_x, delete_button_y + delete_button_height - 10],
            [delete_button_x, delete_button_y + 10]
        ], np.int32)

        # Определяем цвет кнопки стереть на основе таймера
        if delete_timer_started:
            elapsed_time = time.time() - delete_timer_start_time
            color_fraction = min(elapsed_time / 2, 1)
            start_color = np.array([255, 0, 0], dtype=np.float32)  # Синий
            end_color = np.array([0, 255, 0], dtype=np.float32)  # Зеленый
            delete_button_color = start_color + (end_color - start_color) * color_fraction
            delete_button_color = tuple(delete_button_color.astype(int).tolist())

            if elapsed_time >= 2:
                code_digits = []
                current_digit_index = 0
                delete_timer_started = False
                delete_timer_start_time = None
        else:
            delete_button_color = (255, 0, 0)  # Синий

        cv2.fillPoly(frame, [delete_button_rect], delete_button_color)
        cv2.polylines(frame, [delete_button_rect], True, (255, 255, 255), 2)

        # Отображаем метку на кнопке стереть
        delete_label = '<-'
        font = cv2.FONT_HERSHEY_COMPLEX
        text_size = cv2.getTextSize(delete_label, font, 2, 3)[0]
        text_x = delete_button_x + (delete_button_width - text_size[0]) // 2
        text_y = delete_button_y + (delete_button_height + text_size[1]) // 2
        cv2.putText(frame, delete_label, (text_x, text_y), font, 2, (255, 255, 255), 3)

        # Проверяем, находится ли указательный палец на кнопке стереть
        if index_finger_tip is not None and index_finger_straight:
            ix, iy = index_finger_tip
            if delete_button_x <= ix <= delete_button_x + delete_button_width and \
                    delete_button_y <= iy <= delete_button_y + delete_button_height:
                if not delete_timer_started:
                    delete_timer_started = True
                    delete_timer_start_time = time.time()
                    is_delete_button_active = True  # Активируем флаг
            else:
                delete_timer_started = False
                delete_timer_start_time = None
                is_delete_button_active = False  # Сбрасываем флаг
        else:
            delete_timer_started = False
            delete_timer_start_time = None
            is_delete_button_active = False  # Сбрасываем флаг

    if display_message is not None:
        elapsed_time = time.time() - message_timer_start
        if elapsed_time <= 3:  # Отображаем сообщение в течение 3 секунд
            text_size = cv2.getTextSize(display_message, font, 2, 3)[0]
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = (frame.shape[0] + text_size[1]) // 2
            cv2.putText(frame, display_message, (text_x, text_y), font, 2, message_color, 3)
        else:
            display_message = None  # Сбрасываем сообщение после 2 секунд

    cv2.imshow('Fingers', frame)

    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows()