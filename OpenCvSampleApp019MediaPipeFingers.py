# В предыдущих разделах мы рассмотрели, как распознавать жесты руки на основе расстояний между ключевыми точками. Однако
# этот метод может быть недостаточно точным для определения сгиба каждого пальца. В этом разделе мы узнаем, как
# определить, согнут ли палец, используя вычисление углов между сегментами пальцев, а также рассмотрим проект робота,
# который может быть реализован при помощи такой системы управления.
#
# Принцип работы
# Каждый палец состоит из нескольких суставов и фаланг, которые можно представить как точки и отрезки между ними. Чтобы
# определить, согнут ли палец, мы можем вычислить угол между двумя векторами:
#
# Вектор 1 (v1): от основания пальца до его среднего сустава.
# Вектор 2 (v2): от среднего сустава до кончика пальца.
# Если угол между этими векторами меньше определённого порогового значения, палец считается согнутым.
#
# Вычисление угла между векторами
# Угол между двумя векторами в трёхмерном пространстве можно вычислить с помощью скалярного произведения:

# Где:
#
# v
# 1
# ⃗
# v1
#   и
# v
# 2
# ⃗
# v2
#   — векторы.
# ∣
# v
# 1
# ⃗
# ∣
# ∣
# v1
#  ∣ и
# ∣
# v
# 2
# ⃗
# ∣
# ∣
# v2
#  ∣ — длины (нормы) векторов.
# θ
# θ — угол между векторами.
# Угол
# θ
# θ выражается в радианах, но для удобства можно перевести его в градусы.
#
# import cv2
# import mediapipe as mp
# import numpy as np
# import tkinter as tk
#
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
#
# cap = cv2.VideoCapture(0)
#
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1270)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#
# tip_ids = [4, 8, 12, 16, 20]
# base_ids = [0, 5, 9, 13, 17]
# joint_ids = [3, 6, 10, 14, 18]
#
# # Пороговые значения углов для пальцев
# thumb_bend_threshold = 40
# finger_bend_threshold = 50
#
#
# def get_angle(v1, v2):
#     v1 = np.array(v1)
#     v2 = np.array(v2)
#     dot_product = np.dot(v1, v2)
#     norm_v1 = np.linalg.norm(v1)
#     norm_v2 = np.linalg.norm(v2)
#     cosine_angle = dot_product / (norm_v1 * norm_v2)
#     angle = np.arccos(cosine_angle)
#     return np.degrees(angle)
#
#
# def is_finger_bent(base, joint, tip, is_thumb=False):
#     v1 = [joint.x - base.x, joint.y - base.y, joint.z - base.z]
#     v2 = [tip.x - joint.x, tip.y - joint.y, tip.z - joint.z]
#     angle = get_angle(v1, v2)
#     if is_thumb:
#         return angle >= thumb_bend_threshold
#     else:
#         return angle >= finger_bend_threshold
#
#
# while True:
#     reed_ok, frame = cap.read()
#     if not reed_ok:
#         continue
#     frame = cv2.flip(frame, 1)
#
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame)
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#             landmarks = hand_landmarks.landmark
#
#             for id, landmark in enumerate(landmarks):
#                 h, w, c = frame.shape
#                 cx, cy = int(landmark.x * w), int(landmark.y * h)
#                 cv2.putText(frame, str(id), (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
#
#             for finger_index, tip_id in enumerate(tip_ids):
#                 base_id = base_ids[finger_index]
#                 joint_id = joint_ids[finger_index]
#                 is_thumb = (finger_index == 0)
#                 if is_finger_bent(landmarks[base_id], landmarks[joint_id], landmarks[tip_id], is_thumb):
#                     cx, cy = int(landmarks[tip_id].x * frame.shape[1]), int(landmarks[tip_id].y * frame.shape[0])
#                     cv2.circle(frame, (cx, cy), 7, (0, 0, 255), cv2.FILLED)
#                 else:
#                     cx, cy = int(landmarks[tip_id].x * frame.shape[1]), int(landmarks[tip_id].y * frame.shape[0])
#                     cv2.circle(frame, (cx, cy), 7, (0, 255, 0), cv2.FILLED)
#
#     cv2.imshow('Fingers', frame)
#
#     if cv2.waitKey(10) == 27:
#         break
#
# cap.release()
# cv2.destroyAllWindows()
#
# Основные моменты представленного кода разберем подробнее:
# Определение индексов ключевых точек пальцев:
# tip_ids = [4, 8, 12, 16, 20]    # Кончики пальцев
# base_ids = [0, 5, 9, 13, 17]    # Основания пальцев
# joint_ids = [3, 6, 10, 14, 18]  # Средние суставы
# Пороговые значения углов для определения сгиба пальцев:
# thumb_bend_threshold = 40
# finger_bend_threshold = 50
# thumb_bend_threshold: Пороговый угол для большого пальца.
# finger_bend_threshold: Пороговый угол для остальных пальцев.
# Функция для вычисления угла между двумя векторами:
# def get_angle(v1, v2):
#     v1 = np.array(v1)
#     v2 = np.array(v2)
#     dot_product = np.dot(v1, v2)
#     norm_v1 = np.linalg.norm(v1)
#     norm_v2 = np.linalg.norm(v2)
#     cosine_angle = dot_product / (norm_v1 * norm_v2)
#     angle = np.arccos(cosine_angle)
#     return np.degrees(angle)
# Преобразуем входные списки координат v1 и v2 в массивы NumPy для удобства математических операций с помощью NumPy
# Вычисление скалярного произведения векторов:
# dot_product = np.dot(v1, v2)
# Скалярное произведение (dot product) двух векторов вычисляется как сумма произведений соответствующих компонентов:

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Инициализация утилит для рисования
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Шаг 2: Создаем объект GestureRecognizer для распознавания жестов
base_options = python.BaseOptions(model_asset_path='gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

# Захватываем видео с камеры
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Отражаем кадр по горизонтали для естественной (зеркальной) визуализации
    frame = cv2.flip(frame, 1)

    # Конвертируем изображение из BGR в RGB перед обработкой
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Создаем объект mp.Image для распознавания
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

    # Шаг 4: Распознаем жесты на изображении
    recognition_result = recognizer.recognize(mp_image)

    # Копируем кадр для нанесения надписей
    annotated_image = frame.copy()

    # Рисуем ключевые точки руки встроенными средствами
    if recognition_result.hand_landmarks:
        for hand_landmarks in recognition_result.hand_landmarks:
            # Преобразуем список ключевых точек в объект NormalizedLandmarkList
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z)
                for landmark in hand_landmarks
            ])
            # Рисуем ключевые точки и соединения на изображении
            mp_drawing.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # Отображаем названия жестов
    if recognition_result.gestures:
        for i, hand_gestures in enumerate(recognition_result.gestures):
            if hand_gestures:
                # Получаем жест с наибольшей вероятностью
                gesture = hand_gestures[0]
                gesture_name = gesture.category_name
                # Словарь переводов жестов на русский язык
                gesture_translations = {
                    "None": "Нет",
                    "Closed_Fist": "Кулак",
                    "Open_Palm": "Открытая ладонь",
                    "Pointing_Up": "Указательный вверх",
                    "Thumb_Down": "Дизлайк",
                    "Thumb_Up": "Лайк",
                    "Victory": "Победа",
                    "ILoveYou": "Я тебя люблю"
                }
                # Получаем перевод названия жеста на русский язык
                russian_gesture_name = gesture_translations.get(gesture_name, gesture_name)
                # Выводим название жеста на изображении
                cv2.putText(annotated_image, f'Жест: {russian_gesture_name}', (10, 70 + i * 30),
                            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Отображаем изображение
    cv2.imshow('Gesture Recognition', annotated_image)
    if cv2.waitKey(20) == 27:
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
# Пояснения к коду
# Главный цикл обработки видео
# Читаем кадр с камеры.
# Отражаем кадр по горизонтали для зеркального отображения.
# Конвертируем изображение в формат RGB для обработки моделью.
# Создаем объект mp.Image для распознавания жестов.
# Распознаем жесты на текущем кадре.
# Обработка результатов распознавания
# Рисуем виртуальный квадрат.
# Проверяем наличие рук.
# Обрабатываем обнаруженную руку:
# Преобразуем ключевые точки в пиксельные координаты.
# Вычисляем центр руки как среднее значение координат ключевых точек.
# Рисуем центр руки на изображении.
# Отображаем распознанный жест:
# Используем предобученную модель для определения жеста.
# Переводим название жеста на русский язык для отображения.
# Проверяем, распознан ли жест "Кулак".
# Если да и центр руки находится внутри квадрата, активируем режим перетаскивания.
# Запоминаем смещение между центром квадрата и центром руки для плавного перемещения.
# Обновляем позицию квадрата, если is_dragging == True.
# Обновляем координаты квадрата так, чтобы его центр совпадал с центром руки с учетом смещения.
# Ограничиваем квадрат рамками окна.
# Убедитесь, что файл модели gesture_recognizer.task находится в той же директории, что и скрипт.
# Изменение жеста активации: Если вы хотите использовать другой жест для активации перетаскивания (например, "Лайк"), измените проверку if russian_gesture_name == "Кулак" на нужный жест.
# Плавность движения: Для более плавного перемещения квадрата можно использовать фильтрацию координат или предсказание на основе предыдущих положений.
# Используя предобученную модель MediaPipe Gesture Recognizer, мы значительно упростили процесс распознавания жестов и повысили точность приложения. Теперь наше интерактивное приложение может использовать более сложные и точные жесты для управления виртуальными элементами.
#
# Данный подход открывает возможности для разработки более сложных приложений, таких как управление интерфейсами, игры с использованием жестов и другие интерактивные системы.
# # Unable to open file at D:\Project_Work\000_ROBOT_ROS_PYTHON\CoursePython\ProjetEdu\pythonProject\gesture_recognizer.task