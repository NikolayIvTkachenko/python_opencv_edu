# Нормализация расстояний для независимости от удаления от камеры
# В текущем коде жесты распознаются на основе абсолютных расстояний между точками. Это
# работает, если рука находится на фиксированном расстоянии от камеры. Однако, если пользователь отодвинет или
# приблизит руку, абсолютные значения расстояний изменятся, и программа может неверно классифицировать жесты.
# Для решения этой проблемы можно использовать нормализацию расстояний на основе эталонного расстояния между фиксированными точками, например, точками
# 0
# 0 (запястье) и
# 1
# 1 (основание большого пальца). Это расстояние используется как масштабный коэффициент, относительно которого нормализуются все остальные расстояния.
#
# Принцип работы нормализации
# Вычисляем эталонное расстояние между точками
# 0
# 0 и
# 1
# 1 (запястье и основание большого пальца).
# Для всех остальных расстояний делим их абсолютное значение на это эталонное расстояние.
# Используем нормализованные расстояния для классификации жестов.
# Формула нормализации:
# d actual— вычисленное абсолютное расстояние между двумя точками.
# d base
# d base — эталонное расстояние (например, между точками  0 и 1).
# Как изменить код?
# 1. Добавить вычисление эталонного расстояния
# В коде добавим вычисление расстояния между точками 0 и 1 в качестве эталонного:
#
# base_distance = calculate_distance(landmarks[0], landmarks[1])
# 2. Нормализовать все расстояния
# Для всех расстояний из словаря distances делим их значения на base_distance:
#
# distances_normalized = {
#     pair_name: distance / base_distance
#     for pair_name, distance in distances.items()
# }

# 3. Изменить логику классификации жестов
# Обновим функцию classify_gesture для работы с нормализованными расстояниями:
#
# def classify_gesture(distances):
#     """Классифицирует жест на основе нормализованных расстояний."""
#     thumb_to_index = distances["pair_0"]  # Нормализованное расстояние большого пальца и указательного
#     thumb_to_wrist = distances["pair_3"]  # Нормализованное расстояние большого пальца и запястья
#     pinky_to_wrist = distances["pair_2"]  # Нормализованное расстояние мизинца и запястья
#
#     # Раскрытая ладонь
#     if (
#         thumb_to_index > 1.5 and  # Большой палец далеко от указательного
#         thumb_to_wrist > 2.5 and  # Большой палец далеко от запястья
#         pinky_to_wrist > 2.5  # Мизинец далеко от запястья
#     ):
#         return "Открытая ладонь"
#
#     return "Неизвестный жест"
import cv2
import numpy as np
import mediapipe as mp
import math

# Инициализация MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Захват видеопотока
cap = cv2.VideoCapture(0)

def calculate_distance(point1, point2):
    """Вычисляет евклидово расстояние между двумя точками."""
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def classify_gesture(distances):
    """Классифицирует жест на основе нормализованных расстояний."""
    thumb_to_index = distances["pair_0"]
    thumb_to_wrist = distances["pair_3"]
    pinky_to_wrist = distances["pair_2"]

    # Раскрытая ладонь
    if (
        thumb_to_index > 1.5 and
        thumb_to_wrist > 2.5 and
        pinky_to_wrist > 2.5
    ):
        return "Открытая ладонь"

    return "Неизвестный жест"

while True:
    read_ok, frame = cap.read()
    if not read_ok:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            h, w, c = frame.shape

            # Преобразуем ключевые точки в список координат
            landmarks = [
                (int(landmark.x * w), int(landmark.y * h)) for landmark in hand_landmark.landmark
            ]

            # Определяем эталонное расстояние
            base_distance = calculate_distance(landmarks[0], landmarks[1])

            # Определение пар точек для анализа расстояний
            pair_indices = [
                (4, 8),  # Большой палец и указательный
                (0, 12),  # Запястье и средний палец
                (20, 0),  # Мизинец и запястье
                (0, 4)  # Запястье и большой палец
            ]

            # Генерация пар точек
            pairs = {f"pair_{i}": (landmarks[p1], landmarks[p2]) for i, (p1, p2) in enumerate(pair_indices)}

            # Вычисляем расстояния для каждой пары
            distances = {
                pair_name: calculate_distance(pair[0], pair[1])
                for pair_name, pair in pairs.items()
            }

            # Нормализуем расстояния
            distances_normalized = {
                pair_name: distance / base_distance
                for pair_name, distance in distances.items()
            }

            # Классифицируем жест
            gesture = classify_gesture(distances_normalized)

            # Визуализация текста
            cv2.putText(frame, f"Жест: {gesture}", (10, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

            # Визуализация расстояний
            for pair_name, (point1, point2) in pairs.items():
                cv2.line(frame, point1, point2, (255, 0, 0), 2)  # Линия между точками
                cv2.circle(frame, point1, 5, (0, 255, 0), -1)  # Первая точка
                cv2.circle(frame, point2, 5, (0, 255, 0), -1)  # Вторая точка
                midpoint = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
                cv2.putText(frame, f"{distances_normalized[pair_name]:.2f}", midpoint, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Hand Gesture Detection", frame)

    # Выход по клавише Esc
    if cv2.waitKey(30) == 27:
        break

cap.release()
cv2.destroyAllWindows()