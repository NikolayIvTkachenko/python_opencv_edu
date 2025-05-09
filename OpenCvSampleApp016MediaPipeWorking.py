import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

cap = cv2.VideoCapture(0)

while True:
    read_ok, frame = cap.read()
    if not read_ok:
        break
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if results.multi_hand_landmarks is not None:
        for hand_landmark, hand_nandedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            for point_id, landmark in enumerate(hand_landmark.landmark):
                h, w, c = frame.shape
                point_x, point_y = int(landmark.x * w), int(landmark.y * h)
                cv2.circle(frame, (point_x, point_y), 2, (0, 255, 0), -1)
                cv2.putText(frame, str(point_id), (point_x, point_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
                label = hand_nandedness.classification[0].label
                cv2.putText(frame, str(label), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow("camera", frame)
    k = cv2.waitKey(30)
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

# Распознавание ключевых точек руки с использованием MediaPipe
# Распознавание и отслеживание рук в реальном времени — одна из популярных задач в области компьютерного
# зрения. Это позволяет создавать интерактивные приложения, такие как жестовое управление, дополненная реальность
# и многое другое. В этом уроке мы рассмотрим, как использовать библиотеку MediaPipe для детекции и отслеживания
# ключевых точек руки на изображении и в видеопотоке с камеры.
# Шаг за шагом: Распознавание ключевых точек руки в реальном времени
# 1. Импорт необходимых библиотек
# import cv2
# import numpy as np
# import mediapipe as mp
# 2. Инициализация моделей и утилит MediaPipe
#
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp.solutions.hands: Модуль для детекции и отслеживания рук.
# mp.solutions.drawing_utils: Утилиты для рисования аннотаций на изображениях.
#  3. Создание объекта распознавания рук
#
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
# static_image_mode=False: Режим обработки видеопотока (динамические изображения). Если установить в True,
# модель будет считать каждый кадр независимым изображением.
# max_num_hands=1: Максимальное количество распознаваемых рук в кадре.
# min_detection_confidence: Минимальная уверенность для обнаружения руки. Параметр не обязательный.
# min_tracking_confidence: Минимальная уверенность для отслеживания руки. Параметр не обязательный.
# 4. Захват видеопотока с камеры
#
# cap = cv2.VideoCapture(0)
# cv2.VideoCapture(0): Захват видеопотока с камеры. Индекс 0 обычно соответствует встроенной веб-камере.
#
# 5. Основной цикл обработки видеопотока
#
# while True:
#     read_ok, frame = cap.read()
#     if not read_ok:
#         break
#     # Остальной код внутри цикла
# Читаем кадры из видеопотока в бесконечном цикле.
# Если кадр не удалось прочитать (read_ok == False), выходим из цикла.
# 6. Предобработка кадра
#
# frame = cv2.flip(frame, 1)
# frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# cv2.flip(frame, 1): Отражаем кадр по горизонтали для более естественного
# взаимодействия, как в зеркале. Также это необходимо для корректного определения левой-правой руки.
# cv2.cvtColor(frame, cv2.COLOR_BGR2RGB): Конвертируем цветовое пространство из BGR в
# RGB, поскольку MediaPipe работает в формате RGB.

# 7. Обработка кадра моделью распознавания рук
#
# results = hands.process(frame)
# hands.process(frame): Передаем кадр в модель для детекции и распознавания ключевых точек рук.
#
# Когда вы вызываете hands.process(frame), метод обрабатывает входное изображение frame для детекции и отслеживания
# рук. В результате вы получаете объект results, содержащий информацию о распознанных руках и их ключевых точках.
#
# Структура объекта results
# Объект results имеет следующие основные атрибуты:
#
# multi_hand_landmarks: список объектов HandLandmark, где каждый объект содержит 21 ключевую точку обнаруженной руки.
# multi_handedness: список объектов Handedness, предоставляющих информацию о ведущей руке (левая или правая) и
# уверенности классификации.
# multi_hand_world_landmarks: (если включено) список ключевых точек рук в мировых координатах.
# Давайте подробно рассмотрим каждый из этих атрибутов.


# 1. multi_hand_landmarks
# Это список обнаруженных рук, где каждая рука представлена объектом HandLandmark. Каждый такой объект содержит
# 21 ключевую точку руки.
#
# Структура ключевой точки (Landmark):
#
# x: нормализованная координата по оси X (от 0.0 до 1.0), относительно ширины изображения.
# y: нормализованная координата по оси Y (от 0.0 до 1.0), относительно высоты изображения.
# z: нормализованная координата по оси Z (глубина), где меньшие значения означают ближе к камере.
# Нормализованные координаты могут быть легко преобразованы в реальные значения в пикселях умножением на размеры кадра:
#
# h, w, c = frame.shape
# point_x, point_y = int(landmark.x * w), int(landmark.y * h)
# 2. multi_handedness
# Этот атрибут предоставляет информацию о том, является ли обнаруженная рука левой или правой, а также уверенность
# классификации.
#
# Структура Handedness:
#
# classification: список объектов Classification, обычно содержащий один элемент с наивысшей уверенностью.
# label: строка 'Left' или 'Right', указывающая на ведущую руку.
# score: значение от 0.0 до 1.0, представляющее уверенность модели в классификации.
# index: индекс категории (обычно не используется).
# Индексы ключевых точек руки
# MediaPipe Hands определяет 21 ключевую точку на руке. Каждая точка имеет свой индекс:
#
# 8. Постобработка кадра
#
# frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
# Конвертируем кадр обратно из RGB в BGR для корректного отображения с помощью OpenCV.
#
# 9. Анализ результатов и визуализация
#
# if results.multi_hand_landmarks is not None:
#     for hand_landmark, hand_handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
#         # Обработка каждой найденной руки
# results.multi_hand_landmarks: Список найденных рук и их ключевых точек.
# results.multi_handedness: Информация о ведущей руке (левая или правая).
#
# 9.1 Перебор ключевых точек руки
#
# for point_id, landmark in enumerate(hand_landmark.landmark):
#     h, w, c = frame.shape
#     point_x, point_y = int(landmark.x * w), int(landmark.y * h)
#     # Остальная обработка
# enumerate(hand_landmark.landmark): Перебираем 21 ключевую точку руки.
# landmark.x, landmark.y: Нормализованные координаты ключевой точки (от 0 до 1).
# int(landmark.x * w), int(landmark.y * h): Преобразуем нормализованные координаты в пиксели относительно размера кадра.
#
# 9.2
# Визуализация
# ключевых
# точек
#
# cv2.circle(frame, (point_x, point_y), 2, (0, 255, 0), -1)
# cv2.putText(frame, str(point_id), (point_x, point_y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)
#
# cv2.circle: Рисуем
# круг
# в
# позиции
# ключевой
# точки.
# (point_x, point_y): Координаты
# центра
# круга.
# 2: Радиус
# круга.
# (0, 255, 0): Цвет
# круга(зеленый).
# -1: Заполняем
# круг
# цветом.
# cv2.putText: Выводим
# номер
# ключевой
# точки
# рядом
# с
# ней.
# str(point_id): Номер
# ключевой
# точки.
# (point_x, point_y): Позиция
# текста.
# cv2.FONT_HERSHEY_COMPLEX: Шрифт
# текста, поддерживающий
# русский
# язык
# 0.5: Размер
# шрифта.
# (0, 255, 255): Цвет
# текста(желтый).
# 1: Толщина
# линии
# текста.
#
# 9.3 Отображение информации о руке (левая или правая)
#
# label = hand_handedness.classification[0].label
# cv2.putText(frame, str(label), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3)
# hand_handedness.classification[0].label: Получаем метку руки ('Left' или 'Right').
# Выводим эту информацию на кадре в верхнем левом углу.
# (50, 50): Координаты текста.
# (0, 0, 255): Цвет текста (красный).
# 3: Толщина линии текста.