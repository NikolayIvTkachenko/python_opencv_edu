# Как обработать точки?
# В предыдущем примере создание подписей к точкам производилась в цикле:
#
# if results.multi_hand_landmarks:
#         for hand_landmark in results.multi_hand_landmarks:
#             for point_id, landmark in enumerate(hand_landmark.landmark):
#                 h, w, c = frame.shape
#                 point_x, point_y = int(landmark.x * w), int(landmark.y * h)
# Такой подход позволяет быстро и удобно визуализировать данные на начальном этапе, но не очень удобен
# для дальнейшей обработки.
# Для анализа создадим список координат всех точек в пикселях:
#
# if results.multi_hand_landmarks:
#         for hand_landmark in results.multi_hand_landmarks:
#             h, w, c = frame.shape
#             # Преобразуем ключевые точки в список координат в пикселях
#             landmarks = [
#                 (int(landmark.x * w), int(landmark.y * h)) for landmark in hand_landmark.landmark
#             ]
#  И после этого их можно удобно обрабатывать, создавать пары точек, между которыми, в последствии находить
#  расстояние и определять жесты!
#
# Найти расстояние между точками на изображении достаточно просто по теореме Пифагора:
#
# Для удобного вычисления создадим функцию calculate_distance():
#
# def calculate_distance(point1, point2):
#     """Вычисляет расстояние между двумя точками."""
#     return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5
# Параметры:
# point1: координаты первой точки (например, (x1, y1)).
# point2: координаты второй точки (например, (x2, y2)).
# Возвращаемое значение: расстояние между этими точками.
# ((...)+(...)) ** 0.5: возведение в степень 0.5 эквивалентно операции извлечения корня
# Доработанный пример кода со списком точек, созданием словаря пар точек для вычисления расстояний:

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

            # Создаем словарь пар ключевых точек
            # Индексы пар точек
            pair_indices = [
                (4, 8),  # Кончик большого пальца и указательного
                (8, 12),  # Кончики указательного и среднего пальцев
                (0, 12),  # Запястье и средний палец
                (0, 1),  # Запястье и основание большого пальца
                (0, 4),  # Запястье и кончик большого пальца
                (20, 4),  # Кончик мизинца и большого пальца
                (20, 0)  # Кончик мизинца и запястье
            ]

            # Генерация пар точек
            pairs = {f"pair_{i}": (landmarks[p1], landmarks[p2]) for i, (p1, p2) in enumerate(pair_indices)}


            # Вычисляем расстояния для каждой пары
            distances = {
                pair_name: calculate_distance(pair[0], pair[1])
                for pair_name, pair in pairs.items()
            }

            # Визуализация ключевых точек и расстояний
            for pair_name, (point1, point2) in pairs.items():
                cv2.line(frame, point1, point2, (255, 0, 0), 2)  # Линия между точками
                cv2.circle(frame, point1, 5, (0, 255, 0), -1)  # Первая точка
                cv2.circle(frame, point2, 5, (0, 255, 0), -1)  # Вторая точка
                midpoint = ((point1[0] + point2[0]) // 2, (point1[1] + point2[1]) // 2)
                cv2.putText(frame, f"{int(distances[pair_name])}", midpoint, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Hand Gesture Detection", frame)

    # Выход по клавише Esc
    if cv2.waitKey(30)== 27:
        break

cap.release()
cv2.destroyAllWindows()

# # -------------------------------
# # -------------------------------
# Справочные материалы по JSON
# Для решения следующих задач удобно использовать библиотеку json.
#
# Что такое JSON?
# JSON (JavaScript Object Notation) — это текстовый формат для обмена данными между приложениями. Он прост, читаем и похож на словари в Python. JSON используется для хранения и передачи структурированных данных.
#
# Данные представлены в виде пар ключ-значение, как в словаре Python.
#
# Типы данных:
#
# Ключи: строки в двойных кавычках ("key").
# Значения: числа, строки, массивы, объекты (вложенные JSON), логические значения (true, false) и null.
# JSON очень похож на словари Python, но есть несколько различий:
#
# Элемент	JSON	Python-словарь
# Строки	Всегда в двойных кавычках (")	Одинарные или двойные кавычки
# Булевы значения	true, false	True, False
# Пустое значение	null	None
# Работа с JSON в Python
#
# Для работы с JSON в Python используется модуль json. Он предоставляет функции для преобразования между JSON-строкой и Python-объектом (например, словарём).
#
# Чтение JSON из строки:
#
# import json
#
# # JSON-строка
# json_string = '{"name": "Alice", "age": 25, "is_student": false}'
#
# # Преобразование в словарь Python
# data = json.loads(json_string)
# print(data)  # {'name': 'Alice', 'age': 25, 'is_student': False}
# Запись словаря в JSON-строку:
# data = {
#     "name": "Bob",
#     "age": 30,
#     "skills": ["Python", "Java"]
# }
#
# # Преобразование словаря в JSON-строку
# json_string = json.dumps(data, indent=4)
# print(json_string)
# В функции json.dumps() параметр indent отвечает за форматирование выходной JSON-строки, делая её более читаемой.
# Если указать indent=4, каждый уровень вложенности JSON-объекта будет отступать на 4 пробела.
# {
#     "name": "Bob",
#     "age": 30,
#     "skills": [
#         "Python",
#         "Java"
#     ]
# }
# Если параметр indent не указан, JSON-строка будет сгенерирована в компактном формате (без лишних пробелов и переносов строк).
# {"name": "Bob", "age": 30, "skills": ["Python", "Java"]}
# Чтение JSON из файла:
# with open("data.json", "r") as file:
#     data = json.load(file)
#     print(data)  # Содержимое файла преобразуется в словарь
# Запись JSON в файл:
# data = {
#     "name": "Charlie",
#     "age": 22,
#     "hobbies": ["gaming", "reading"]
# }
#
# with open("output.json", "w") as file:
#     json.dump(data, file, indent=4)

# import json
#
# lines = []
# while True:
#     try:
#         line = input().strip()
#         if line:
#             lines.append(line)
#         else:
#             break
#     except EOFError:
#         break
#
# # Объединяем строки в одну строку JSON
# input_data = " ".join(lines)
#
#  # Преобразуем JSON-строку в Python-объекты
# data = json.loads(input_data)
# landmarks = data["landmarks"]
# pairs = data["pairs"]
#
#
# #Ваш код