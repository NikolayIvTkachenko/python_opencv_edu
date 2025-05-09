# Расширение функционала с использованием MediaPipe Face Mesh
# В этом уроке мы переходим от базового определения лиц (как это было в предыдущем коде с использованием
# MediaPipe Face Detection) к более сложному анализу лица с помощью MediaPipe Face Mesh. Face Mesh позволяет
# находить 468 ключевых точек на лице, что делает его весьма удобным инструментом для задач, связанных
# с отслеживанием деталей лица: для дополненной реальности, анимации и анализа мимики.
#
# # Официальная документация
# Ключевые отличия между Face Detection и Face Mesh
# Face Detection: обнаруживает область лица в кадре и возвращает базовые ключевые точки (глаза, нос, уши, рот).
# Подходит для простого позиционирования объектов.
#
# Face Mesh: создаёт подробную сетку из 468 ключевых точек на лице. Эти точки формируют «каркас» лица, который
# можно использовать для точного анализа и визуализации.
# Пример кода, с которого начнется изучение Face Mesh:

import cv2
import mediapipe as mp

# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Захват видео с камеры
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)


while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Поворот кадра для зеркального отображения
    frame = cv2.flip(frame, 1)

    # Конвертация в RGB для MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обработка кадра
    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec
            )
    else:
        cv2.putText(frame, 'Лицо не обнаружено', (30, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face Mesh', frame)

    key = cv2.waitKey(20)
    if key == 27:  # ESC для выхода
        break

cap.release()
cv2.destroyAllWindows()


# Инициализация Face Mesh
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=False,  # Режим обработки видео (False) или изображений (True)
#     max_num_faces=1,  # Максимальное количество лиц, которые нужно анализировать
#     refine_landmarks=True,  # Используется для уточнения ключевых точек вокруг глаз, губ и зрачков
#     min_detection_confidence=0.5,  # Минимальная уверенность в обнаружении лица
#     min_tracking_confidence=0.5  # Минимальная уверенность в отслеживании
# )
# Настройка визуализации точек
# mp_drawing = mp.solutions.drawing_utils
# drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
# mp_drawing.DrawingSpec: Определяет параметры рисования:
#
# thickness: толщина линий.
# circle_radius: радиус окружностей для точек.
# Рисование ключевых точек и соединений
# if results.multi_face_landmarks:
#     for face_landmarks in results.multi_face_landmarks:
#         mp_drawing.draw_landmarks(
#             image=frame,
#             landmark_list=face_landmarks,
#             connections=mp_face_mesh.FACEMESH_TESSELATION,
#             landmark_drawing_spec=None,
#             connection_drawing_spec=drawing_spec
#         )
# else:
#     cv2.putText(frame, 'Лицо не обнаружено', (30, 50),
#                 cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
# results.multi_face_landmarks:
# Содержит список ключевых точек для всех обнаруженных лиц.
#
# draw_landmarks:
# Рисует ключевые точки и соединения между ними (тесселяция).
#
# FACEMESH_TESSELATION: Предопределённый список связей между точками, образующий сетку.
# FACEMESH_CONTOURS: Рисует контуры лица, глаз
# Если лицо не обнаружено, выводится сообщение "Лицо не обнаружено".