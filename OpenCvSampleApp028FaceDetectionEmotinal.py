# Распознавание эмоций через анализ ключевых точек
# Функция recognize_emotion анализирует соотношение расстояний между ключевыми точками лица:
#
# Расстояние между зрачками (eye_distance): Измеряется расстояние между точками 159 (левый глаз) и 386 (правый глаз).
# Ширина рта (mouth_width): Измеряется расстояние между точками 61 (левый уголок рта) и 291 (правый уголок рта).
# def recognize_emotion(landmarks, frame_shape):
#     # Извлекаем необходимые точки
#     left_eye = landmarks[159]
#     right_eye = landmarks[386]
#     left_mouth = landmarks[61]
#     right_mouth = landmarks[291]
#
#     # Конвертируем нормализованные координаты в пиксели
#     def denormalize(point):
#         x = int(point.x * frame_shape[1])
#         y = int(point.y * frame_shape[0])
#         return np.array([x, y])
#
#     left_eye = denormalize(left_eye)
#     right_eye = denormalize(right_eye)
#     left_mouth = denormalize(left_mouth)
#     right_mouth = denormalize(right_mouth)
#
#     # Вычисляем метрики
#     eye_distance = np.linalg.norm(left_eye - right_eye)
#     mouth_width = np.linalg.norm(left_mouth - right_mouth)
#
#     # Простое вычисление для определения улыбки
#     smile_metric = mouth_width / eye_distance
#     if smile_metric > 0.85:
#         emotion = 'Радость'
#     else:
#         emotion = 'Нейтральное'
#
#     return emotion

# Извлечение нормализованных координат:
# Координаты точек, возвращаемые MediaPipe, находятся в диапазоне от 0 до 1. Чтобы преобразовать
# их в пиксельные значения, используется масштабирование относительно размеров кадра:
#
# x = int(point.x * frame_shape[1])
# y = int(point.y * frame_shape[0])
# Это позволяет точно позиционировать точки на кадре.
#
# Вычисление расстояний:
# Функция np.linalg.norm используется для вычисления евклидова расстояния между двумя точками, например:
#
# eye_distance = np.linalg.norm(left_eye - right_eye)
#
# Вычисления для распознавания эмоций:
# Соотношение ширины рта к межзрачковому расстоянию (smile_metric) позволяет сделать предположение об эмоции:
#
# Если smile_metric > 0.85, рот достаточно широк относительно глаз – это предполагает радость. Конечно же, этот параметр можно настраивать.
# В противном случае – нейтральное выражение.
# Интеграция функции в основной код
# После обработки кадра и получения ключевых точек (results.multi_face_landmarks), происходит:
#
# Рисование всех ключевых точек и соединений с помощью mp_drawing.draw_landmarks.
# Передача списка ключевых точек (face_landmarks.landmark) и размеров кадра в функцию recognize_emotion.
# 3. Вывод результата
# Эмоция, определённая функцией recognize_emotion, выводится на экран через cv2.putText:
#
# cv2.putText(frame, f'Эмоция: {emotion}', (30, 100), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
# Если лицо не обнаружено, выводится сообщение.

import cv2
import mediapipe as mp
import numpy as np

# Функция для распознавания эмоций
def recognize_emotion(landmarks, frame_shape):
    # Извлекаем необходимые точки
    left_eye = landmarks[159]
    right_eye = landmarks[386]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]

    # Конвертируем нормализованные координаты в пиксели
    def denormalize(point):
        x = int(point.x * frame_shape[1])
        y = int(point.y * frame_shape[0])
        return np.array([x, y])

    left_eye = denormalize(left_eye)
    right_eye = denormalize(right_eye)
    left_mouth = denormalize(left_mouth)
    right_mouth = denormalize(right_mouth)

    # Вычисляем метрики
    eye_distance = np.linalg.norm(left_eye - right_eye)
    mouth_width = np.linalg.norm(left_mouth - right_mouth)

    smile_metric = mouth_width / eye_distance
    if smile_metric > 0.85:
        emotion = 'Радость'
    else:
        emotion = 'Нейтральное'

    return emotion

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
            # Рисование ключевых точек
            mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=drawing_spec
            )

            # Определяем эмоцию
            emotion = recognize_emotion(face_landmarks.landmark, frame.shape)

            # Вывод эмоции на экран
            cv2.putText(frame, f'Эмоция: {emotion}', (30, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, 'Лицо не обнаружено', (30, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow('Face Mesh with Emotion Detection', frame)

    key = cv2.waitKey(20)
    if key == 27:  # ESC для выхода
        break

cap.release()
cv2.destroyAllWindows()