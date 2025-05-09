# Определение позы человека (Pose Landmark Detection) с помощью MediaPipe
# Познакомимся со следующим с инструментом: MediaPipe Pose Landmarker, позволяющим
# определять ключевые точки человеческого
# тела (позу) в режиме реального времени. Научиться инициализировать, использовать
# и визуализировать результаты детектирования позы, а также получить первичные навыки анализа положения тела.
#
# Официальная документация:
# MediaPipe Pose
#
# MediaPipe Pose Landmarker — это решение, которое позволяет детектировать 33 ключевые точки человеческого тела:
# голову, плечи, локти, запястья, таз, колени, лодыжки и некоторые дополнительные точки для детального анализа ног
# и рук. Благодаря этому можно:
#
# Отслеживать движение: определять углы в суставах и анализировать динамику.
# Сценарии использования: фитнес-приложения (отслеживание правильности выполнения упражнений), танцевальные или
# спортивные анализаторы, жестовый контроль в AR/VR-приложениях.

import cv2
import mediapipe as mp
import numpy as np

# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=False,
    model_complexity=1,
    enable_segmentation=False,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Поворот кадра для удобства (зеркальное отображение)
    frame = cv2.flip(frame, 1)

    # Конвертируем в RGB, т.к. MediaPipe ожидает именно RGB-формат
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Обрабатываем кадр с помощью pose
    results = pose.process(frame_rgb)

    # Если landmarks обнаружены
    if results.pose_landmarks:
        # Рисуем ключевые точки и скелет человека
        mp_drawing.draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
        )
    else:
        cv2.putText(frame, 'Позы не обнаружено', (30, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    cv2.putText(frame, 'MediaPipe Pose Demo', (30, 100),
                cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Pose Landmarks', frame)
    if cv2.waitKey(20) == 27:  # ESC для выхода
        break

cap.release()
cv2.destroyAllWindows()

#
# Объект results.pose_landmarks содержит 33 точки, каждая со структурой, похожей на Face Mesh:
#
# for i, lm in enumerate(results.pose_landmarks.landmark):
#     # lm.x, lm.y, lm.z, lm.visibility
#     # lm.x и lm.y - нормализованные координаты в диапазоне [0, 1]
#     # z - глубина (относительная, отрицательное значение - ближе к камере)
#     # visibility - вероятность, что точка видима
# Для вычислений в пикселях, умножьте lm.x на ширину кадра, lm.y на высоту кадра:
#
# height, width, _ = frame.shape
# x_pixel = int(lm.x * width)
# y_pixel = int(lm.y * height)
# Если модель работает нестабильно, можно повысить min_detection_confidence или min_tracking_confidence или выбрать
# более сложную модель (model_complexity=2).