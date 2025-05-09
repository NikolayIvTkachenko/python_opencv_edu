# Проект: Создание приложения для распознавания эмоций и автоматической фотосъемки
# Научиться использовать MediaPipe Face Mesh для распознавания эмоций, автоматизировать съемку фотографии при улыбке,
# а также отобразить анимацию обратного отсчета и визуальное подтверждение о сохранении изображения.
#
# Основные этапы:
#
# Распознавание лица и ключевых точек при помощи MediaPipe Face Mesh:
# Мы уже знаем, как инициализировать MediaPipe и получать набор из 468 ключевых точек лица. Это дает нам точную геометрию,
# на основе которой можно анализировать различные метрики.
#
# Определение эмоции «Радость»:
# Используя функцию recognize_emotion, мы извлекаем координаты ключевых точек, отвечающих за глаза и уголки рта.
# Сравнивая ширину рта (расстояние между точками 61 и 291) с межзрачковым расстоянием (расстояние между точками 159 и 386), мы вычисляем smile_metric:
#
# smile_metric = mouth_width / eye_distance
#
# Если smile_metric > 0.85, мы считаем, что на лице появилась улыбка.
#
# Автоматическая съемка фото при улыбке:
# Как только определяется эмоция «Радость», запускается таймер обратного отсчета (5 секунд). Это дает пользователю
# время зафиксировать улыбку. По истечении 5 секунд код автоматически делает снимок и сохраняет его в папку photos.
#
# Анимация обратного отсчета:
# В правом верхнем углу экрана отображается круговая шкала, заполняющаяся по мере истечения времени. Внутри круга
# показано количество оставшихся секунд. Эта наглядная анимация помогает пользователю понять, сколько времени осталось до срабатывания камеры.
#
# Вывод подсказок и сообщений:
#
# До начала обратного отсчета на экране показывается текст: «Улыбнитесь и фото будет сохранено», а также информация
# об обнаруженной эмоции.
# Во время обратного отсчета сетка лица и надписи о сохранении временно скрываются, чтобы не отвлекать пользователя.
# После сохранения фото на 2 секунды в нижней части экрана появляется сообщение: «Фото сохранено в файл photos/имя_файла»,
# информирующее о том, что снимок успешно создан.
# Структура кода:
#
# Инициализация MediaPipe и OpenCV.
# Проверка наличия папки photos и создание её при необходимости.
# В главном цикле:
# Захват и обработка кадра.
# Обнаружение лица и вычисление эмоции.
# Запуск и отображение таймера при улыбке.
# Сохранение изображения по истечении обратного отсчета.
# Показ сервисных сообщений и подсказок.

import cv2
import mediapipe as mp
import numpy as np
import os
import time

# Проверяем и создаем папку для фото
if not os.path.exists("photos"):
    os.makedirs("photos")


# Функция для распознавания эмоций
def recognize_emotion(landmarks, frame_shape):
    # Извлекаем необходимые точки
    left_eye = landmarks[159]
    right_eye = landmarks[386]
    left_mouth = landmarks[61]
    right_mouth = landmarks[291]

    def denormalize(point):
        x = int(point.x * frame_shape[1])
        y = int(point.y * frame_shape[0])
        return np.array([x, y])

    left_eye = denormalize(left_eye)
    right_eye = denormalize(right_eye)
    left_mouth = denormalize(left_mouth)
    right_mouth = denormalize(right_mouth)

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

# Переменные для анимации обратного отсчета
start_time = None
countdown_duration = 5.0  # Сколько секунд ждать до сохранения фото
is_countdown_active = False

# Переменные для отображения сообщения о сохранении фото
photo_saved_message = ""
photo_saved_time = None
message_duration = 2.0  # Секунд показывать сообщение

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Поворот кадра для зеркального отображения
    frame = cv2.flip(frame, 1)
    original_frame = frame.copy()

    # Конвертация в RGB для MediaPipe
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    emotion = 'Нейтральное'
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Определяем эмоцию, если таймер не активен
            if not is_countdown_active:
                emotion = recognize_emotion(face_landmarks.landmark, frame.shape)
                # Если эмоция "Радость" и таймер не запущен, запускаем обратный отсчет
                if emotion == 'Радость' and not is_countdown_active:
                    start_time = time.time()
                    is_countdown_active = True

    # Если таймер активен, показываем только обратный отсчет
    if is_countdown_active:
        elapsed = time.time() - start_time
        remaining = countdown_duration - elapsed

        # Рисуем анимацию ожидания
        center = (frame.shape[1] - 100, 100)  # Позиция в правом верхнем углу
        circle_radius = 40
        fraction = min(elapsed / countdown_duration, 1.0)
        angle = int(360 * fraction)
        axes = (circle_radius, circle_radius)

        # Если время истекло, делаем снимок
        if remaining <= 0:
            filename = f"photos/photo_{time.time()}.png"
            cv2.imwrite(filename, original_frame)
            is_countdown_active = False
            # Устанавливаем сообщение о сохранении
            photo_saved_message = f"Фото сохранено в файл {filename}"
            photo_saved_time = time.time()
        else:
            # Зеленая дуга для прошедшего времени
            cv2.ellipse(frame, center, axes, -90, 0, angle, (0, 255, 0), 5)
            # Синяя дуга для оставшегося времени
            cv2.ellipse(frame, center, axes, -90, angle, 360, (255, 0, 0), 2)

            # Центрируем текст с оставшимся временем
            countdown_text = f'{remaining:.1f}'
            (text_width, text_height), _ = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
            text_x = center[0] - text_width // 2
            text_y = center[1] + text_height // 2
            cv2.putText(frame, countdown_text, (text_x, text_y),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)


    else:
        # Если таймер не активен, показываем обычную сцену
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=frame,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=drawing_spec
                )

            # Информационное сообщение для пользователя (справа вверху)
            cv2.putText(frame, 'Улыбнитесь и фото будет сохранено', (frame.shape[1] - 750, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)

            # Показ эмоции
            cv2.putText(frame, f'Эмоция: {emotion}', (30, 100),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, 'Лицо не обнаружено', (30, 50),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    # Отображение сообщения о сохранении фото, если есть и не истекло время
    if photo_saved_message and photo_saved_time is not None:
        elapsed_since_save = time.time() - photo_saved_time
        if elapsed_since_save < message_duration:
            (msg_width, msg_height), _ = cv2.getTextSize(photo_saved_message, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
            msg_x = (frame.shape[1] - msg_width) // 2
            msg_y = frame.shape[0] - 30  # немного отступим от низа
            cv2.putText(frame, photo_saved_message, (msg_x, msg_y),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
        else:
            # Сбрасываем сообщение по истечению 2 секунд
            photo_saved_message = ""
            photo_saved_time = None

    cv2.imshow('Emotion Detection', frame)

    key = cv2.waitKey(20)
    if key == 27:  # ESC для выхода
        break

cap.release()
cv2.destroyAllWindows()