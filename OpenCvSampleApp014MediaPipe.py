# Детекция объектов в реальном времени
# Детекция объектов на видеопотоке с камеры позволяет выполнять анализ окружающей среды в реальном времени.
# Для этого нам нужно захватывать кадры с камеры, выполнять детекцию на каждом кадре и отображать результаты.
#
# Основной код этого скрипта будет совпадать с определением классов на изображении:
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MARGIN = 10  # отступ в пикселях
ROW_SIZE = 20  # высота строки в пикселях
FONT_SIZE = 1  # размер шрифта
FONT_THICKNESS = 1  # толщина шрифта

# Словарь цветов для каждой категории
CLASS_COLORS = {
    'person': (255, 0, 0),    # красный
    'car': (0, 255, 0),       # зеленый
    'dog': (0, 0, 255),       # синий
    'bicycle': (255, 255, 0), # голубой
    'bench': (255, 0, 255),   # розовый
    # Добавьте другие категории и цвета по необходимости
}

def visualize(image, detection_result) -> np.ndarray:
    for detection in detection_result.detections:
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"
        color = CLASS_COLORS.get(category_name, (255, 255, 255))

        bbox = detection.bounding_box
        start_point = int(bbox.origin_x), int(bbox.origin_y)
        end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
        cv2.rectangle(image, start_point, end_point, color, 2)

        text_location = (int(MARGIN + bbox.origin_x), int(MARGIN + ROW_SIZE + bbox.origin_y))
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_SIMPLEX, FONT_SIZE, color, FONT_THICKNESS)
    return image

# Настройка модели детекции объектов
base_options = python.BaseOptions(model_asset_path='efficientdet-d2-640 (1).h5') #efficientdet.tflite
options = vision.ObjectDetectorOptions(
    base_options=base_options,
    score_threshold=0.5,
    max_results=5
)
detector = vision.ObjectDetector.create_from_options(options)

# Захват видеопотока с камеры
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Не удалось получить кадр с камеры")
        break

    # Преобразование кадра в формат MediaPipe
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Выполнение детекции объектов
    detection_result = detector.detect(mp_image)

    # Визуализация результатов
    annotated_image = visualize(frame, detection_result)

    # Отображение кадра
    cv2.imshow('MediaPipe Object Detection', annotated_image)

    # Выход по нажатию клавиши 'Esc'
    if cv2.waitKey(20) == 27:
        break

cap.release()
cv2.destroyAllWindows()