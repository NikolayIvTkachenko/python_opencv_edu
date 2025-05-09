# # Что такое MediaPipe?
# # # MediaPipe — это универсальная библиотека для обработки медиаданных в реальном времени, разработанная Google.
# # # Она предназначена для создания различных приложений, связанных с обработкой видео, изображений и других медиа.
# # # Основное преимущество MediaPipe заключается в том, что она предоставляет мощные инструменты для распознавания
# # # и отслеживания различных объектов, таких как руки, лицо, тело, а также для применения фильтров и эффектов в
# # # реальном времени.
# #
# # # Ключевые особенности MediaPipe:
# # # 1. MediaPipe разработана для эффективной обработки видеопотоков в реальном времени на различных устройствах, включая мобильные.
# # # 2. Библиотека состоит из модульных компонентов, что позволяет комбинировать различные функции и создавать сложные приложения.
# # # 3. MediaPipe поддерживает работу на различных платформах, включая Android, iOS, настольные системы и веб-браузеры.
# # # 4. MediaPipe предлагает готовые решения для задач, связанных с распознаванием рук, лица, позы, а также для работы с
# # # объектами и жестами.
# # # 5. MediaPipe предоставляет гибкие возможности для настройки и расширения функционала, что делает ее подходящей для
# # # широкого круга приложений.
# # #
# # # Примеры использования MediaPipe:
# # # 1. Отслеживание и анализ положения и движений рук, что полезно в интерактивных приложениях, жестовом управлении и
# # # виртуальной реальности.
# # # 2. Выделение ключевых точек лица для создания масок, фильтров и других эффектов дополненной реальности.
# # # 3. Определение положения тела и его частей, что используется в фитнес-приложениях, виртуальных тренерах и системах
# # # контроля осанки.
# # # 4. Создание трехмерных моделей объектов на основе видеопотока, что полезно в 3D-сценах и дополненной реальности.
# # #
# # # Возможности MediaPipe
# # # MediaPipe предлагает множество готовых решений и инструментов для работы с медиа. Рассмотрим основные функции и возможности:
# # #
# # # 1. MediaPipe Hands:
# # #    - Модуль для распознавания и отслеживания рук в реальном времени. Определяет 21 ключевую точку на каждой руке и
# # # позволяет анализировать жесты и движения.
# # #
# # # 2. MediaPipe Face Mesh:
# # #    - Модуль для распознавания лица, который выделяет 468 ключевых точек на лице, что позволяет точно
# # # отслеживать мимику и накладывать маски и эффекты.
# # #
# # # 3. MediaPipe Pose:
# # #    - Модуль для анализа позы человека. Определяет 33 ключевые точки на теле и позволяет отслеживать движения и положение частей тела.
# # #
# # # 4. MediaPipe Object Detection:
# # #    - Модуль для обнаружения и отслеживания объектов на видео. Поддерживает детекцию множества объектов одновременно.
# # #
# # # 5. MediaPipe Holistic:
# # #    - Комплексный модуль, который объединяет распознавание рук, лица и позы, позволяя отслеживать все тело человека в реальном времени.
# # #
# # # 6. MediaPipe Image Segmentation:
# # #    - Модуль для сегментации изображений, который позволяет, к примеру,  отделить человека от остальной части изображения (фона)
# #
# # #
# # # Установка MediaPipe
# # # MediaPipe легко установить с помощью пакета pip, который автоматически загружает все необходимые зависимости.
# # # Для этого выполните следующие шаги:
# # #
# # # 1. Убедитесь, что у вас установлен Python 3.6 или выше:
# # #    - Вы можете проверить версию Python с помощью команды в командной строке Windows или Linux:
# # #      python --version
# # #      Также для Linux возможен вариант команды python3 --version
# # #
# # # 2. Установка MediaPipe:
# # #    - Выполните следующую команду в терминале или командной строке для установки MediaPipe:
# # #      pip install mediapipe
# # #
# # # 3. Проверка установки:
# # #    - Чтобы убедиться, что MediaPipe установлена правильно, попробуйте импортировать библиотеку в Python:
# # #
# # # import mediapipe as mp
# # # print(mp.__version__)
# # #    - Если версия библиотеки отобразилась без ошибок, MediaPipe установлена корректно.
# # #
# # # 4. Возможные ошибки:
# # #
# # # После установки библиотеки при попытке импорта появляется ошибка:
# # # File "C:\Users\<myusername>\AppData\Local\Programs\Python\Python39\lib\site-packages\mediapipe\python\__init__.py",
# # # line 17, in <module>
# # #     from mediapipe.python._framework_bindings import resource_util
# # # ImportError: DLL load failed while importing _framework_bindings: The specified module could not be found.
# # #
# # # Process finished with exit code 1
# # # Решается при помощи установки дополнительного модуля
# # #
# # # pip install msvc-runtime
# # # также, в Windows, может потребоваться установить пакет Microsoft Visual C++ Redistributable: Ссылка
# # #
# # # При обнаружении объектов возникает предупреждение:
# # #
# # # Warning (from warnings module):
# # #   File "C:\Users\{username}\AppData\Roaming\Python\Python311\site-packages\google\protobuf\symbol_database.py", line 55
# # #     warnings.warn('SymbolDatabase.GetPrototype() is deprecated. Please '
# # # UserWarning: SymbolDatabase.GetPrototype() is deprecated. Please use message_factory.GetMessageClass() instead.
# # # SymbolDatabase.GetPrototype() will be removed soon.
# # # Решается при помощи установки библиотеки protobuf версии 3.20.3
# # #
# # # pip uninstall protobuf
# # #
# # # pip install protobuf==3.20.3
# #
# # # Детекция объектов с использованием MediaPipe
# # # Детекция объектов — одна из ключевых задач компьютерного зрения, которая позволяет обнаруживать и классифицировать
# # # объекты на изображениях или в видеопотоках. В этой главе мы рассмотрим, как использовать библиотеку MediaPipe для
# # # выполнения детекции объектов на изображениях с помощью предобученной модели.
# # #
# # # MediaPipe — это кроссплатформенный фреймворк от Google. Он широко используется для задач обработки изображений и
# # # видео, таких как детекция рук, лица, позы и т.д.
# # #
# # # Официальная документация от Google доступна тут: https://ai.google.dev/edge/mediapipe/solutions/guide
# #
# # #
# # # Детекция объектов включает два основных этапа:
# # # Локализация объектов: Определение местоположения объекта на изображении, обычно в виде ограничивающего прямоугольника
# # # (bounding box).
# # # Классификация объектов: Определение класса объекта (например, "человек", "автомобиль", "собака").
# # # Для этой задачи мы будем использовать предобученную модель EfficientDet, которая оптимизирована для мобильных устройств
# # # и встраиваемых систем.
# # #
# # # Для задачи детекции доступны разные модели с разной скоростью работы и точностью! В примерах вам уже будет доступна
# # # одна из моделей, а для собственных экспериментов вы всегда можете скачать модель из документации (не забудьте изменить
# # # имя файла модели в скрипте):
# # #
# # # https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector#efficientdet-lite0_model_recommended
# #
# # # Установка необходимых библиотек
# # # Перед началом убедитесь, что у вас установлены необходимые библиотеки:
# # #
# # # pip install mediapipe opencv-contrib-python
# # #  Шаг за шагом: Детекция объектов на изображении
# #
# # # 1. Импорт библиотек
# #
# # import cv2
# # import numpy as np
# # import mediapipe as mp
# # from mediapipe.tasks import python
# # from mediapipe.tasks.python import vision
# # # 2. Инициализация параметров и цветов для категорий
# #
# # filename = 'detect.png'
# # MARGIN = 10  # отступ в пикселях
# # ROW_SIZE = 10  # высота строки в пикселях
# # FONT_SIZE = 2  # размер шрифта
# # FONT_THICKNESS = 2  # толщина шрифта
# #
# # # Словарь цветов для каждой категории
# # CLASS_COLORS = {
# #     'person': (255, 0, 0),  # красный
# #     'car': (0, 255, 0),     # зеленый
# #     'dog': (0, 0, 255),     # синий
# #     # Добавьте другие категории и цвета по необходимости
# # }
# #
# # # 3. Функция отображения результатов
# #
# # def visualize(image, detection_result) -> np.ndarray:
# #     """Отображает результаты детекции на изображении.
# #
# #     Args:
# #         image (np.ndarray): Исходное изображение.
# #         detection_result: Результаты детекции объектов.
# #
# #     Returns:
# #         np.ndarray: Изображение с нанесенными аннотациями.
# #     """
# #     for detection in detection_result.detections:
# #         # Извлекаем информацию о категории
# #         category = detection.categories[0]
# #         category_name = category.category_name
# #         probability = round(category.score, 2)
# #         result_text = f"{category_name} ({probability})"
# #
# #         # Определяем цвет для текущей категории
# #         color = CLASS_COLORS.get(category_name, (255, 255, 255))  # белый по умолчанию
# #
# #         # Рисуем ограничивающий прямоугольник
# #         bbox = detection.bounding_box
# #         start_point = bbox.origin_x, bbox.origin_y
# #         end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
# #         cv2.rectangle(image, start_point, end_point, color, 3)
# #
# #         # Рисуем метку и вероятность
# #         text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
# #         cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, color, FONT_THICKNESS)
# #     return image
# # # Проходим по всем найденным объектам в detection_result.
# # # Извлекаем название категории и вероятность (доверие модели).
# # # Определяем цвет для категории из словаря CLASS_COLORS.
# # # Рисуем ограничивающий прямоугольник вокруг объекта.
# # # Добавляем текстовую метку с названием класса и вероятностью.
# #
# # # 4. Настройка и загрузка модели детекции объектов
# #
# # # Настройка модели детекции объектов
# # base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
# # options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
# # detector = vision.ObjectDetector.create_from_options(options)
# # # Загружаем предобученную модель efficientdet.tflite. Файл с моделью должен лежать в одной папке со скриптом.
# # # Устанавливаем порог вероятности score_threshold=0.5 — объекты с вероятностью ниже 0.5 будут игнорироваться.
# # # Создаем экземпляр детектора объектов.
# #
# # # Загрузка изображения
# # image = mp.Image.create_from_file(filename)
# #
# # # Мы используем MediaPipe для загрузки изображения из файла detect.png. Этого требует модель.
# # #
# # # 6. Выполнение детекции объектов
# # #
# # # # Выполнение детекции объектов
# # detection_result = detector.detect(image)
# # # Вызываем метод detect() для выполнения детекции объектов на загруженном изображении.
# # # Один из ключевых моментов в работе с MediaPipe - изучить структуру данных!
# # # В результате работы получим такую структуру:
# #
# # DetectionResult(
# #     detections=[
# #         Detection(
# #             bounding_box=BoundingBox(
# #                 origin_x=586,
# #                 origin_y=622,
# #                 width=366,
# #                 height=239
# #             ),
# #             categories=[
# #                 Category(
# #                     index=None,
# #                     score=0.83203125,
# #                     display_name=None,
# #                     category_name='dog'
# #                 )
# #             ],
# #             keypoints=[]
# #         ),
# #         Detection(
# #             bounding_box=BoundingBox(
# #                 origin_x=1207,
# #                 origin_y=374,
# #                 width=395,
# #                 height=186
# #             ),
# #             categories=[
# #                 Category(
# #                     index=None,
# #                     score=0.76953125,
# #                     display_name=None,
# #                     category_name='car'
# #                 )
# #             ],
# #             keypoints=[]
# #         ),
# #         Detection(
# #             bounding_box=BoundingBox(
# #                 origin_x=1209,
# #                 origin_y=589,
# #                 width=548,
# #                 height=372
# #             ),
# #             categories=[
# #                 Category(
# #                     index=None,
# #                     score=0.75390625,
# #                     display_name=None,
# #                     category_name='bicycle'
# #                 )
# #             ],
# #             keypoints=[]
# #         ),
# #         # ... остальные объекты  ...
# #     ]
# # )
# # # DetectionResult: основной объект, содержащий список всех обнаружений (detections).
# # # detections: список объектов Detection.
# # # Detection: представляет одно обнаружение на изображении.
# # # bounding_box: объект BoundingBox, определяющий положение и размер обнаруженного объекта.
# # # origin_x, origin_y: координаты верхнего левого угла ограничивающего прямоугольника.
# # # width, height: ширина и высота ограничивающего прямоугольника.
# # # categories: список возможных категорий для данного обнаружения (обычно содержит один элемент с наивысшей вероятностью).
# # # Category: содержит информацию о категории объекта.
# # # index: индекс категории (может быть None).
# # # score: вероятность того, что объект принадлежит данной категории.
# # # display_name: отображаемое имя категории (может быть None).
# # # category_name: имя категории, например, 'dog', 'car', 'bicycle'.
# # # keypoints: список ключевых точек (в данном случае пустой список []).
# # # 7. Визуализация результатов
# #
# # # Копирование изображения и визуализация результатов
# # image_copy = np.copy(image.numpy_view())
# # annotated_image = visualize(image_copy, detection_result)
# #
# # 8. Сохранение результата
# #
# # # Конвертация изображения из BGR в RGB и сохранение результата
# # rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
# # cv2.imwrite("result.png", rgb_annotated_image)
# # # OpenCV использует цветовое пространство BGR, а MediaPipe RGB, поэтому мы конвертируем изображение в BGR перед сохранением.
# # # Сохраняем размеченное изображение как result.png.
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

filename = 'detect.png'
MARGIN = 10  # отступ в пикселях
ROW_SIZE = 10  # высота строки в пикселях
FONT_SIZE = 1  # размер шрифта
FONT_THICKNESS = 1  # толщина шрифта

# Словарь цветов для каждой категории
CLASS_COLORS = {
    'person': (255, 0, 0),  # красный
    'car': (0, 255, 0),  # зеленый
    'dog': (0, 0, 255),  # синий
    # Добавьте другие категории и цвета по необходимости
}


def visualize(image, detection_result) -> np.ndarray:
    """Отображает результаты детекции на изображении.

    Args:
        image (np.ndarray): Исходное изображение.
        detection_result: Результаты детекции объектов.

    Returns:
        np.ndarray: Изображение с нанесенными аннотациями.
    """
    for detection in detection_result.detections:
        # Извлекаем информацию о категории
        category = detection.categories[0]
        category_name = category.category_name
        probability = round(category.score, 2)
        result_text = f"{category_name} ({probability})"

        # Определяем цвет для текущей категории
        color = CLASS_COLORS.get(category_name, (255, 255, 255))  # белый по умолчанию

        # Рисуем ограничивающий прямоугольник
        bbox = detection.bounding_box
        start_point = bbox.origin_x, bbox.origin_y
        end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
        cv2.rectangle(image, start_point, end_point, color, 3)

        # Рисуем метку и вероятность
        text_location = (MARGIN + bbox.origin_x, MARGIN + ROW_SIZE + bbox.origin_y)
        cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN, FONT_SIZE, color, FONT_THICKNESS)
    return image


# Настройка модели детекции объектов
base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options, score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# Загрузка изображения
image = mp.Image.create_from_file(filename)

# Выполнение детекции объектов
detection_result = detector.detect(image)

# Копирование изображения и визуализация результатов
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)

# Конвертация изображения из BGR в RGB и сохранение результата
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
cv2.imwrite("result.png", rgb_annotated_image)
