
# Генерация листа по углам ArUco коды/диаграммы
# import cv2
# import numpy as np
# import os
# from fpdf import FPDF
#
#
# def createArucoMarkerWithBorder(markerSize, markerId, imgSize):
#     """
#     Создает Aruco маркер с белой рамкой.
#
#     :param markerSize: Размер маркера (например, 4x4, 5x5).
#     :param markerId: ID маркера.
#     :param imgSize: Разрешение изображения маркера.
#     :return: Возвращает изображение маркера.
#     """
#     # Создаем словарь Aruco маркеров нужного размера
#     dict_sizes = {
#         4: cv2.aruco.DICT_4X4_250,
#         5: cv2.aruco.DICT_5X5_250,
#         6: cv2.aruco.DICT_6X6_250,
#         7: cv2.aruco.DICT_7X7_250
#     }
#
#     if markerSize in dict_sizes:
#         arucoDict = cv2.aruco.getPredefinedDictionary(dict_sizes[markerSize])
#     else:
#         raise ValueError("Неподдерживаемый размер маркера. Поддерживаемые размеры: 4, 5, 6, 7.")
#
#     # Генерируем маркер
#     markerImage = np.zeros((imgSize, imgSize), dtype=np.uint8)
#     cv2.aruco.generateImageMarker(arucoDict, markerId, imgSize, markerImage, 1)
#
#     # Рассчитываем размер нового изображения с рамкой
#     borderSize = imgSize // (markerSize + 2)
#     newSize = imgSize + borderSize * 2 + 2
#
#     # Создаем новое изображение с белой рамкой
#     newImage = np.ones((newSize, newSize), dtype=np.uint8) * 255
#     newImage[borderSize + 1:-borderSize - 1, borderSize + 1:-borderSize - 1] = markerImage
#
#     # Добавляем пунктирную линию на крайних пикселях рамки
#     for i in range(0, newSize, 4):
#         newImage[i:i + 2, 0] = 0
#         newImage[i:i + 2, -1] = 0
#         newImage[0, i:i + 2] = 0
#         newImage[-1, i:i + 2] = 0
#
#     # Добавляем текст с ID маркера
#     text = f"{markerId}"
#     targetTextHeight = imgSize * 0.07  # 7% от высоты изображения
#     fontScale = 0.1  # Начальный масштаб шрифта
#     thickness = max(1, int(imgSize / 500))
#     textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)[0]
#
#     # Подбираем масштаб шрифта, чтобы высота текста была приблизительно 7% от imgSize
#     while textSize[1] < targetTextHeight:
#         fontScale += 0.1
#         textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)[0]
#
#     textX = newSize - textSize[0] - int(imgSize * 0.02)  # от правого края
#     textY = newSize - int(imgSize * 0.02)  # от нижнего края
#     cv2.putText(newImage, text, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), thickness)
#
#     # Возвращаем изображение маркера
#     return newImage
#
#
# def createArucoMarkersPDF(markerList, mmSize):
#     """
#     Создает PDF файл с Aruco маркерами, учитывая поля в 15 мм.
#
#     :param markerList: Список маркеров в формате (размер маркера, ID, размер изображения).
#     :param mmSize: Размер маркера в миллиметрах.
#     """
#     # Проверяем и создаем папку для маркеров
#     folderName = "ArucoMarkers"
#     if not os.path.exists(folderName):
#         os.makedirs(folderName)
#
#     # Создаем маркеры и сохраняем их в папку
#     for markerSize, markerId, imgSize in markerList:
#         markerImage = createArucoMarkerWithBorder(markerSize, markerId, imgSize)
#         cv2.imwrite(f"{folderName}/aruco_marker_{markerId}_{markerSize}x{markerSize}.png", markerImage)
#
#     # Создаем PDF
#     pdf = FPDF('P', 'mm', 'A4')
#     pdf.add_page()
#     margin = 15  # Поля в мм
#     current_x, current_y = margin, margin
#     page_width, page_height = 297, 210  # Размеры страницы A4 в мм
#
#     for markerSize, markerId, imgSize in markerList:
#         filePath = f"{folderName}/aruco_marker_{markerId}_{markerSize}x{markerSize}.png"
#         if os.path.exists(filePath):
#             if current_x + mmSize > page_width - margin:
#                 current_x = margin
#                 current_y += mmSize
#             if current_y + mmSize > page_height - margin:
#                 pdf.add_page()
#                 current_x, current_y = margin, margin
#             pdf.image(filePath, x=current_x, y=current_y, w=mmSize, h=mmSize)
#             current_x += mmSize
#     # Сохраняем PDF
#     pdf.output(f"{folderName}/ArucoMarkers.pdf")
#
#
# # Пример использования функции
# marker_dict = 4
# marker_resolution = 200
# marker_size_mm = 60
# marker_list = [(marker_dict, i, marker_resolution) for i in range(1, 10, 3)]
# createArucoMarkersPDF(marker_list,
#                       marker_size_mm)  # Создает PDF с маркерами 4x4 размером 60 мм и изображениями разрешением 200x200 пикселей
#
#
# # Добавляем функцию для создания PDF с 4 маркерами по углам
# def createArucoMarkersPDFWithCorners(markerSize, markerIds, imgSize, mmSize, pageSize_mm=(297, 210), margin_mm=5):
#     """
#     Создает PDF файл с 4 Aruco маркерами, расположенными по углам страницы.
#
#     :param markerSize: Размер маркера (например, 4x4, 5x5).
#     :param markerIds: Список из 4 ID маркеров.
#     :param imgSize: Разрешение изображения маркера.
#     :param mmSize: Размер маркера в миллиметрах.
#     :param pageSize_mm: Размер страницы в мм, по умолчанию A4.
#     :param margin_mm: Отступ от краев страницы в мм.
#     """
#     if len(markerIds) != 4:
#         raise ValueError("Необходимо передать ровно 4 ID маркеров.")
#
#     # Проверяем и создаем папку для маркеров
#     folderName = "ArucoMarkers_Corners"
#     if not os.path.exists(folderName):
#         os.makedirs(folderName)
#
#     # Создаем маркеры и сохраняем их в папку
#     for markerId in markerIds:
#         markerImage = createArucoMarkerWithBorder(markerSize, markerId, imgSize)
#         cv2.imwrite(f"{folderName}/aruco_marker_{markerId}_{markerSize}x{markerSize}.png", markerImage)
#
#     # Создаем PDF
#     pdf = FPDF('P', 'mm', pageSize_mm)
#     pdf.add_page()
#     page_width, page_height = pageSize_mm
#
#     # Позиции для 4 углов
#     positions = [
#         (margin_mm, margin_mm),  # Верхний левый угол
#         (page_width - margin_mm - mmSize, margin_mm),  # Верхний правый угол
#         (page_width - margin_mm - mmSize, page_height - margin_mm - mmSize),  # Нижний правый угол
#         (margin_mm, page_height - margin_mm - mmSize)  # Нижний левый угол
#     ]
#
#     # Добавляем маркеры на страницу
#     for i, (x, y) in enumerate(positions):
#         markerId = markerIds[i]
#         filePath = f"{folderName}/aruco_marker_{markerId}_{markerSize}x{markerSize}.png"
#         if os.path.exists(filePath):
#             pdf.image(filePath, x=x, y=y, w=mmSize, h=mmSize)
#
#     # Сохраняем PDF
#     pdf.output(f"{folderName}/ArucoMarkers_Corners.pdf")
#
#
# # Пример использования новой функции
# markerSize = 4
# markerIds = [10, 20, 30, 40]  # Ваши 4 ID маркеров
# imgSize = 200  # Разрешение изображения маркера
# mmSize = 50  # Размер маркера на странице в мм
#
# createArucoMarkersPDFWithCorners(markerSize, markerIds, imgSize, mmSize)

# =======================================================================
# import cv2
# import numpy as np
#
# # Загружаем основное изображение и изображение для вставки
# main_image = cv2.imread('01.jpg')
# insert_image = cv2.imread('image.jpg')
#
# # Проверка, что изображения загружены
# if main_image is None or insert_image is None:
#     raise ValueError("Не удалось загрузить одно из изображений")
#
# # Настройка словаря и параметров детектора ArUco
# aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
# detector_params = cv2.aruco.DetectorParameters()
# detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)
#
# # Обнаружение маркеров на основном изображении
# corners, ids, _ = detector.detectMarkers(main_image)
#
# # Проверяем, что нужные маркеры обнаружены
# required_ids = {100, 101, 102, 103}
# if ids is None or not required_ids.issubset(set(ids.flatten())):
#     raise ValueError("Не все маркеры (100, 101, 102, 103) обнаружены на изображении")
#
# # Сопоставляем маркеры с угловыми точками
# marker_corners = {id[0]: corner for id, corner in zip(ids, corners)}
#
# # Упорядочиваем точки по ID маркеров
# dst_points = np.array([
#     marker_corners[100][0][0],  # Точка 0 - верхний левый угол (Маркер 100)
#     marker_corners[101][0][1],  # Точка 1 - верхний правый угол (Маркер 101)
#     marker_corners[102][0][2],  # Точка 2 - нижний правый угол (Маркер 102)
#     marker_corners[103][0][3]   # Точка 3 - нижний левый угол (Маркер 103)
# ], dtype="float32")
#
# # Определяем угловые точки изображения для вставки
# h, w = insert_image.shape[:2]
# src_points = np.array([
#     [0, 0],           # Верхний левый угол
#     [w - 1, 0],       # Верхний правый угол
#     [w - 1, h - 1],   # Нижний правый угол
#     [0, h - 1]        # Нижний левый угол
# ], dtype="float32")
#
# # Вычисляем матрицу перспективного преобразования
# matrix = cv2.getPerspectiveTransform(src_points, dst_points)
#
# # Применяем перспективное преобразование к изображению для вставки
# warped_insert = cv2.warpPerspective(insert_image, matrix, (main_image.shape[1], main_image.shape[0]))
#
# # Сохраняем деформированное изображение
# cv2.imwrite("warped_insert.jpg", warped_insert)
#
# # Создаем маску для вставляемого изображения
# mask = np.zeros_like(main_image)
# cv2.fillPoly(mask, [np.int32(dst_points)], (255, 255, 255))
#
# # Сохраняем маску
# cv2.imwrite("mask.jpg", mask)
#
# # Инвертируем маску для удаления области вставки на исходном изображении
# mask_inv = cv2.bitwise_not(mask)
#
# # Сохраняем инвертированную маску
# cv2.imwrite("mask_inv.jpg", mask_inv)
#
# # Удаляем область вставки на основном изображении
# main_image_bg = cv2.bitwise_and(main_image, mask_inv)
#
# # Сохраняем изображение фона с удаленной областью вставки
# cv2.imwrite("main_image_bg.jpg", main_image_bg)
#
# # Объединяем фон с деформированным изображением для вставки
# result = cv2.add(main_image_bg, warped_insert)
#
# # Сохраняем финальное изображение
# cv2.imwrite("final_result.jpg", result)
# print("Изображение успешно создано и сохранено как final_result.jpg")


# Теперь подробно разберем каждый этап
# Загрузка изображений: Основное изображение (01.jpg) и изображение для вставки (image.jpg) загружаются с помощью OpenCV.
# Проверка на загрузку помогает избежать ошибок, если файлы не были найдены.
#
# Обнаружение ArUco маркеров: Используем словарь DICT_4X4_250 и параметры детектора для нахождения маркеров 4x4.
# На изображении должны присутствовать маркеры с ID 100, 101, 102 и 103, чтобы указать углы области вставки.
#
# Сопоставляем маркеры с угловыми точками: marker_corners = {id[0]: corner for id, corner in zip(ids, corners)}
#
# zip(ids, corners):
#
# ids — это массив, содержащий идентификаторы (ID) обнаруженных ArUco маркеров. Например, если на
# изображении найдены маркеры с ID 100, 101, 102, и 103, то ids может выглядеть так: [[100], [101], [102], [103]].
# corners — это массив, в котором хранятся координаты угловых точек каждого найденного маркера. Например, если каждый
# маркер имеет четыре угловые точки, corners будет выглядеть как многоуровневый массив:
# [
#     [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]],  # Углы маркера с ID 100
#     [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]],  # Углы маркера с ID 101
#     [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]],  # Углы маркера с ID 102
#     [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]   # Углы маркера с ID 103
# ]
# zip(ids, corners) создает пары, где каждый элемент из ids связывается с соответствующим элементом из corners,
# то есть zip соединяет ID каждого маркера с его угловыми точками.
# {id[0]: corner for id, corner in zip(ids, corners)}:
#
# Это генератор словаря, который создает новый словарь marker_corners.
#
# Для каждой пары (id, corner) из zip(ids, corners), id[0] используется в качестве ключа, а corner — как значение.
#
# id[0] — это идентификатор маркера в виде целого числа. В массиве ids каждый ID представлен как вложенный массив,
# например, [100]. Мы используем id[0], чтобы извлечь целое число 100.
#
# corner — это массив, представляющий координаты четырех углов маркера. Значение corner для маркера с
# ID 100 может выглядеть так:
#
# [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]]]
#
# Здесь corner является списком из четырех точек (углов), и каждая точка содержит координаты x и y.
#
# Результат:
#
# После выполнения этой строки marker_corners будет словарем, где ключ — это ID каждого обнаруженного маркера,
# а значение — массив координат его четырех угловых точек.
# marker_corners = {
#     100: [[[167, 167], [1166, 167], [1166, 1166], [167, 1166]]],  # Углы маркера с ID 100
#     101: [[[1267, 167], [2266, 167], [2266, 1166], [1267, 1166]]], # Углы маркера с ID 101
#     102: [[[167, 1267], [1166, 1267], [1166, 2266], [167, 2266]]], # Углы маркера с ID 102
#     103: [[[1267, 1267], [2266, 1267], [2266, 2266], [1267, 2266]]] # Углы маркера с ID 103
# }
# Сопоставление маркеров с угловыми точками: В массиве corners хранятся угловые точки каждого обнаруженного маркера.
# Мы извлекаем точки для каждого маркера:
#
# Эти точки указывают на область на основном изображении, куда будет вставлено изображение image.jpg.
#
# Маркер 100: Верхний левый угол (corners[100][0][0])
# Маркер 101: Верхний правый угол (corners[101][0][1])
# Маркер 102: Нижний правый угол (corners[102][0][2])
# Маркер 103: Нижний левый угол (corners[103][0][3])
# Определение углов изображения для вставки: Массив src_points задает углы изображения для вставки (image.jpg).
# Порядок точек важен: верхний левый, верхний правый, нижний правый, нижний левый.
#
# Вычисление матрицы перспективного преобразования: cv2.getPerspectiveTransform вычисляет матрицу преобразования,
# которая "растягивает" изображение вставки так, чтобы его углы совпали с заданными угловыми точками на основном изображении.
#
# Применение перспективного преобразования: Функция cv2.warpPerspective использует матрицу для трансформации изображения.
# Результат сохраняется как warped_insert.jpg.
#
# Создание маски и инвертированной маски: Создаем черно-белую маску, где белым цветом отмечена область вставки.
# Затем инвертируем маску (mask_inv.jpg), чтобы выделить область вне вставки.


# =================================================================================================================
# Вставка изображения по ArUco-маркерам в видео
# В предыдущих шагах мы научились обнаруживать ArUco-маркеры на изображениях и использовать их для перспективной
# трансформации и вставки других изображений в заданные области. Теперь пришло время применить эти знания к видео.
# В этом шаге мы рассмотрим, как обработать видео, вставив изображение в каждый кадр, используя те же методы,
# что и для статичных изображений.
#
# Зачем обрабатывать видео по кадрам?
# Видео — это последовательность изображений (кадров), воспроизводимых с определенной скоростью (обычно 24–30
# кадров в секунду). Чтобы применить рассмотренные ранее алгоритмы к видео, нам нужно обработать каждый
# кадр так же, как мы обрабатывали отдельное изображение. Это позволяет создавать впечатляющие эффекты
# дополненной реальности, накладывая изображения или 3D-объекты на видео в реальном времени.
# В этом курсе уже рассматривалось обработка видео, а также создание видео-файлов MP4 и GIF форматов
#
# Шаг 1: Подготовка необходимых файлов
# Перед началом убедитесь, что у вас есть:
#
# Видео с ArUco-маркерами: видеофайл 00.mp4, на котором по углам области присутствуют ArUco-маркеры с ID 100, 101, 102 и 103.
# Изображение для вставки: файл image.jpg, который вы хотите вставить в видео. Вы можете использовать предложенное
# изображение из предыдущих глав или создать свое собственное, например, с помощью генеративной сети Kandinsky.
#  Шаг 2: Пишем код
# -----------------------

import cv2
import numpy as np

# Загрузка видеофайла и изображения для вставки
cap = cv2.VideoCapture('00.mp4')
insert_image = cv2.imread('image.jpg')
if not cap.isOpened() or insert_image is None:
    raise ValueError("Ошибка загрузки видео или изображения для вставки")

# Параметры для записи выходного видео
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('result.mp4', fourcc, fps, (width, height))

# Настройка словаря и параметров детектора ArUco
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
detector_params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params)

# Параметры для гомографии
h, w = insert_image.shape[:2]
src_points = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")

# Чтение и обработка каждого кадра
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Обнаружение маркеров на текущем кадре
    corners, ids, _ = detector.detectMarkers(frame)

    # Проверка, что все маркеры обнаружены
    required_ids = {100, 101, 102, 103}
    if ids is not None and required_ids.issubset(set(ids.flatten())):
        marker_corners = {id[0]: corner for id, corner in zip(ids, corners)}

        # Определяем угловые точки по ID маркеров
        dst_points = np.array([
            marker_corners[100][0][0],  # Точка 0 - верхний левый угол (Маркер 100)
            marker_corners[101][0][1],  # Точка 1 - верхний правый угол (Маркер 101)
            marker_corners[102][0][2],  # Точка 2 - нижний правый угол (Маркер 102)
            marker_corners[103][0][3]  # Точка 3 - нижний левый угол (Маркер 103)
        ], dtype="float32")

        # Вычисляем матрицу перспективного преобразования и деформируем изображение
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_insert = cv2.warpPerspective(insert_image, matrix, (width, height))

        # Создаем маску и инвертируем ее
        mask = np.zeros_like(frame)
        cv2.fillPoly(mask, [np.int32(dst_points)], (255, 255, 255))
        mask_inv = cv2.bitwise_not(mask)

        # Удаляем область вставки и совмещаем кадр с деформированным изображением
        frame_bg = cv2.bitwise_and(frame, mask_inv)
        frame_result = cv2.add(frame_bg, warped_insert)
    else:
        # Если маркеры не найдены, оставляем исходный кадр
        frame_result = frame

    # Запись кадра в выходной файл
    out.write(frame_result)

# Освобождаем ресурсы
cap.release()
out.release()
print("Видео успешно сохранено как result.mp4")

# Шаг 3: Разбор ключевых моментов
# Обработка видео как последовательности изображений
# Основной принцип обработки видео заключается в том, что мы обрабатываем его кадр за кадром. Каждый кадр — это обычное
# изображение, к которому мы можем применять все изученные ранее методы.
#
# Обнаружение ArUco-маркеров в видео
# Поскольку в видео могут меняться условия освещения, угол съемки и другие факторы, важно обеспечить надежное обнаружение
# маркеров. Если маркеры временно пропали из кадра, мы обрабатываем этот случай, сохраняя исходный кадр без изменений.
#
# Перспективное преобразование в реальном времени
# Для каждого кадра мы вычисляем новую матрицу перспективного преобразования, поскольку положение маркеров может
# меняться. Это позволяет корректно деформировать изображение для вставки в зависимости от положения области на кадре.
# В этом случае важным будет оценить производительность на конечном устройстве!
#
# Создание маски и наложение изображений
# Используя маски, мы аккуратно удаляем область вставки с текущего кадра и накладываем деформированное изображение,
# сохраняя при этом остальные части кадра без изменений.
#
# Возможные улучшения
# Предобработка кадров: Если обнаружение маркеров нестабильно, можно добавить предобработку кадров (например,
# улучшение контраста или шумоподавление) перед детектированием.
# Интерполяция позиций маркеров: Если маркеры временно не обнаружены, можно интерполировать их положение на основе
# предыдущих кадров для плавности эффекта.
# Оптимизация производительности: Для ускорения обработки можно использовать многопоточность или обрабатывать кадры
# с пониженным разрешением.
# В этом уроке мы расширили навыки работы с изображениями на обработку видео. Мы научились применять те же методы
# обнаружения маркеров и перспективной трансформации для каждого кадра видео, создавая впечатляющие эффекты дополненной
# реальности. Теперь вы можете экспериментировать с различными видео и изображениями для вставки, создавая
# свои собственные проекты и углубляя понимание компьютерного зрения. А также реализовать проект с веб-камерой!