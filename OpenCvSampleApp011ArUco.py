# import cv2
# import numpy as np
#
# # Инициализация словаря маркеров
# marker_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
#
# # Создание параметров детектирования
# detector_params = cv2.aruco.DetectorParameters()
#
# # Инициализация детектора ArUco
# detector = cv2.aruco.ArucoDetector(marker_dict, detector_params)
#
# # Детектирование маркеров на изображении
# corners, ids, rejectedImgPoints = detector.detectMarkers(image)

# corners: список координат углов каждого обнаруженного маркера.
# ids: массив идентификаторов (ID) обнаруженных маркеров.
# rejectedImgPoints: дополнительные выходные данные, которые мы не используем.

# Шаг 2: Проверка наличия маркеров на изображении
# После попытки детектирования маркеров необходимо проверить, были ли они обнаружены.
#
# Код для проверки:
#
# if ids is None or len(ids) == 0:
#     print("Маркеры не обнаружены.")
# else:
#     # Продолжаем обработку
# Если ids равно None или является пустым, значит, маркеры не были обнаружены на изображении.

# Шаг 3: Поиск маркера с заданным ID среди обнаруженных маркеров
# Если маркеры обнаружены, нужно проверить, есть ли среди них маркер с нашим заданным ID.
#
# Когда мы используем функцию detectMarkers из модуля cv2.aruco, она возвращает массив ids, который содержит идентификаторы обнаруженных маркеров.
#
# Однако важно отметить, что структура массива ids может быть не одномерной, а двумерной. Давайте посмотрим, как выглядит ids после детектирования.
#
# Пример:
#
# Предположим, что мы обнаружили три маркера с идентификаторами 5, 10 и 15. Тогда массив ids может выглядеть так:
#
# ids = array([[ 5],
#              [10],
#              [15]], dtype=int32)
# Это двумерный массив, где каждая строка содержит один элемент — идентификатор маркера. То есть ids имеет размерность (N, 1), где N — количество обнаруженных маркеров.
#
# Почему нужно преобразовать ids в одномерный массив
# Для удобства обработки и поиска маркеров по их идентификаторам нам удобнее работать с одномерным массивом или списком, где каждый элемент — это идентификатор маркера.
#
# Проблема с двумерным массивом:
#
# Когда ids является двумерным массивом, прямое сравнение с числом может не работать так, как ожидается.
# Например, если мы попытаемся проверить, содержится ли заданный marker_id в ids с помощью if marker_id in ids:, то это может не сработать, потому что ids содержит вложенные массивы, а не непосредственные значении, т.к. использовании if marker_id in ids: сравнение происходит между числом и массивом, что приводит к некорректным результатам.
# Преобразование в одномерный массив:
#
# Преобразовав ids в одномерный массив, мы получаем простой список идентификаторов, с которым удобно работать.
# Это позволяет нам использовать стандартные операции, такие как проверка на вхождение (in), поиск индекса и т.д.


# 3. Как работает метод flatten()
# Метод flatten() из библиотеки NumPy преобразует многомерный массив в одномерный, то есть «сплющивает» его.
#
# import numpy as np
#
# # Изначальный двумерный массив
# ids = np.array([[5],
#                 [10],
#                 [15]], dtype=int32)
#
# print("До flatten:")
# print(ids)
# print("Форма массива:", ids.shape)
#
# # Применяем flatten()
# ids_flat = ids.flatten()
#
# print("\nПосле flatten:")
# print(ids_flat)
# print("Форма массива:", ids_flat.shape)

# Пояснение:
#
# До flatten(): ids — это двумерный массив с формой (3, 1), то есть 3 строки и 1 столбец.
# После flatten(): ids_flat — это одномерный массив с формой (3,), содержащий элементы [5, 10, 15].

# 4. Применение в контексте поиска маркера с заданным ID
# Теперь, когда ids является одномерным массивом, мы можем легко проверить, содержится ли заданный marker_id в массиве ids, и найти его индекс.
#
# Код без flatten():
#
# marker_id = 10
#
# if marker_id in ids:
#     print("Маркер найден")
# else:
#     print("Маркер не найден")
# Однако это может не сработать, потому что ids — это двумерный массив, и сравнение происходит с вложенными массивами, а не с числами.
#
# Код с flatten():
#
# ids = ids.flatten()
#
# if marker_id in ids:
#     print("Маркер найден")
# else:
#     print("Маркер не найден")
# Теперь сравнение работает корректно, потому что ids — одномерный массив чисел.


# Наш план по поиску выглядит так:
# Преобразовать массив ids в одномерный для удобства обработки.
# Проверить, присутствует ли наш marker_id в массиве ids.
# Если да, получить индекс этого маркера для доступа к его углам.
# Код:
#
# # Преобразуем ids в одномерный массив
# ids = ids.flatten()
#
# # Проверяем, есть ли заданный ID среди обнаруженных
# if marker_id in ids:
#     # Находим индекс маркера с заданным ID
#     index = np.where(ids == marker_id)[0][0]
#     # Продолжаем обработку
# else:
#     print("Маркеры не обнаружены.")
# ids.flatten(): преобразует массив ids в одномерный массив.
# np.where(ids == marker_id): возвращает индексы элементов, равных marker_id.
# [0][0]: извлекаем первый индекс из результата np.where.
# Дополнительные [0][0] нужны, чтобы из структуры, которую возвращает np.where извлечь индекс элемента:
#
# # Предположим, ids = [5, 10, 15]
# marker_id = 10
#
# # Находим индексы элементов, равных marker_id
# indices = np.where(ids == marker_id)
#
# print("Индексы маркера с ID", marker_id, ":", indices)
# Вывод:
#
# Индексы маркера с ID 10 : (array([1], dtype=int64),)
# np.where(ids == marker_id) возвращает кортеж, где первый элемент — массив индексов элементов, удовлетворяющих условию ids == marker_id.
# В данном случае, индекс маркера с ID 10 равен 1.
# Шаг 4: Извлечение координат углов маркера с заданным ID
# Зная индекс маркера с нужным ID, можем получить его координаты углов из массива corners.
#
# Код:
#
# # Получаем координаты углов маркера с заданным ID
# marker_corners = corners[index][0]  # Размерность (4, 2)
# corners[index][0]: выбираем первый (и единственный) элемент внутреннего массива углов для маркера с заданным индексом.
#
# Структура marker_corners:
#
# [
#     [x0, y0],  # Точка 0: верхний левый угол
#     [x1, y1],  # Точка 1: верхний правый угол
#     [x2, y2],  # Точка 2: нижний правый угол
#     [x3, y3]   # Точка 3: нижний левый угол
# ]

#=====================================
# Первым делом научимся создавать маркеры:
# Рассмотрим следующий код:
#
# import cv2
# import numpy as np
#
#
# arucoDict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
# markerId = 17
# imgSize = 500
#
# markerImage = np.zeros((imgSize, imgSize), dtype=np.uint8)
# cv2.aruco.generateImageMarker(arucoDict, markerId, imgSize, markerImage, 1)
#
# cv2.imwrite(f"aruco_marker_{markerId}.png", markerImage)
# arucoDict: мы выбираем предопределённый словарь маркеров. В данном случае используется словарь DICT_4X4_250,
# который содержит 250 различных маркеров размером 4x4.
# markerId: идентификатор маркера, который мы хотим создать. Он должен быть в диапазоне от 0 до 249 для этого словаря.
# imgSize: размер изображения маркера в пикселях. Здесь мы устанавливаем размер 500x500 пикселей, одно число указываем,
# т.к. маркеры квадратные.
# Создание пустого изображения для маркера:
#
# markerImage = np.zeros((imgSize, imgSize), dtype=np.uint8)
# Создаём пустой двумерный массив (изображение) размером imgSize x imgSize, заполненный нулями (чёрный цвет),
# с типом данных uint8 (целые числа от 0 до 255). Этот массив будет служить холстом для нашего маркера.

# Генерация маркера:
#
# cv2.aruco.generateImageMarker(arucoDict, markerId, imgSize, markerImage, 1)
# Здесь мы вызываем функцию generateImageMarker, которая создаёт изображение маркера и помещает его в markerImage.
#
# Параметры функции:
#
# arucoDict: словарь маркеров, выбранный ранее.
# markerId: ID маркера, который мы хотим сгенерировать.
# imgSize: размер изображения в пикселях.
# markerImage: массив, куда будет записан сгенерированный маркер.
# 1: толщина границы маркера в битах.
# Сохранение изображения маркера:
#
# cv2.imwrite(f"aruco_marker_{markerId}.png", markerImage)
# В результате получим вот такое изображение:


# Однако, такой маркер еще не готов к внедрению в массы! К нему нужно добавить белый бордюр (контур)!
#
# Этот контур играет важную роль в повышении надёжности и точности обнаружения маркера.
#
# Почему нужен белый бордюр вокруг ArUco-маркера?
#
# Улучшение распознавания: Белый контур создаёт чёткий контраст между чёрными элементами маркера и окружающей средой,
# что облегчает его обнаружение алгоритмами компьютерного зрения.
#
# Тихая зона: Бордюр служит своего рода "буфером" или "тихой зоной" вокруг маркера, свободной от лишних деталей или шума,
# которые могут помешать правильному распознаванию.
#
# Устойчивость к освещению: Дополнительная белая область помогает компенсировать перепады освещённости и тени,
# обеспечивая стабильное распознавание маркера в различных условиях.
#
# Соответствие стандартам: Некоторые алгоритмы и стандарты требуют наличия белого бордюра вокруг маркера
# для корректной работы и повышения точности обнаружения.
#
# Оформим все в виде функции, которая сразу будет формировать маркер с заданными характеристиками,
# и даже пунктирной рамкой (чтобы удобно было вырезать).


# import cv2
# import numpy as np
#
#
# def createArucoMarkerWithBorder(markerSize, markerId, imgSize):
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
#     thickness = 1 * int(imgSize / 500)
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
#     # Сохраняем маркер в файл
#     cv2.imwrite(f"aruco_marker_{markerId}_{markerSize}x{markerSize}.png", newImage)
#
#
# # Пример использования функции
# createArucoMarkerWithBorder(4, 10,
#                             350)  # Создает маркер 4x4 с ID 2, размером изображения 200x200 пикселей и белой рамкой
#
#



# о и это еще не все!
#
# Для удобной печати большого количества маркеров воспользуемся библиотекой fpdf, и сделаем универсальное
# решение, позволяющее генерировать маркеры быстро и просто!
import cv2
import numpy as np
import os
from fpdf import FPDF


def createArucoMarkerWithBorder(markerSize, markerId, imgSize):
    # Создаем словарь Aruco маркеров нужного размера
    dict_sizes = {
        4: cv2.aruco.DICT_4X4_250,
        5: cv2.aruco.DICT_5X5_250,
        6: cv2.aruco.DICT_6X6_250,
        7: cv2.aruco.DICT_7X7_250
    }

    if markerSize in dict_sizes:
        arucoDict = cv2.aruco.getPredefinedDictionary(dict_sizes[markerSize])
    else:
        raise ValueError("Неподдерживаемый размер маркера. Поддерживаемые размеры: 4, 5, 6, 7.")

    # Генерируем маркер
    markerImage = np.zeros((imgSize, imgSize), dtype=np.uint8)
    cv2.aruco.generateImageMarker(arucoDict, markerId, imgSize, markerImage, 1)

    # Рассчитываем размер нового изображения с рамкой
    borderSize = imgSize // (markerSize + 2)
    newSize = imgSize + borderSize * 2 + 2

    # Создаем новое изображение с белой рамкой
    newImage = np.ones((newSize, newSize), dtype=np.uint8) * 255
    newImage[borderSize + 1:-borderSize - 1, borderSize + 1:-borderSize - 1] = markerImage

    # Добавляем пунктирную линию на крайних пикселях рамки
    for i in range(0, newSize, 4):
        newImage[i:i + 2, 0] = 0
        newImage[i:i + 2, -1] = 0
        newImage[0, i:i + 2] = 0
        newImage[-1, i:i + 2] = 0

    # Добавляем текст с ID маркера
    text = f"{markerId}"
    targetTextHeight = imgSize * 0.07  # 7% от высоты изображения
    fontScale = 0.1  # Начальный масштаб шрифта
    thickness = 1 * int(imgSize / 500)
    textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)[0]

    # Подбираем масштаб шрифта, чтобы высота текста была приблизительно 7% от imgSize
    while textSize[1] < targetTextHeight:
        fontScale += 0.1
        textSize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale, thickness)[0]

    textX = newSize - textSize[0] - int(imgSize * 0.02)  # от правого края
    textY = newSize - int(imgSize * 0.02)  # от нижнего края
    cv2.putText(newImage, text, (textX, textY), cv2.FONT_HERSHEY_SIMPLEX, fontScale, (0, 0, 0), thickness)

    # Сохраняем маркер в файл
    # cv2.imwrite(f"aruco_marker_{markerId}_with_border.png", newImage)
    return newImage


def createArucoMarkersPDF(markerList, mmSize):
    # Проверяем и создаем папку для маркеров
    folderName = "ArucoMarkers"
    if not os.path.exists(folderName):
        os.makedirs(folderName)

    # Создаем маркеры и сохраняем их в папку
    for markerSize, markerId, imgSize in markerList:
        markerImage = createArucoMarkerWithBorder(markerSize, markerId, imgSize)
        cv2.imwrite(f"{folderName}/aruco_marker_{markerId}_{markerSize}x{markerSize}.png", markerImage)

    # Создаем PDF
    pdf = FPDF('P', 'mm', 'A4')
    pdf.add_page()
    margin = 15  # Поля в мм
    current_x, current_y = margin, margin
    page_width, page_height = 210, 297  # Размеры страницы A4 в мм

    for markerSize, markerId, imgSize in markerList:
        filePath = f"{folderName}/aruco_marker_{markerId}_{markerSize}x{markerSize}.png"
        if os.path.exists(filePath):
            if current_x + mmSize > page_width - margin:
                current_x = margin
                current_y += mmSize
            if current_y + mmSize > page_height - margin:
                pdf.add_page()
                current_x, current_y = margin, margin
            pdf.image(filePath, x=current_x, y=current_y, w=mmSize, h=mmSize)
            current_x += mmSize
    # Сохраняем PDF
    pdf.output(f"{folderName}/ArucoMarkers.pdf")


marker_dict = 4
marker_resolution = 200
marker_size_mm = 60
marker_list = [(marker_dict, i, marker_resolution) for i in range(1, 10, 3)]
createArucoMarkersPDF(marker_list,
                      marker_size_mm)  # Создает PDF с маркерами 4x4 размером 60 мм. И изображения разрешением 200px*200px
