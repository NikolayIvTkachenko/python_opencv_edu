# Исследование цветовых моделей
# В этом уроке мы изучим различные цветовые модели, доступные в OpenCV, и научимся их использовать
# для обработки изображений. Мы также рассмотрим, какие задачи лучше всего решаются с помощью каждой из цветовых моделей.
#
# Цветовые модели RGB и BGR
# RGB (Red, Green, Blue): Это стандартное цветовое пространство, которое используется большинством библиотек
# визуализации, таких как Matplotlib. Оно более интуитивно понятно, так как соответствует порядку восприятия цветов человеком.
#
# BGR (Blue, Green, Red): Это стандартное цветовое пространство, используемое в OpenCV. В отличие
# от более привычного формата RGB, где порядок цветов — красный, зеленый, синий, в BGR порядок — синий, зеленый, красный.
# Этот формат используется многими камерами и устройствами для захвата изображений.
#
# Для преобразования из одного формата в другой в OpenCV применяется функция cv2.cvtColor(), которая
# используется для преобразования цветового пространства изображения. Она позволяет конвертировать изображения
# из одного цветового формата в другой, что часто необходимо при работе с изображениями, поскольку разные библиотеки
# и устройства могут использовать разные цветовые модели.
#
# Рассмотрим строку:
#
# image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#
# Аргументы функции:
#
# image_bgr: Это входное изображение, которое представлено в формате BGR.
# cv2.COLOR_BGR2RGB: Это флаг, указывающий, какое преобразование цветового пространства нужно выполнить.
# В данном случае он указывает на преобразование из BGR в RGB.
# Преобразования:
#
# Каждый пиксель изображения преобразуется из BGR в RGB.
# В случае именно смены форматов RGB и BGR для каждого пикселя меняются местами значения синего и красного
# каналов. Например, пиксель с BGR значениями (255, 0, 0) (чистый синий) преобразуется в RGB (0, 0, 255) (чистый красный).
# Для примера создадим изображение с градиентом:



# import cv2
# import numpy as np
#
# height, width = 300, 300
# image_gradient = np.zeros((height, width, 3), dtype=np.uint8)
#
# # Заполнение изображения градиентом
# for i in range(height):
#     for j in range(width):
#         # Вычисление цвета для каждого пикселя
#         blue = int(255 * (1 - i / height) * (1 - j / width))
#         green = int(255 * (i / height) * (1 - j / width))
#         red = int(255 * (j / width) * (i / height))
#
#         image_gradient[i, j] = (blue, green, red)
#
# cv2.imwrite('gradientBGR.jpg', image_gradient)
# image_rgb = cv2.cvtColor(image_gradient, cv2.COLOR_BGR2RGB)
# cv2.imwrite('gradientRGB.jpg', image_rgb)



# Grayscale (оттенки серого)
# Grayscale (оттенки серого) — это цветовое пространство, в котором каждый пиксель представляется одним
# числовым значением интенсивности света, без информации о цвете. Изображение в градациях серого состоит
# только из серых тонов, варьирующихся от черного до белого. В таких изображениях нет разделения на каналы
# по цветам (красный, зеленый, синий); в простейшем случае у нас есть лишь один канал яркости.
#
# Пример:
#
# Цветное изображение (BGR или RGB) обычно представляется в виде трехмерного массива H x W x 3, где:
#
# H — высота изображения (количество строк пикселей),
# W — ширина изображения (количество столбцов пикселей),
# 3 — количество цветовых каналов.
# Каждый пиксель задается тремя значениями: (B, G, R) или (R, G, B) в зависимости от используемой цветовой модели.
# Градации серого (Grayscale) обычно представляются в виде двумерного массива H x W, где у каждого пикселя
# есть только одно значение интенсивности. В этом случае формально мы имеем один канал.
#
# Преимущества использования Grayscale:
# Упрощение обработки: Снижает вычислительную сложность обработки изображений, так как нет необходимости
# учитывать цветовую информацию. В цветных изображениях, каждый пиксель имеет три канала (красный, зеленый и синий).
# Однако в изображении в градациях серого каждый пиксель представлен только одним значением интенсивности. Это приводит
# к уменьшению объема данных примерно в три раза.
# Повышенная скорость: Переход от трех каналов к одному означает, что изображение занимает меньше места в памяти,
# что может быть важным для обработки больших объемов данных или встраивания в системы с ограниченными ресурсами.
# Фокус на форме и текстуре: Исключение цвета помогает сфокусироваться на геометрии, контуре, текстуре и яркостных
# переходах в изображении. Это бывает важно при решении задач распознавания объектов, где цвет может быть менее
# значимым или даже мешать.
# Преобразование RGB или BGR в Grayscale
# Преобразование изображения из цветного в градации серого обычно происходит путем "взвешивания" каждого
# канала в соответствии с его вкладом в восприятие яркости.

# Преобразование из RGB или BGR в Grayscale:
#
# Преобразование обычно осуществляется путем взвешенного суммирования каналов с коэффициентами, отражающими человеческое восприятие яркости разных цветов. Наиболее распространенная формула:
#
# Y
# =
# 0.299
# ⋅
# R
# +
# 0.587
# ⋅
# G
# +
# 0.114
# ⋅
# B
# Y=0.299⋅R+0.587⋅G+0.114⋅B
#
# Где:
#
# Y — яркость (интенсивность пикселя в градациях серого).
# R,G,B — значения красного, зеленого и синего каналов соответственно.
# Эти коэффициенты подобраны таким образом, чтобы результат выглядел для человеческого глаза максимально естественно с точки зрения яркостного восприятия.
#
# Пример кода для преобразования в Grayscale:

import cv2

# Загрузка цветного изображения
image_bgr = cv2.imread('image (1).png')

# Преобразование в оттенки серого
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

# Сохранение и отображение изображения
cv2.imwrite('image_gray.jpg', image_gray)

# Структура данных для изображений:
#
# Цветное изображение (BGR): трехмерный NumPy-массив размера (H, W, 3).
# Пример: image.shape может быть (480, 640, 3), где 480 — высота, 640 — ширина, 3 — количество цветовых каналов.
#
# Градации серого (Grayscale): двумерный NumPy-массив размера (H, W).
# Пример: image_gray.shape может быть (480, 640). Каждый элемент — интенсивность от 0 до 255 (для 8-битных изображений).
#
# Однако, grayscale-изображение можно представить и в виде трехмерного массива, если вам это
# нужно для унификации с другими операциями. Например, можно "соединить трижды" однослойный
# grayscale-канал по оси каналов, чтобы получить (H, W, 3),  сделав каждый канал одинаковым.
# Это часто делается, когда алгоритмы или библиотеки ожидают на вход "цветной" формат, а у вас есть только черно-белые данные.
#
# Есть два удобных способа:
# 1-Посредством NumPy:
import numpy as np
import cv2

img = cv2.imread('image (1).png')   # BGR, размер (H, W, 3)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # теперь размер (H, W)

# Преобразование двумерного массива серых оттенков в "3-канальное" изображение:
gray_3ch = np.stack((gray, gray, gray), axis=2)
# gray_3ch теперь (H, W, 3), каждый канал одинаковый.

# 2-Посредством OpenCV и cvtColor():
# import cv2
#
# img = cv2.imread('image (1).png')   # BGR, размер (H, W, 3)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # теперь размер (H, W)
#
# # Преобразование Gray в BGR
# gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
# # Теперь gray_3ch — (H, W, 3), каждый канал одинаков.

# Как определить по массиву, что это RGB/BGR или Grayscale?
#
# Проверьте размерность массива (image.shape):
#
# Если у массива len(image.shape) == 3 и последний размер равен 3, то, скорее всего, это цветное изображение
# в формате (H, W, 3).
# Если len(image.shape) == 2, то это почти наверняка Grayscale (или отдельный канал цветного изображения).
# Если у вас 3-канальное изображение, но все каналы идентичны (например, np.all(image[:,:,0] == image[:,:,1])
# и то же для третьего канала), то по сути это grayscale, просто представленное в трех каналах.
# Однако, этот метод — эвристический. Он не всегда верен (возможно, у вас изображение действительно имеет
# одинаковые значения каналов по всей области).
#
# При чтении файлов в OpenCV можно сразу указывать необходимый формат:
#
# cv2.imread('path.jpg', cv2.IMREAD_GRAYSCALE) вернет 2D массив.
# cv2.imread('path.jpg', cv2.IMREAD_COLOR) вернет 3D массив (H, W, 3).

#  Комбинирование цветных и серых изображений:
#
# Когда нужно соединить вместе (например, горизонтально или вертикально) два изображения: одно цветное,
# а другое — градации серого. Проблема в том, что цветное изображение — трехканальное, а серое — одноканальное.
# Их нельзя просто склеить без приведения их к единому формату.
#
# Если вы хотите получить комбинированное изображение, нужно привести оба изображения к одному формату:
#
# Либо сконвертировать цветное изображение в оттенки серого (тогда оба будут 2D и их можно склеивать как 2D массивы).
# Либо наоборот, преобразовать grayscale в 3D, продублировав канал, чтобы получить (H, W, 3). Тогда
# можно соединить цветное и "псевдо-цветное" grayscale-изображение по ширине или высоте.
# Способы соединения изображений:

# Через срезы (индексацию массива)
# Если размеры совпадают по высоте (H) и вы хотите просто поставить изображение серым слева,
# а цветное справа, можно создать пустой массив нужного размера и "вписать" два изображения в разные области массива.
# Например, если у нас два изображения одинаковой высоты H и ширин W1 и W2, мы можем сделать так:
#
# H, W1 = gray_3ch.shape[:2]
# W2 = color_img.shape[1]
#
# combined = np.zeros((H, W1+W2, 3), dtype=gray_3ch.dtype)
# combined[:, :W1] = gray_3ch
# combined[:, W1:] = color_img
# Через np.hstack() или np.vstack()
# Если размеры совпадают по измерению, вдоль которого соединяем, можно воспользоваться функциями NumPy
# для горизонтального или вертикального "стекания":
#
# Горизонтальное соединение (по ширине):
# combined = np.hstack((gray_3ch, color_img))
#
# Вертикальное соединение (по высоте):
# combined = np.vstack((gray_3ch, color_img))
#
# Аналогично можно использовать np.concatenate() с указанием оси:
# combined = np.concatenate((gray_3ch, color_img), axis=1) # по ширине
#
# Через OpenCV-функции cv2.hconcat() и cv2.vconcat()
# OpenCV предлагает удобные функции для горизонтального и вертикального объединения:
#
# Горизонтальное объединение:
# combined = cv2.hconcat([gray_3ch, color_img])
#
# Вертикальное объединение:
# combined = cv2.vconcat([gray_3ch, color_img])

#-----------------------------------------------------
# Теперь решим задачу несколько сложнее, чем на предыдущих уроках. Поле достаточно большое, и в нем нужно найти объект!
#
# Прежде чем начать обработку изображения, необходимо преобразовать его в градации серого. Этот шаг уже знаком вам:
#
# img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# Бинаризация изображения
# После преобразования изображения в градации серого мы можем применить бинаризацию. Бинаризация — это процесс преобразования градаций серого в черно-белое изображение. Для этого можно использовать функцию np.where для округления значений до 0 или 255 в зависимости от порога:
#
# img_bin = np.where(img_gray > 15, 255, 0).astype(np.uint8)
#
# Этот код устанавливает все значения пикселей выше 15 на 255 (белый), а все значения ниже или равные 15 — на 0 (черный).
#
# Также возможно применить команду:
#
# img_bin = cv2.inRange(img_gray, 15, 255)
#
# Которая делает аналогичную операцию методами OpenCV.

# Нахождение контуров
# Для нахождения контуров на бинаризованном изображении используется функция cv2.findContours:
#
# _, cont, h = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# img_bin — входное бинаризованное изображение.
# cv2.RETR_EXTERNAL — режим извлечения, который извлекает только внешние контуры.
# cv2.CHAIN_APPROX_SIMPLE — метод аппроксимации контуров, который удаляет все избыточные точки и сжимает контур, сохраняя его основную структуру.
# Функция возвращает три значения для OpenCV 3 версии:
#
# _ — изображение с нарисованными контурами (не используется в нашей задаче)).
# cont — список найденных контуров.
# h — иерархия контуров (не используется в нашей задаче).
# Примечание: для 4х версий OpenCV cv2.findContours возвращает два значения.
# В проверяющей системе Stepik, при выборе Python 3.10 используется 4 версия OpenCV
#
# cont, h = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
# cont — список найденных контуров.
# h — иерархия контуров (не используется).
# Нахождение прямоугольника, ограничивающего объект
# Для нахождения ограничивающего прямоугольника вокруг первого (и в нашем случае, единственного) найденного контура используется функция cv2.boundingRect:
#
# x, y, w, h = cv2.boundingRect(cont[0])
# cont[0] — первый найденный контур.
# x, y — координаты верхнего левого угла прямоугольника.
# w, h — ширина и высота прямоугольника.
# Вычисление координат центра объекта
# Координаты центра объекта вычисляются как средние значения координат прямоугольника:
#
# center_x = x + w // 2
# center_y = y + h // 2

#------------
# Преобразование между RGB и HSV в OpenCV
# OpenCV предоставляет функции для преобразования между RGB и HSV цветовыми моделями:
#
# RGB -> HSV:
# import cv2
#
# image = cv2.imread('path_to_image.jpg')
#
# # Преобразование из BGR в HSV
# hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
# HSV -> RGB:
# import cv2
#
#
# image = cv2.imread('path_to_image.jpg')
#
# # Преобразование из HSV в BGR
# bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# import cv2
# import numpy as np
#
# cv2.namedWindow("result")
# cv2.namedWindow("settings")
#
# cv2.createTrackbar('h1', 'settings', 0, 180, lambda x: x)
# cv2.createTrackbar('s1', 'settings', 0, 255, lambda x: x)
# cv2.createTrackbar('v1', 'settings', 0, 255, lambda x: x)
# cv2.createTrackbar('h2', 'settings', 180, 180, lambda x: x)
# cv2.createTrackbar('s2', 'settings', 255, 255, lambda x: x)
# cv2.createTrackbar('v2', 'settings', 255, 255, lambda x: x)
#
# while True:
#     img = cv2.imread('duck.png')
#     h, w, _ = img.shape
#     img = cv2.resize(img, (w // 5, h // 5))
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     # считываем значения бегунков
#     h1 = cv2.getTrackbarPos('h1', 'settings')
#     s1 = cv2.getTrackbarPos('s1', 'settings')
#     v1 = cv2.getTrackbarPos('v1', 'settings')
#     h2 = cv2.getTrackbarPos('h2', 'settings')
#     s2 = cv2.getTrackbarPos('s2', 'settings')
#     v2 = cv2.getTrackbarPos('v2', 'settings')
#     h_min = np.array((h1, s1, v1), np.uint8)
#     h_max = np.array((h2, s2, v2), np.uint8)
#     img_bin = cv2.inRange(hsv, h_min, h_max)
#     cv2.imshow('result', img_bin)
#     cv2.imshow('original', img)
#     ch = cv2.waitKey(5)
#     if ch == 27:
#         break
# cv2.destroyAllWindows()


# Как мы видим, уточка желтая, но за счет теней желтый цвет достаточно "разнообразный"!
#
# В RGB модели желтый цвет представлен высокой интенсивностью красного и зеленого компонентов. Однако, любые изменения в освещении могут привести к тому, что желтый цвет будет трудно обнаружить.
#
# В HSV можно задать диапазон значений, который точно определяет желтый цвет, и выполнить сегментацию.
#
# В интернете широко распространен скрипт для подбора диапазонов HSV.

import cv2
import numpy as np

# Глобальные переменные для хранения координат области
ref_point = []
cropping = False
selection_done = False
canvas_copy = None


def create_hsv_circle(radius):
    y, x = np.ogrid[-radius:radius, -radius:radius]
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(y, x) * 180 / np.pi
    theta = (theta + 360) % 360

    hsv_circle = np.zeros((radius * 2, radius * 2, 3), dtype=np.uint8)

    mask = r <= radius
    hsv_circle[mask, 0] = (theta[mask] / 2).astype(np.uint8)  # H: 0-179
    hsv_circle[mask, 1] = (r[mask] / radius * 255).astype(np.uint8)  # S: 0-255
    hsv_circle[mask, 2] = 255  # V: 255

    hsv_circle = cv2.cvtColor(hsv_circle, cv2.COLOR_HSV2BGR)
    return hsv_circle


def resize_with_padding(image, target_size):
    h, w = image.shape[:2]
    scale = target_size / max(h, w)
    new_w = int(w * scale)
    new_h = int(h * scale)

    resized_img = cv2.resize(image, (new_w, new_h))
    delta_w = target_size - new_w
    delta_h = target_size - new_h
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_image = cv2.copyMakeBorder(resized_img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_image


def apply_hsv_mask(hsv_circle, h1, s1, v1, h2, s2, v2):
    hsv = cv2.cvtColor(hsv_circle, cv2.COLOR_BGR2HSV)

    mask_h = (hsv[..., 0] >= h1) & (hsv[..., 0] <= h2)
    mask_s = (hsv[..., 1] >= s1) & (hsv[..., 1] <= s2)
    mask_v = (hsv[..., 2] >= v1) & (hsv[..., 2] <= v2)

    combined_mask = mask_h & mask_s & mask_v

    masked_hsv_circle = hsv_circle.copy()
    masked_hsv_circle[~combined_mask] = 0
    return masked_hsv_circle


def click_and_crop(event, x, y, flags, param):
    global ref_point, cropping, selection_done, canvas_copy

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]
        cropping = True

    elif event == cv2.EVENT_MOUSEMOVE:
        if cropping:
            # Создаем копию canvas, чтобы не рисовать новый прямоугольник поверх старого
            canvas_copy = canvas.copy()
            cv2.rectangle(canvas_copy, ref_point[0], (x, y), (0, 255, 0), 2)
            cv2.imshow("HSV Tool", canvas_copy)

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))
        cropping = False
        selection_done = True

        # Рисуем финальный прямоугольник
        cv2.rectangle(canvas, ref_point[0], ref_point[1], (0, 255, 0), 2)
        cv2.imshow("HSV Tool", canvas)


def adjust_hsv_ranges():
    global ref_point

    # Извлечение выделенной области
    x1, y1 = ref_point[0]
    x2, y2 = ref_point[1]

    # Убедимся, что область находится внутри изображения
    x1 = max(0, min(x1, hsv_resized_img.shape[1] - 1))
    x2 = max(0, min(x2, hsv_resized_img.shape[1] - 1))
    y1 = max(0, min(y1, hsv_resized_img.shape[0] - 1))
    y2 = max(0, min(y2, hsv_resized_img.shape[0] - 1))

    if x1 == x2 or y1 == y2:
        return None, None  # Если область нулевая, возвращаем None

    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1

    roi = hsv_resized_img[y1:y2, x1:x2]

    # Поиск минимальных и максимальных значений HSV в области
    h_min, s_min, v_min = np.min(roi[:, :, 0]), np.min(roi[:, :, 1]), np.min(roi[:, :, 2])
    h_max, s_max, v_max = np.max(roi[:, :, 0]), np.max(roi[:, :, 1]), np.max(roi[:, :, 2])

    # Установка трекбаров в соответствующие значения
    cv2.setTrackbarPos('H_min', 'HSV Tool', h_min)
    cv2.setTrackbarPos('S_min', 'HSV Tool', s_min)
    cv2.setTrackbarPos('V_min', 'HSV Tool', v_min)
    cv2.setTrackbarPos('H_max', 'HSV Tool', h_max)
    cv2.setTrackbarPos('S_max', 'HSV Tool', s_max)
    cv2.setTrackbarPos('V_max', 'HSV Tool', v_max)

    return (h_min, s_min, v_min), (h_max, s_max, v_max)


radius = 150  # Радиус круга HSV

cv2.namedWindow("HSV Tool", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HSV Tool", 600, 700)

cv2.createTrackbar('H_min', 'HSV Tool', 0, 180, lambda x: x)
cv2.createTrackbar('S_min', 'HSV Tool', 0, 255, lambda x: x)
cv2.createTrackbar('V_min', 'HSV Tool', 0, 255, lambda x: x)
cv2.createTrackbar('H_max', 'HSV Tool', 180, 180, lambda x: x)
cv2.createTrackbar('S_max', 'HSV Tool', 255, 255, lambda x: x)
cv2.createTrackbar('V_max', 'HSV Tool', 255, 255, lambda x: x)

# Добавление возможности выделения области мышью
cv2.setMouseCallback("HSV Tool", click_and_crop)

hsv_min = hsv_max = None  # Инициализация переменных

while True:
    img = cv2.imread('duck.png')
    resized_img = resize_with_padding(img, 400)
    hsv_resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)

    h1 = cv2.getTrackbarPos('H_min', 'HSV Tool')
    s1 = cv2.getTrackbarPos('S_min', 'HSV Tool')
    v1 = cv2.getTrackbarPos('V_min', 'HSV Tool')
    h2 = cv2.getTrackbarPos('H_max', 'HSV Tool')
    s2 = cv2.getTrackbarPos('S_max', 'HSV Tool')
    v2 = cv2.getTrackbarPos('V_max', 'HSV Tool')
    h_min = np.array((h1, s1, v1), np.uint8)
    h_max = np.array((h2, s2, v2), np.uint8)

    img_bin = cv2.inRange(hsv_resized_img, h_min, h_max)

    # Подготовка основного изображения 700x800
    canvas = np.zeros((700, 800, 3), dtype=np.uint8)

    # Обновление круга HSV с учетом трекбаров
    hsv_circle = create_hsv_circle(radius)
    masked_hsv_circle = apply_hsv_mask(hsv_circle, h1, s1, v1, h2, s2, 255)

    # Размещение изображений на канвасе
    canvas[:400, :400] = resized_img
    canvas[:400, 400:800] = cv2.cvtColor(img_bin, cv2.COLOR_GRAY2BGR)
    canvas[400:700, :300] = masked_hsv_circle

    # Если выделение завершено, автоматически подобрать диапазоны цветов
    if selection_done:
        hsv_min, hsv_max = adjust_hsv_ranges()
        selection_done = False

    text = f"HSV_min: {h_min}\nHSV_max: {h_max}"

    y0, dy = 450, 50
    for i, line in enumerate(text.split('\n')):
        y = y0 + i * dy
        cv2.putText(canvas, line, (320, y), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Если область выделяется, показать промежуточный результат
    if cropping and canvas_copy is not None:
        cv2.imshow("HSV Tool", canvas_copy)
    else:
        cv2.imshow("HSV Tool", canvas)

    ch = cv2.waitKey(5)
    if ch == 27:
        break

cv2.destroyAllWindows()