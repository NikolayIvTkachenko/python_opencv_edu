# import cv2
# import numpy as np
#
# img = cv2.imread('car.png')
# #cv2.imshow('Car image',img) #'Car image' - заголовок окна, img - переменная, содержащая изображение
#
#
# #cv2.waitKey(0) #ожидает нажатия любой клавиши
# #cv2.destroyAllWindows() #закроет окно перед завершением скрипта
#
# img = cv2.imread('car.png')
#
# if img is not None:
#     print(f"Прочитано изображение размером {img.shape}")
#     print(f"Ширина - {img.shape[1]} px, высота - {img.shape[0]} px, количество каналов: {img.shape[2]}")
# else:
#     print("Ошибка чтения изображения")

# # import cv2
# # import numpy as np
# #
# #
# # image = read_image(input())
# # print(image)

# import cv2
# import numpy as np
# image = read_image(input())  # читаем изображение
# print(f"Ширина - {image.shape[1]} px, высота - {image.shape[0]} px, количество каналов: {image.shape[2]}")

# import cv2
# import numpy as np
#
# # Загрузка исходного изображения
# image = cv2.imread('car.png')
#
# # Проверка, что изображение загружено
# if image is None:
#     print("Изображение не прочитано")
#     exit()
#
# red_channel = image[:, :, 2]  # Красный канал
# green_channel = image[:, :, 1]  # Зеленый канал
# blue_channel = image[:, :, 0]  # Синий канал
#
# # Сохранение изображений
# cv2.imwrite('red_channel.jpg', red_channel)
# cv2.imwrite('green_channel.jpg', green_channel)
# cv2.imwrite('blue_channel.jpg', blue_channel)
#
# print("Images saved successfully.")
#
# # Обработка изображений с использованием NumPy
# # При обработке изображений в формате JPG и PNG могут возникать различия из-за их особенностей. Например, изображения JPG могут содержать артефакты из-за сжатия с потерями. В этом шаге мы рассмотрим, как можно обойти эти проблемы с помощью библиотеки NumPy.
# #
# # Округление значений цветов с использованием NumPy
# # Иногда при обработке изображений может потребоваться округлить значения цветов до определенных значений. Рассмотрим команду np.where и её использование для этой цели.
# #
# # Команда np.where используется для выбора элементов из массива на основе условия. Рассмотрим пример, где мы округляем значения цветов в изображении до 0 или 255:
# import cv2
# import numpy as np
#
# # Чтение изображения
# image = cv2.imread('circle.png')
#
# image = np.where(image > 230, 255, 0)
#
# cv2.imwrite('result_image.png', image)
#
# # Все значения выше заданного порога (в данном случае 230) устанавливаются на 255 , а остальные - на 0 .
# #
# # Эту команду можно рассматривать как оператор if, внутри цикла for, примененный к каждому значению каждого пикселя изображения:
#
# import cv2
# import numpy as np
#
# image = cv2.imread('path_to_image.jpg')
#
#
# for i in range(image.shape[0]):
#     for j in range(image.shape[1]):
#         for k in range(image.shape[2]):
#              if image[i,j,k] > 230:
#                  image[i,j,k] = 255
#              else:
#                  image[i,j,k] = 0
#
#
# cv2.imwrite('result_image.png', image)

# image = np.where(image > 230, 255, image)


# import cv2
#
# # Чтение изображения
# image = cv2.imread('car.png')
#
# # Преобразование в градации серого
# gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
# # Сохранение результата
# cv2.imwrite('gray_image.jpg', gray_image)
#
# # Отображение результата
# cv2.imshow('Gray Image', gray_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2

image = cv2.imread('.venv/car.png')

# Преобразование из BGR в HSV
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

import cv2


image = cv2.imread('.venv/car.png')

# Преобразование из HSV в BGR
bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
