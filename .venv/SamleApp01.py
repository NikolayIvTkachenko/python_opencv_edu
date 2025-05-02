
























# a=float(input())
# b=float(input())
# c=float(input())
# summ=a+b
# if abs(summ-c) < 0.0000001:
#     print("Результат верный")
# else:
#     print("Это не подходит")
#
# a = 20
# b = 5
# result = a / b
#
# # Проверка с учетом погрешности
# if abs(result - 4) < 0.0000001:
#     print("Результат верный")
# else:
#     print("Результат с погрешностью")
#
#
# a1=float(input())
# print(round(a1))

# a1=int(input())
# a2=float(input())
# a3=int(input())
# ss=3.14*((a1/2)**2)
# piec=ss/(a3+1)
#
# if piec>=a2:
#     print("Всем точно хватит")
# else:
#     print("Придется уменьшить порции")
#
# a1=int(input())
# a2=float(input())
# a3=int(input())
# ss=3.14*((a1/2)**2)
# piec=ss/a3
#
# print(a1)
# print(a2)
# print(a3)
# print(piec)
#
# if piec>=a2:
#     print("Всем точно хватит")
# else:
#     print("Придется уменьшить порции")

# a1=int(input())
# a2=float(input())
# a3=int(input())
# ss=3.14*((a1/2)**2)
# piec=ss/a3
# if piec>=a2:
#     print("Всем точно хватит")
# else:
#     print("Придется уменьшить порции")
#
# a1=float(input())
# a2=float(input())
# sum=round(a1,1)+round(a2,1)
# print(round(sum,1))


# a1=float(input())
# a2=float(input())
# sum=round(a1,1)+round(a2,1)
# print(sum)
#
#
# # Целые числа
# a = 10**16 + 1
# b = 3
#
# # Деление с образованием погрешности
# result = a / b
# print(f"Результат деления: {result:.15f}")
#
# # Проверка с использованием оператора if
# expected_result = 10**16 / 3
# if result == expected_result:
#     print("Результат точный")
# else:
#     print(f"Результат с погрешностью: {result - expected_result:.15f}")
#
# # Изначально переменная целое число
# num = 10
# print(f"Тип переменной num: {type(num)}")  # Вывод: <class 'int'>
#
# # Деление переменной, результат - вещественное число
# num = num / 2
# print(f"Значение переменной num: {num}")
# print(f"Тип переменной num: {type(num)}")  # Вывод: <class 'float'>

#
# animal_type = input("Введите тип животного (млекопитающее, птица): ")
# size = input("Размер животного (крупное, маленькое): ")
# can_fly = input("Умеет ли животное летать (да/нет): ")
#
# # создание переменной-флага
# flag = False
#
# # Проверяем условия и выводим соответствующие сообщения
# if animal_type == "млекопитающее":
#     if size == "крупное":
#         print("Это слон")
#         flag = True
#     elif size == "маленькое":
#         print("Это мышь")
#         flag = True
# elif animal_type == "птица":
#     if can_fly == "да":
#         print("Это воробей")
#         flag = True
#
# # Если ни одно из условий не подошло
# if not flag:
#     print("Неизвестный вид животного")



# legs=int(input())
# par=legs/2
# if legs==8:
#     #print(f"{legs} ног — паук")
#     print(f"паук")
# elif par%2!=0 and 15<=par<=80:
#     print(f"многоножка")
# else:
#     print(f"мы не знаем, что это такое")
#
# x = 25
# if 15 <= x <= 80:
#     print("x находится в пределах от 15 до 80")
# else:
#     print("x не входит в указанный диапазон")
# x = 25
# if 15 <= x and x <= 80:
#     print("x находится в пределах от 15 до 80")
# else:
#     print("x не входит в указанный диапазон")

# a1=input()
# p=a.lower().replace(" ", "")
# res=p[::-1]
# if res==p:
#     print("Эта фраза - палиндром")
# else:
#     print("Ничего подобного")
#
# a1=input()
# p=a1.lower().replace(" ", "")
# res=p[::-1]
# print(res)

# res=int(input())
# a1=int(input())
# a2=int(input())
# a3=int(input())
# a4=int(input())
#
# if res>=a1:
#     print("великолепно")
# elif res<a1 and res >=a2:
#     print("достойно")
# elif res<a2 and res >=a3:
#     print("моглобыбытьилучше")
# elif res<a3 and res>=a4:
#     print("нутакое")
# else:
#     print("нда")

# a1=int(input())
# a2=input()
#
# if a2=="подкрадули":
#     if a1==52 or a1==53:
#         print(f"Подкрадули - лучшее, что может быть, размерчик, кстати мой")
#     else:
#         print(f"Подкрадули - лучшее, что может быть, маловаты, как обычно")
# else:
#     if a1==52 or a1==53:
#         print(f'{a2} - совсем не подходит мне по стилю, размерчик, кстати мой')
#     else:
#         print(f'{a2} - совсем не подходит мне по стилю, маловаты, как обычно')

# a1=input()
# a2=input()
# if (a1=="не_продаю" and a2=="показываю") or (a2=="не_продаю" and a1=="показываю"):
#     print("красивое")
# else:
#     print("грустное")


# a1=input()
# a2=int(input())
# if a1=="сентябрь" and a2 == 3:
#     print("Горят костры рябин")
# else:
#     print("Ждем 3 сентября")
#
# password = "securepassword123"
# user_active = True
# is_admin = False
#
# if (password == "securepassword123" or password == "admin123") and user_active and not is_admin:
#     print("Пароль принят, пользователь активен и не является администратором")
# else:
#     print("Либо пароль неверен, либо пользователь не активен, либо это администратор")
#
# is_admin = False
#
# if not is_admin:
#     print("Доступ запрещен")
# else:
#     print("Добро пожаловать, администратор")
#
# password = "admin123"
# backup_password = "adminbackup"
#
# if password == "admin123" or password == "adminbackup":
#     print("Пароль принят")
# else:
#     print("Пароль неверен")

# a1 =input()
# a2 =input()
# if a1= ="файлообменник":
#     if a2= ="скайп":
#         print("С-К-А-А-А-Й-П")


# a1=int(input())
# if a1%2 == 0:
#     print("YES")
# else:
#     print("NO")

# a1=int(input())
# print(f"a1%10 = {a1%10}")
# print(f"a1//10 = {a1//10}")
#
# if a1%10 == a1//10:
#     print("Как две капли")
# else:
#     print("Не совпали")

# a1=input()
# a2=a1[::-1]
# if a1==a2:
#     print("YES")
# else:
#     print("NO")


# a1=input()
# if a1.isupper():
#     print("Не ори, пожалуйста!")
# else:
#     print("Хорошо сидим, болтаем")

# a1=int(input())
# if a1 < 18:
#     print("Школьник")
# else:
#     print("Студент")
#
# a1=input()
# l=len(a1)
# if l >= 10:
#     print("Она огромная")
# else:
#     print("Она что, маленькая?")

# a1=input()
# d1=int(input())
# d2=int(input())
# d3=int(input())
# print(a1[d1:d2:d3])
#
# a1=input()
# print(a1[18:-1])

# a1=input()
# a2=int(input())
# print(f'В строке "{a1}" символ с индексом {a2} вот такой: "{a1[a2]}"')
#
# a1=input()
# print(f'В строке "{a1}" первый символ: "{a1[0]}", а последний символ "{a1[-1]}"')

# a1=input()
# print(a1[0])
#
# a1=input()
# print(f"Проверка, что тут только цифры! Результат: {a1.isdigit()}")
#
# a1 = input()
# print(a1.startswith("Дорогой"))


# a1 = input()
# res = len(a1)
# print(f'В строке "Исходная строка" - {res} символов, включая пробелы')
#
# a1 = input()
# res = len(a1)
# print(f'В строке "{a1}" - {res} символов, включая пробелы')


# a1=input()
# print(f'"{a1.lower()}"')

# a1=input()
# print(a1.upper())
#
#
# a1 = input()
# res = len(a1)
# print(res)

# a1=input()
# result = eval(a1)
# print(f"{a1}={result}")

# a1 = int(input())
# a2 = int(input())
# print(f"Сумма чисел {a1} и {a2} равна: {a1+a2}, при этом их разность равна: {a1-a2}")
#
# a1 = int(input())
# a2 = int(input())
# print(f"Первое число: {a1}")
# print(f"Второе число: {a2}")
# print(f"Сумма чисел: {a1+a2}")
# print(f"Разность чисел: {a1-a2}")
# print(f"Остаток от деления: {a1%a2}")

# a = 3
# b = 4
# print(f"Сумма чисел {a} и {b} равна {a + b}")
#
# a = 5
# b = 4
# print(f"Сумма чисел {a} и {b} равна {a + b}\nПроизведение чисел {a} и {b} равно {a * b}")

# year = int(input())
# result = year + 1
# print(result)
#
# year = int(input())
# classChild = year-7+1
# print(classChild,"класс")

# print("apple", "banana", "cherry", sep=", ")
#
# my_symbol = "@"
# print(1, "2 слово", 3, sep=my_symbol)
#
# one, two, three, four  = input(), input(), input(), input()

# print(6+9)
# a = "pe"
# b = "rec"
# print(a + b)
#
# a = "h"
# b = 6
# print(a * b)
#
# name = input("Введите свое имя: ")
# age = int(input("Введите свой возраст: "))
#
# a = input("Введите что-нибудь: ")
# print("Вы ввели:", a)
#
# str1 = "Hello"
# str2 = "World"
# result = str1 + " " + str2
# print(result)  # Вывод: Hello World
#
# num1 = 10
# num2 = 20
# result = num1 + num2
# print(result)  # Вывод: 30
#
# num1 = input("Введите первое число: ")
# num2 = input("Введите второе число: ")
#
# num1 = int(num1)
# num2 = int(num2)
#
# result = num1 + num2
# print("Сумма чисел:", result)
#
# num1 = int(input("Введите первое число: "))
# num2 = int(input("Введите второе число: "))
# result = num1 + num2
# print("Сумма чисел:", result)
#
# text = "Привет"
# num = 3
# result = text * num
# print(result)  # Вывод: ПриветПриветПривет