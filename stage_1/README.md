# Мое решение для первого этапа VK Cup ML 2021

Решение, по большей части, взято из приложенной к условию статьи. Рассматриваем друзей друзей (далее "друзья^2") как потенциальных будущих новых друзей.

Изначально посчитаем граф друзей с помощью файла train.csv и сохраним в виде словаря, где ключи - это айдишники пользователей, а значения - массив из айдишников их друзей

Затем необходимо для каждого айдишника из baseline.txt порекомендовать друзей

Для этого для каждого друга^2 мы находим общих друзей с текущим и считаем немного модифицированный Adamic/Adar:
```
relevance = h1 * h2 / log(count_of_friends)
```
Все эти релевантности складываются для конкретного друга^2, сохраняются в отдельном массиве.

Затем этот массив сортируется и выбираются самые релевантные друзья.

**P.S.** в изначальной формуле Adamic/Adar не были использованны h1 и h2, я решил их добавить в числитель, что дало существенный буст конечного скора

**P.S.S. NOTE:** решение требует большого количества оперативной памяти для хранения графа друзей, приблизительно 6 Гб.

Скорее всего, такое решение вряд ли подойдет для продакшна, ведь хранить весь граф друзей в ОЗУ вряд ли представляется возможным. 

Но в качестве решения задачи VK Cup это решение заняло 8 место на ЛБ.
