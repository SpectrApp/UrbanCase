## Baseline - How To 🚀

Ссылка на кейс - [тык](https://drive.google.com/file/d/1Y9CVBVeERDdQNafWxdnpqD6z63VPMm79/view?pli=1). 


### Структура baseline

- Файл `solution.py` основной файл, где идет инициализация моделей, 
запуск инференса и форматирование предсказаний 💫, 
- Файл `model.py` содержит код, отвечающий за инференс модели,
- Файл `scorer.py` содержит код с подсчетом метрики.


### Сборка решения 🐳

Для начала вам необходимо установить Docker 🐳,
подробную иструкцию для этого вы сможете найти по ссылке: https://docs.docker.com/get-docker/

После того, как Вы установили Docker и убедились в его работоспособности,
вы можете собрать текущий бейзлайн. Для этого запустите следующую команду
из корневой директории проекта:
```bash
docker build -t urbancode-baseline .
```

Теперь вы можете собранный контейнер с помощью команды:
```bash
docker run urbancode-baseline
```

в результате чего, в `stdout` будет выведено число - значение метрики mAP@.50. 

**N.B. Если вы хотите проверить работоспособность вашего контейнера локально 
(вместе с подсчетом метрики), то вам нужно сделать следующее:**

0. Положить ваши тестовые картинки (например, из числа тех, которые предоставели организаторы) в папку `./images`, 
а файлы разметки в `./labels` (txt-файлы в формате `label xmin ymin xmax ymax`) 
1. Прописать в `Dockerfile` (после соотвествующих `mkdir`-ов:
```
COPY images ./private/images
COPY labels ./private/labels
```

2. Собрать и запустить контейнер
```bash
docker build -t urbancode-baseline .
```
```bash
docker run urbancode-baseline
````


### Формат решения
Для каждого изображения из тестового набора вы должны предсказывать а) координаты ячеек, 
их б) степень их готовности и в) уверенность в этом предсказании.

Иными словами,
ваше решение должно считывать изображения в форматах `.jpg`/`.png` из папки 
`./private/images` и для каждого изображения `<image-name>.<format>` из входной папки записывать файл
с названием `<image-name>.txt` в формате:
```
label1 score1 xmin1 ymin1 xmax1 xmax1
label2 score2 xmin2 ymin2 xmax2 xmax2
...
```
где
- `label` - метка класса, целые число (1, 2 или 3), 
- `score` - уверенность вашей модели в предсказании, вещественные числа в промежутке [0, 1], 
- `xmin, ymin, xmax, ymax` - координаты ячеек, целые числа.