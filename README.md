# CoffeeChurnPredictor
Репозиторий с необходимыми файлами для учебного проекта. 

## Содержание репозитория
- `requiments.txt` - файл с версиями библиотек, используемых в проекте. 
- `churn_project.py` - скрипт с кастомными классами `OutlierClipper`, `ChurnPredictorStrategy` и кастомной функцией `create_new_features()`.
- `churn_predictor_model.pkl` - файл пайплана итоговой модели.
- `coffee_churn_dataset_for_check.csv` - csv-файл с 20 пользователями для тестирования итога проекта. 
- `test_notebook_model.ipynb` - блокнот JupyterNotebook с примерным запуском итога проекта для продакшена.

## Рекомендации для бизнеса по использованию
1. Загрузить все библиотеки и обновить версии до необходимых:
```python
%pip install -r requirements.txt
```
2. Импортировать все классы и функцию из скрипта:
```python
from churn_project import OutlierClipper, create_new_features, ChurnPredictorStrategy
import joblib
import pandas as pd
```
3. Загрузить модель и необходимый датасет из файла (сейчас в примере собственный датасет для тестирования):
```python
model = joblib.load('churn_predictor_model.pkl')
data = pd.read_csv('coffee_churn_dataset_for_check.csv', sep=';', decimal='.')
```
4. Создать объект предсказателя оттока клиентов:
   - Для этого необходимо выбрать стратегию предсказания (можно создать несколько объектов на несколько стратегий), где:
   - `max_recall` - широкий охват, точность ниже. 
   - `max_precision` - высокая точность, но охватывает меньшую часть уходящих клиентов. 
   - `max_f1_score` - сбалансированная стратегия.
  
```python
predictor = ChurnPredictorStrategy(model, 'max_recall')
predictor.predict(data)
```
5. Получить массив с предсказаниями об уходе каждого конкретного клиента (1 - уйдет, 0 - останется).

---
Такой пример использования можно найти в блокноте `test_notebook.ipynb`. 
