
import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, ParameterSampler
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score
from clearml import Task, Logger


# Настройка окружения ClearML
# Замените на свои ключи доступа к ClearML серверу


'''
После подготовки и возможного создания виртуального окружения в терминале исполните комманду 
clearml-init

После скопируйте сгененрированный ключ API Credentials вашего рабочего пространства

api {
  # Viktor Tatarnikov's workspace
  web_server: https://app.clear.ml/
  api_server: https://api.clear.ml
  files_server: https://files.clear.ml
  credentials {
    "access_key" = "YOUR_ACCESS_KEY"
    "secret_key" = "YOUR_SECRET_KEY"
  }
}'''


# Инициализация ClearML задачи
print("Инициализация ClearML задачи...")
task = Task.init(project_name='Titanic_CatBoost_Demo',
           task_name='CatBoost_Classification_v1',
           tags=['Model:CatBoost', 'Dataset:Titanic', 'Type:Classification'])


# Загрузка и первичный анализ данных
print("\nЗагрузка датасета Titanic...")
# Убедитесь, что файл titanic.csv находится в той же папке
df_raw = pd.read_csv('titanic.csv')

# Загрузка сырых данных как артефакт в ClearML
task.upload_artifact(name='data.raw', artifact_object=df_raw)

print(f"Размер датасета: {df_raw.shape}")
print("Первые 5 строк:")
print(df_raw.head())


# Статистическое описание данных
print("\nСтатистическое описание данных...")
task.upload_artifact(name='eda.describe', artifact_object=df_raw.describe())
print(df_raw.describe())


# Анализ целевой переменной
print("\nАнализ целевой переменной...")
target_distribution = df_raw['Survived'].value_counts(normalize=True).reset_index()
task.upload_artifact(name='target.distribution', artifact_object=target_distribution)

print("Распределение целевой переменной:")
print(target_distribution)

# Визуализация распределения
plt.figure(figsize=(8, 6))
sns.countplot(data=df_raw, x='Survived')
plt.title('Распределение выживших пассажиров')
plt.xlabel('Выжил (1) / Не выжил (0)')
plt.ylabel('Количество')
plt.show()


# Предобработка данных
print("\nПредобработка данных...")
# Удаляем колонки, которые не несут полезной информации для модели
df_preproc = df_raw.drop(columns=['PassengerId', 'Name', 'Ticket'])

# Преобразуем категориальные признаки в строковый тип
# CatBoost автоматически обработает их как категориальные
for col in ['Sex', 'Cabin', 'Embarked']:
    df_preproc[col] = df_preproc[col].astype(str)
    
# Загружаем предобработанные данные как артефакт
task.upload_artifact(name='data.preprocessed', artifact_object=df_preproc)

print(f"Размер после предобработки: {df_preproc.shape}")
print("\nКолонки:")
print(df_preproc.columns.tolist())


# Разделение данных на тренировочную и тестовую выборки
print("\nРазделение данных на тренировочную и тестовую выборки...")
# Разделяем данные на тренировочную (67%) и тестовую (33%) выборки
train, test = train_test_split(df_preproc, test_size=0.33, random_state=42)

# Загружаем разделённые данные как артефакты
task.upload_artifact(name='data.train', artifact_object=train)
task.upload_artifact(name='data.test', artifact_object=test)

print(f"Тренировочная выборка: {train.shape}")
print(f"Тестовая выборка: {test.shape}")


# Подготовка признаков для модели
print("\nПодготовка признаков для модели...")
# Выделяем признаки и целевую переменную для тренировочной выборки
X_train = train.drop(columns=['Survived']) 
X_train = X_train[['Sex', 'Cabin', 'Embarked']]  # Используем только категориальные признаки
y_train = train['Survived']

# Выделяем признаки и целевую переменную для тестовой выборки
X_test = test.drop(columns=['Survived']) 
X_test = X_test[['Sex', 'Cabin', 'Embarked']]
y_test = test['Survived']

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# Настройка гиперпараметрической оптимизации
print("\nНастройка гиперпараметрической оптимизации...")
# Сетка гиперпараметров для поиска
param_grid = {
    'depth' : [4, 5, 6, 7, 8, 10, 16],        # Глубина дерева
    'learning_rate': [0.1, 0.05, 0.01, 0.0005, 0.0001],  # Скорость обучения
    'iterations': [30, 50, 100, 150, 200]     # Количество итераций
}

# Инициализация логгера для записи метрик
log = Logger.current_logger()

# Переменные для отслеживания лучшей модели
best_score = 0 
best_model = None
iteration = 0

print(f"Будет протестировано 4 случайные комбинации из {len(param_grid['depth']) * len(param_grid['learning_rate']) * len(param_grid['iterations'])} возможных")


# Обучение моделей и поиск лучших гиперпараметров
print("\nОбучение моделей и поиск лучших гиперпараметров...")
# Перебираем случайные комбинации гиперпараметров
for param in ParameterSampler(param_grid, n_iter=4, random_state=42):
    print(f"\nТестируем параметры: {param}")
    
    # Подключаем параметры к задаче ClearML для отслеживания
    parametrs_dict = Task.current_task().connect(param)
    
    # Создаём и обучаем модель CatBoost
    model = CatBoostClassifier(**param, silent=True)
    # ВАЖНО: обучаем на тренировочных данных!
    model.fit(X_train, y_train, cat_features=['Sex', 'Cabin', 'Embarked'])
    
    # Оценка модели на тестовой выборке
    test_scores = model.eval_metrics(
        data=Pool(X_test, y_test, cat_features=['Sex', 'Cabin', 'Embarked']),
        metrics=['Logloss', 'AUC'])
    test_logloss = round(test_scores['Logloss'][-1], 4)
    test_roc_auc = round(test_scores['AUC'][-1]*100, 4)
    
    # Оценка модели на тренировочной выборке (для контроля переобучения)
    train_scores = model.eval_metrics(
        data=Pool(X_train, y_train, cat_features=['Sex', 'Cabin', 'Embarked']),
        metrics=['Logloss', 'AUC'])
    train_logloss = round(train_scores['Logloss'][-1], 4)
    train_roc_auc = round(train_scores['AUC'][-1]*100, 4)
    
    print(f"Тестовая выборка - Logloss: {test_logloss}, ROC AUC: {test_roc_auc}%")
    print(f"Тренировочная выборка - Logloss: {train_logloss}, ROC AUC: {train_roc_auc}%")
    
    # Сохраняем лучшую модель
    if test_roc_auc > best_score:
        best_score = test_roc_auc
        best_model = model
        
        # Логируем метрики в ClearML
        log.report_scalar('Logloss', 'Test', iteration=iteration, value=test_logloss)
        log.report_scalar('Logloss', 'Train', iteration=iteration, value=train_logloss)
        
        log.report_scalar('ROC AUC', 'Test', iteration=iteration, value=test_roc_auc)
        log.report_scalar('ROC AUC', 'Train', iteration=iteration, value=train_roc_auc)
        
        iteration += 1
        print(f"🎉 Новая лучшая модель! ROC AUC: {best_score}%")


# Финальные результаты и сохранение модели
print("\nФинальные результаты и сохранение модели...")
# Логируем финальные результаты в ClearML
log.report_single_value(name='Best ROC AUC', value=best_score)
log.report_single_value(name='Best Logloss', value=test_logloss)
log.report_single_value(name='Train Rows', value=X_train.shape[0])
log.report_single_value(name='Test Rows', value=X_test.shape[0])
log.report_single_value(name='Features', value=X_train.shape[1])

print(f"\n Лучший результат: ROC AUC = {best_score}%")
print(f" Размер тренировочной выборки: {X_train.shape[0]} строк")
print(f" Размер тестовой выборки: {X_test.shape[0]} строк")
print(f" Количество признаков: {X_train.shape[1]}")


# Сохранение лучшей модели
print("\nСохранение лучшей модели...")
best_model_name = 'best_catboost_model.cbn'
best_model.save_model(best_model_name)

# Загружаем модель как артефакт в ClearML
task.upload_artifact(name=best_model_name, artifact_object=best_model)

print(f"Модель сохранена как '{best_model_name}'")


# Закрытие задачи ClearML
print("\nЗакрытие задачи ClearML...")
task.close()

print("Эксперимент завершён! Проверьте результаты в ClearML веб-интерфейсе.")


"""
Заключение

Этот проект демонстрирует:
1. Интеграцию CatBoost с ClearML для отслеживания экспериментов
2. Автоматическое логирование метрик, параметров и артефактов
3. Гиперпараметрическую оптимизацию с помощью ParameterSampler
4. Правильную практику разделения данных на train/test

Все результаты эксперимента автоматически сохраняются в ClearML и доступны для анализа в веб-интерфейсе.

Основные исправления в коде:
- Исправлена ошибка обучения модели на тестовых данных (было X_test, стало X_train)
- Добавлены подробные комментарии к каждому блоку кода
- Улучшена структура и читаемость кода
- Добавлены информативные print-сообщения для отслеживания прогресса
"""

