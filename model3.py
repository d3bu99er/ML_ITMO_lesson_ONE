import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
from scipy.sparse import lil_matrix, csr_matrix, hstack
from collections import Counter

# Установливаем случайное начальное число для воспроизводимости
np.random.seed(42)


def preprocess_domain(domain):
    """Preprocess domain names by extracting features"""
    domain = str(domain).lower().strip()

    # Убеждаемся, что домен не пустой
    if len(domain) == 0:
        return [0, 0, 0, 0, 0, 0, 0, 0, 0]

    # Основные характеристики
    length = len(domain)
    digit_count = sum(c.isdigit() for c in domain)
    hyphen_count = domain.count('-')
    dot_count = domain.count('.')

    # Безопасный подсчет количества символов
    vowel_count = sum(1 for c in domain if c in 'aeiou')
    consonant_count = sum(1 for c in domain if c in 'bcdfghjklmnpqrstvwxyz')

    # Рассчитываем энтропию
    if length <= 1:
        entropy = 0.0
    else:
        chars, counts = np.unique(list(domain), return_counts=True)
        probabilities = counts / length
        entropy = float(-np.sum(probabilities * np.log2(probabilities)))

    features = [
        float(length),
        float(digit_count),
        float(hyphen_count),
        float(dot_count),
        float(entropy),
        float(vowel_count / length),
        float(consonant_count / length),
        float(digit_count / length),
        1.0 if '.' in domain else 0.0
    ]

    return features


def extract_simple_ngram_features(domains, max_features=500):
    """Extract simple n-gram features without TF-IDF"""
    all_ngrams = Counter()

    for domain in domains:
        domain = str(domain).lower()
        # Extract 2-grams
        ngrams = [domain[i:i + 2] for i in range(len(domain) - 1)]
        all_ngrams.update(ngrams)

    # Оставляем только самые распространенные n-граммы
    common_ngrams = [ngram for ngram, count in all_ngrams.most_common(max_features)]

    # Создаём матрицу признаков
    features = lil_matrix((len(domains), len(common_ngrams)))
    ngram_to_idx = {ngram: idx for idx, ngram in enumerate(common_ngrams)}

    for i, domain in enumerate(domains):
        domain = str(domain).lower()
        ngrams = [domain[i:i + 2] for i in range(len(domain) - 1)]
        for ngram in ngrams:
            if ngram in ngram_to_idx:
                features[i, ngram_to_idx[ngram]] += 1

    return features.tocsr(), common_ngrams


def extract_simple_ngram_features_test(domains, common_ngrams):
    """Extract n-gram features for test data using training vocabulary"""
    features = lil_matrix((len(domains), len(common_ngrams)))
    ngram_to_idx = {ngram: idx for idx, ngram in enumerate(common_ngrams)}

    for i, domain in enumerate(domains):
        domain = str(domain).lower()
        ngrams = [domain[i:i + 2] for i in range(len(domain) - 1)]
        for ngram in ngrams:
            if ngram in ngram_to_idx:
                features[i, ngram_to_idx[ngram]] += 1

    return features.tocsr()


def create_features_optimized(df, common_ngrams=None, is_training=True):
    """Create feature matrix - FIXED VERSION"""
    domains = df['domain'].astype(str).values

    # Извлечение вручную созданных особенностей
    print("Extracting handcrafted features...")
    handcrafted_features = []
    for i, domain in enumerate(domains):
        if i % 10000 == 0 and len(domains) > 10000:
            print(f"Processed {i}/{len(domains)} domains")
        features = preprocess_domain(domain)
        handcrafted_features.append(features)

    handcrafted_array = np.array(handcrafted_features, dtype=np.float64)

    # Извлечение признаков n-грамм
    print("Extracting n-gram features...")
    if is_training:
        ngram_features, common_ngrams = extract_simple_ngram_features(domains)
    else:
        if common_ngrams is None:
            raise ValueError("common_ngrams must be provided for test feature creation")
        ngram_features = extract_simple_ngram_features_test(domains, common_ngrams)

    # Преобразовать в разреженную матрицу и объединить
    handcrafted_sparse = csr_matrix(handcrafted_array)

    # Объединить функции
    combined_features = hstack([ngram_features, handcrafted_sparse])

    if is_training:
        return combined_features, common_ngrams, handcrafted_array.shape[1]
    else:
        return combined_features


def main_optimized():
    # Загрузить данные обучения
    print("Loading training data...")
    try:
        # Пробуем сначала прочитать с заголовком, а затем без него, если не получится
        try:
            train_df = pd.read_csv('train.csv')
            if 'domain' not in train_df.columns or 'label' not in train_df.columns:
                train_df = pd.read_csv('train.csv', names=['domain', 'label'],
                                       header=None)
        except:
            train_df = pd.read_csv('train.csv', names=['domain', 'label'], header=None)

        print(f"Training data loaded: {len(train_df)} samples")

        # Если набор данных слишком большой, делаем выборку
        if len(train_df) > 50000:
            print("Dataset is large, sampling to 50000 samples for training...")
            train_df = train_df.sample(n=50000, random_state=42)
            print(f"Using {len(train_df)} samples for training")

    except FileNotFoundError:
        print("Training file not found. Please ensure 'train.csv' exists.")
        return
    except Exception as e:
        print(f"Error loading train.csv: {e}")
        return

    # Загрузка тестовых данных
    print("Loading test data...")
    try:
        test_df = pd.read_csv('test.csv', names=['id', 'domain'], header=None)
        print(f"Test data loaded: {len(test_df)} samples")

        # Отладочная информация
        print(f"Test columns: {test_df.columns.tolist()}")
        print(f"First few IDs: {test_df['id'].head().tolist()}")
        print(f"Test ID range: {test_df['id'].min()} to {test_df['id'].max()}")

    except FileNotFoundError:
        print("Test file not found. Please ensure 'test.csv' exists.")
        return
    except Exception as e:
        print(f"Error loading test.csv: {e}")
        return

    # Чистим данные — обеспечиваем правильные типы данных
    train_df = train_df.dropna()
    test_df = test_df.dropna()

    # Преобразовываем метки в целые числа, если они являются строками
    train_df['label'] = train_df['label'].astype(int)

    # Распределение классов отображения
    print(f"\nTraining class distribution:\n{train_df['label'].value_counts()}")

    # Создание признаков для обучающих данных
    print("\nCreating features for training data...")
    try:
        X_train, common_ngrams, n_handcrafted = create_features_optimized(train_df, is_training=True)
        y_train = train_df['label'].values

        print(f"Feature matrix shape: {X_train.shape}")
        print(f"Feature matrix data type: {X_train.dtype}")
    except Exception as e:
        print(f"Error creating training features: {e}")
        # Возврат к простой версии
        simple_version()
        return

    # Создание признаков для тестовых данных
    print("Creating features for test data...")
    try:
        X_test = create_features_optimized(test_df, common_ngrams=common_ngrams, is_training=False)
        print(f"Test feature matrix shape: {X_test.shape}")
    except Exception as e:
        print(f"Error creating test features: {e}")
        # Пробуем альтернативный подход
        print("Trying alternative feature extraction...")
        alternative_approach(train_df, test_df)
        return

    # Тренируем модель
    print("\nTraining model...")
    try:
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            random_state=42,
            n_jobs=-1
        )

        # Используем проверку, если у нас достаточно данных
        if len(train_df) > 1000:
            X_train_split, X_val, y_train_split, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            model.fit(X_train_split, y_train_split)

            # Валидируем
            y_val_pred = model.predict(X_val)
            val_accuracy = accuracy_score(y_val, y_val_pred)
            print(f"Validation Accuracy: {val_accuracy:.4f}")

            # Формируем отчёт как строку
            report_str = classification_report(y_val, y_val_pred)
            print("\nValidation Classification Report:")
            print(report_str)

            # Сохраняем отчёт и параметры в score.txt (перезапись)
            with open('score.txt', 'w', encoding='utf-8') as f:
                f.write("Validation Classification Report:\n")
                f.write(report_str)
                f.write(f"\nn=50000\n")
                f.write(f"random_state=42\n")
                f.write(f"test_size=0.2\n")

        else:
            model.fit(X_train, y_train)
            # Если нет валидации — всё равно создаём файл с пометкой
            with open('score.txt', 'w', encoding='utf-8') as f:
                f.write("Validation Classification Report:\n")
                f.write("Not enough data for validation (n <= 1000).\n")
                f.write(f"\nn=50000\n")
                f.write(f"random_state=42\n")
                f.write(f"test_size=0.2\n")

        # Делаем прогнозы на тестовом наборе
        print("\nMaking predictions on test set...")
        test_predictions = model.predict(X_test)

        # ПРОВЕРКА: Убеждаемся, что у нас такое же количество прогнозов, как и в тестовых образцах
        print(f"Number of test samples: {len(test_df)}")
        print(f"Number of predictions: {len(test_predictions)}")

        if len(test_predictions) != len(test_df):
            print("WARNING: Mismatch between test samples and predictions!")
            test_predictions = test_predictions[:len(test_df)]
            print(f"Truncated predictions to {len(test_predictions)}")

        # Создаём файл для отправки
        submission = pd.DataFrame({
            'id': test_df['id'].values,
            'label': test_predictions
        })

        print(f"Submission ID range: {submission['id'].min()} to {submission['id'].max()}")
        print(f"Submission has {len(submission)} rows")

        # Сохранить прогнозы
        submission_file = 'predictions_optimized.csv'
        submission.to_csv(submission_file, index=False)
        print(f"\nPredictions saved to {submission_file}")
        print(f"Prediction distribution:\n{submission['label'].value_counts()}")

        # Отобразить первые несколько прогнозов с их фактическими идентификаторами
        print("\nFirst 10 predictions:")
        print(submission.head(10))

        # Сохранить модель для использования в будущем
        joblib.dump({
            'model': model,
            'common_ngrams': common_ngrams,
            'n_handcrafted': n_handcrafted
        }, 'dga_classifier_optimized.pkl')
        print("\nModel saved for future use")

    except Exception as e:
        print(f"Error during model training/prediction: {e}")


def alternative_approach(train_df, test_df):
    """Alternative approach using only handcrafted features"""
    print("Using alternative approach (handcrafted features only)...")

    try:
        # Извлечение вручную созданных функций для обучения и тестирования
        def extract_features(df):
            features = []
            for domain in df['domain']:
                features.append(preprocess_domain(domain))
            return np.array(features, dtype=np.float64)

        X_train = extract_features(train_df)
        X_test = extract_features(test_df)
        y_train = train_df['label'].values

        print(f"Training features shape: {X_train.shape}")
        print(f"Test features shape: {X_test.shape}")

        # Train model
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )

        # Разделение для проверки
        if len(train_df) > 1000:
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
            )
            model.fit(X_tr, y_tr)

            # Валидация
            val_pred = model.predict(X_val)
            accuracy = accuracy_score(y_val, val_pred)
            print(f"Validation Accuracy: {accuracy:.4f}")
            report_str = classification_report(y_val, val_pred)
            print("\nValidation Classification Report:")
            print(report_str)

            # Сохраняем в score.txt даже в альтернативном подходе
            with open('score.txt', 'w', encoding='utf-8') as f:
                f.write("Validation Classification Report (Alternative):\n")
                f.write(report_str)
                f.write(f"\nn=50000\n")
                f.write(f"random_state=42\n")
                f.write(f"test_size=0.2\n")
        else:
            model.fit(X_train, y_train)
            with open('score.txt', 'w', encoding='utf-8') as f:
                f.write("Validation Classification Report (Alternative):\n")
                f.write("Not enough data for validation.\n")
                f.write(f"\nn=50000\n")
                f.write(f"random_state=42\n")
                f.write(f"test_size=0.2\n")

        # Предсказывание
        test_pred = model.predict(X_test)

        # Сохраняем результаты
        submission = pd.DataFrame({
            'id': test_df['id'],
            'label': test_pred
        })
        submission.to_csv('predictions_alternative.csv', index=False)

        print("Alternative approach completed!")
        print(f"Predictions saved. Distribution:\n{submission['label'].value_counts()}")

        # Сохраняем модель
        joblib.dump(model, 'dga_classifier_alternative.pkl')

    except Exception as e:
        print(f"Error in alternative approach: {e}")
        simple_version()


def simple_version():
    """Simple version using basic features"""
    print("Using simple version...")

    try:
        # Загрузка данных
        train_df = pd.read_csv('train.csv', names=['domain', 'label'])
        test_df = pd.read_csv('test.csv', names=['id', 'domain'])

        # Базовое извлечение признаков
        def get_basic_features(domains):
            features = []
            for domain in domains:
                domain = str(domain).lower()
                length = len(domain)
                if length == 0:
                    features.append([0, 0, 0, 0])
                else:
                    features.append([
                        length,
                        sum(c.isdigit() for c in domain),
                        domain.count('.'),
                        domain.count('-'),
                    ])
            return np.array(features, dtype=np.float64)

        X_train = get_basic_features(train_df['domain'])
        X_test = get_basic_features(test_df['domain'])
        y_train = train_df['label'].astype(int).values

        # Train and predict
        model = RandomForestClassifier(n_estimators=30, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Save results
        result_df = pd.DataFrame({
            'id': test_df['id'],
            'label': predictions
        })
        result_df.to_csv('predictions_simple.csv', index=False)

        # Запись заглушки в score.txt
        with open('score.txt', 'w', encoding='utf-8') as f:
            f.write("Validation Classification Report (Simple version):\n")
            f.write("No validation performed in simple version.\n")
            f.write(f"\nn=50000\n")
            f.write(f"random_state=42\n")
            f.write(f"test_size=0.2\n")

        print("Simple version completed!")
        print(f"Prediction distribution:\n{result_df['label'].value_counts()}")

    except Exception as e:
        print(f"Error in simple version: {e}")


if __name__ == "__main__":
    print("Starting DGA Domain Classification...")
    main_optimized()