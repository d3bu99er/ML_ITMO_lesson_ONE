import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, recall_score
from scipy.sparse import lil_matrix, csr_matrix, hstack
from collections import Counter
import re
import warnings
warnings.filterwarnings("ignore")

# Опционально: проверка на словарь
USE_DICTIONARY = False
try:
    import enchant
    DICTIONARY = enchant.Dict("en_US")
    USE_DICTIONARY = True
except ImportError:
    pass

np.random.seed(42)


def is_real_word_part(domain: str, min_len=3) -> float:
    if not USE_DICTIONARY or len(domain) < min_len:
        return 0.0
    domain = domain.lower()
    for i in range(len(domain)):
        for j in range(i + min_len, len(domain) + 1):
            substr = domain[i:j]
            if substr.isalpha() and DICTIONARY.check(substr):
                return 1.0
    return 0.0


def extract_features(domain):
    domain = str(domain).lower().strip()
    if not domain:
        return [0.0] * 14

    length = len(domain)
    digit_count = sum(c.isdigit() for c in domain)
    hyphen_count = domain.count('-')
    dot_count = domain.count('.')
    vowel_count = sum(1 for c in domain if c in 'aeiou')
    consonant_count = sum(1 for c in domain if c in 'bcdfghjklmnpqrstvwxyz')
    unique_ratio = len(set(domain)) / length if length > 0 else 0.0

    digit_seqs = re.findall(r'\d+', domain)
    max_digit_seq = max((len(s) for s in digit_seqs), default=0)

    # Энтропия
    if length <= 1:
        entropy = 0.0
    else:
        from collections import Counter as PyCounter
        counts = np.array(list(PyCounter(domain).values()))
        probs = counts / length
        entropy = -np.sum(probs * np.log2(probs + 1e-12))  # защита от log(0)

    # Редкие биграммы
    rare_bigrams = {'qx', 'qz', 'qj', 'qk', 'xj', 'xq', 'zj', 'zq', 'jk', 'kq'}
    rare_count = sum(1 for i in range(len(domain)-1) if domain[i:i+2] in rare_bigrams)

    word_like = is_real_word_part(domain)

    return [
        float(length),
        float(digit_count),
        float(hyphen_count),
        float(dot_count),
        float(entropy),
        float(vowel_count / length) if length > 0 else 0.0,
        float(consonant_count / length) if length > 0 else 0.0,
        float(digit_count / length) if length > 0 else 0.0,
        float(unique_ratio),
        float(max_digit_seq),
        float(rare_count),
        float(word_like),
        1.0 if '.' in domain else 0.0,
        float(len(digit_seqs))
    ]


def extract_top_bigrams(domains, max_features=800):
    counter = Counter()
    for domain in domains:
        domain = str(domain).lower()
        bigrams = [domain[i:i+2] for i in range(len(domain) - 1)]
        counter.update(bigrams)
    return [bg for bg, _ in counter.most_common(max_features)]


def vectorize_bigrams(domains, bigram_list):
    n_samples = len(domains)
    n_features = len(bigram_list)
    bg_to_idx = {bg: i for i, bg in enumerate(bigram_list)}
    matrix = lil_matrix((n_samples, n_features), dtype=np.float32)

    for i, domain in enumerate(domains):
        domain = str(domain).lower()
        for j in range(len(domain) - 1):
            bg = domain[j:j+2]
            if bg in bg_to_idx:
                matrix[i, bg_to_idx[bg]] += 1

    return matrix.tocsr()


def create_feature_matrix(df, bigram_list=None, is_training=False):
    domains = df['domain'].astype(str).values

    # Handcrafted features
    print("→ Extracting handcrafted features...")
    handcrafted = np.array([extract_features(d) for d in domains], dtype=np.float32)
    handcrafted_sparse = csr_matrix(handcrafted)

    # Bigram features
    print("→ Extracting bigram features...")
    if is_training:
        bigram_list = extract_top_bigrams(domains, max_features=800)
    bigram_features = vectorize_bigrams(domains, bigram_list)

    # Combine
    combined = hstack([bigram_features, handcrafted_sparse], format='csr')
    return combined, bigram_list


def load_and_sample_data(max_samples_per_class=75_000):
    # Load train
    try:
        train_df = pd.read_csv('train.csv')
        if 'domain' not in train_df.columns or 'label' not in train_df.columns:
            train_df = pd.read_csv('train.csv', names=['domain', 'label'], header=None)
    except:
        train_df = pd.read_csv('train.csv', names=['domain', 'label'], header=None)

    # Load test
    with open('test.csv', 'r') as f:
        first_line = f.readline().strip()
    if 'id' in first_line and 'domain' in first_line:
        test_df = pd.read_csv('test.csv', index_col='id')
    else:
        test_df = pd.read_csv('test.csv', names=['id', 'domain'], header=None, index_col='id')

    train_df = train_df.dropna().reset_index(drop=True)
    test_df = test_df.dropna().reset_index(drop=True)
    train_df['label'] = train_df['label'].astype(int)

    # Stratified sampling to limit size
    sampled_dfs = []
    for label in [0, 1]:
        class_data = train_df[train_df['label'] == label]
        if len(class_data) > max_samples_per_class:
            class_data = class_data.sample(n=max_samples_per_class, random_state=42)
        sampled_dfs.append(class_data)
    train_df = pd.concat(sampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Train samples after sampling: {len(train_df)}")
    print("Class distribution:", train_df['label'].value_counts().to_dict())
    print(f"Test samples: {len(test_df)}")
    return train_df, test_df


def main():
    print("🔍 DGA Detection (Lightweight Mode)")
    train_df, test_df = load_and_sample_data(max_samples_per_class=75_000)

    # Create features
    X_train, bigram_list = create_feature_matrix(train_df, is_training=True)
    y_train = train_df['label'].values

    X_test, _ = create_feature_matrix(test_df, bigram_list=bigram_list, is_training=False)

    print(f"Feature matrix shape: {X_train.shape}")

    # Train model
    model = RandomForestClassifier(
        n_estimators=80,
        max_depth=16,
        min_samples_split=8,
        min_samples_leaf=3,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # Validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    model.fit(X_tr, y_tr)

    val_pred = model.predict(X_val)
    print("\n✅ Validation Recall (DGA detection):", recall_score(y_val, val_pred))
    print(classification_report(y_val, val_pred))

    # Predict with lower threshold for higher sensitivity
    test_proba = model.predict_proba(X_test)[:, 1]
    test_pred = (test_proba > 0.30).astype(int)  # prioritize recall

    # Save
    submission = pd.DataFrame({
        'id': test_df.index,
        'label': test_pred
    })
    submission.to_csv('predictions_light.csv', index=False)
    print(f"\n💾 Predictions saved to predictions_light.csv")
    print("Prediction stats:", dict(Counter(test_pred)))

    # Optional: save model
    import joblib
    joblib.dump({'model': model, 'bigram_list': bigram_list}, 'dga_model_light.pkl')
    print("Model saved.")


if __name__ == "__main__":
    main()