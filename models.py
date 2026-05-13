import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # Yüzdelik ilerleme çubuğu için
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from analyse import get_cleaned_data


def train_ml_process():
    print("\n[1/4] Veri hazırlanıyor...")
    df = get_cleaned_data()

    # Analizler sonucu en ayırt edici seçilen özellikler
    X = df[['i_q', 'i_d', 'motor_speed', 'ambient']]
    y = df['Motor_Health']

    # %80 Eğitim, %20 Test ayrımı
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    print("[2/4] Random Forest Modeli Eğitiliyor...")
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, warm_start=True)
    # İlerleme çubuğu eklendi
    for i in tqdm(range(1, 101, 10), desc="Eğitim İlerlemesi"):
        rf_model.n_estimators = i
        rf_model.fit(X_train, y_train)

    print("\n[3/4] XGBoost Modeli Eğitiliyor...")
    xgb_model = XGBClassifier(eval_metric='logloss', n_estimators=100)
    xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=25)

    print("\n[4/4] Performans Metrikleri Hesaplanıyor...")
    rf_preds = rf_model.predict(X_test)
    xgb_preds = xgb_model.predict(X_test)

    # Hata Matrisi (Confusion Matrix) kaydı
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, rf_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='magma')
    plt.title("Tahmin Başarısı (Confusion Matrix)")
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()

    results = {
        "report": classification_report(y_test, rf_preds),
        "auc": roc_auc_score(y_test, xgb_preds),
        "importances": rf_model.feature_importances_
    }

    return results, X.columns