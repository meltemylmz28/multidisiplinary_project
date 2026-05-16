import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from analyse import get_cleaned_data


def train_and_compare_advanced():
    # Çıktı klasörünü tanımla
    output_dir = os.path.join('version2', 'meltem_v2')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("\n[1/5] 9 Sütunlu (Power Loss dahil) veri seti hazırlanıyor...")
    # Analyse modülünden temiz veriyi alıyoruz
    df = get_cleaned_data(generate_report_visual=True)

    # Özellik seti (9 Sütun)
    features = ['i_q', 'i_d', 'u_q', 'u_d', 'motor_speed', 'ambient', 'coolant', 'torque', 'power_loss']
    X = df[features]
    y = df['Motor_Health']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )

    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced'),
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, scale_pos_weight=9)
    }

    performance_data = []

    print("[2/5] Algoritmalar 'Er Meydanı'na çıkıyor (Eğitim ve Hız Testi)...")
    for name, model in models.items():
        # 1. Kriter: Eğitim Hızı
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time

        preds = model.predict(X_test)

        # 2. Kriter: Performans Metrikleri
        performance_data.append({
            "Algoritma": name,
            "Accuracy": accuracy_score(y_test, preds),
            "Recall": recall_score(y_test, preds),
            "F1-Score": f1_score(y_test, preds),
            "ROC-AUC": roc_auc_score(y_test, preds),
            "Egitim_Suresi_Sn": train_time
        })

        # 3. Kriter: Karar Şeffaflığı (Özellik Önem Sıralaması Görseli)
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.figure(figsize=(10, 6))
        plt.title(f'{name} - Parametre Önem Sıralaması')
        plt.barh(range(len(indices)), importances[indices], color='teal', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Bağıl Önem Skoru')
        plt.tight_layout()

        # meltem/ yerine meltem_v2 klasörüne kayıt
        plt.savefig(os.path.join(output_dir, f'{name.replace(" ", "_")}_onem_analizi.png'), dpi=300)
        plt.close()

    # Sonuç Tablosu
    results_df = pd.DataFrame(performance_data)
    print("\n--- MODEL KARŞILAŞTIRMA TABLOSU ---")
    print(results_df)

    # 4. Kriter: Performans Çizgi Grafiği (Metrik Yarışı)
    plt.figure(figsize=(12, 6))
    plot_df = results_df.melt(id_vars="Algoritma", value_vars=["Accuracy", "Recall", "F1-Score", "ROC-AUC"])
    sns.lineplot(data=plot_df, x="variable", y="value", hue="Algoritma", marker="s", markersize=10, linewidth=3)
    plt.title("Algoritmaların Performans Karşılaştırması")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(os.path.join(output_dir, 'final_performans_yarisi.png'), dpi=300)
    plt.close()

    # 5. Kriter: Hata Dağılımı (Confusion Matrix Karşılaştırması)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for i, (name, model) in enumerate(models.items()):
        preds = model.predict(X_test)
        cm = confusion_matrix(y_test, preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu', ax=axes[i])
        axes[i].set_title(f'{name} Hata Matrisi')
        axes[i].set_xlabel('Tahmin Edilen')
        axes[i].set_ylabel('Gerçek Durum')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'final_hata_matrisi_kiyaslama.png'), dpi=300)
    plt.close()

    print(f"\n[BILGI] Tüm analiz grafikleri '{output_dir}' klasörüne kaydedildi.")
    return results_df


if __name__ == "__main__":
    train_and_compare_advanced()