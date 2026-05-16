import pandas as pd
from models import train_ml_process


def main():
    print("=== ELEKTRİKLİ MOTOR SAĞLIĞI ANALİZ MERKEZİ ===")

    try:
        # Modelleri eğit ve sonuçları döndür
        results, feature_names = train_ml_process()

        print("\n" + "=" * 50)
        print("AŞAMA 3: MODEL PERFORMANS DEĞERLENDİRMESİ")
        print(results["report"])
        print(f"XGBoost ROC-AUC Skoru: {results['auc']:.4f}")
        print("=" * 50)

        # Aşama 4: Özellik Önem Sıralaması
        importance_df = pd.DataFrame({
            'Sensör Parametresi': feature_names,
            'Önem Skoru': results["importances"]
        }).sort_values(by='Önem Skoru', ascending=False)

        print("\n[Aşama 4] Endüstriyel PdM İçin Kritik Parametreler:")
        print(importance_df)
        print("\nBilgi: Tüm grafikler yüksek çözünürlüklü olarak kaydedildi.")

    except Exception as e:
        print(f"Hata oluştu: {e}")


if __name__ == "__main__":
    main()