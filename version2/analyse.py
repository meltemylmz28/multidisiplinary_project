import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os


def get_cleaned_data(file_path='newdata.csv', generate_report_visual=True):
    # Terminal ana dizindeyken çalıştırdığın için newdata.csv direkt okunur.
    # Eğer hata alırsan 'file_path' kısmını '../newdata.csv' olarak güncelleyebilirsin.
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        # Kod version2 içindeyken manuel çalıştırılırsa bir üst dizine bakar
        df = pd.read_csv(os.path.join('..', file_path))

    # 1. Eksik Veri ve Tekrarlar
    df = df.drop_duplicates()
    df = df.fillna(df.median())

    # 2. ÖZELLİK MÜHENDİSLİĞİ: Tahmini Güç Kaybı (Power Loss)
    # Formül: P = (u_d * i_d) + (u_q * i_q)
    df['power_loss'] = (df['u_d'] * df['i_d']) + (df['u_q'] * df['i_q'])

    # 3. Hedef Değişken Tanımlama (%90 Eşik Değeri)
    limit = df['stator_winding'].quantile(0.90)
    df['Motor_Health'] = (df['stator_winding'] > limit).astype(int)

    # 4. IQR ile Aykırı Değer Temizliği
    cols_to_fix = ['i_q', 'i_d', 'u_q', 'u_d', 'motor_speed', 'ambient', 'coolant', 'torque', 'power_loss']
    for col in cols_to_fix:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        df = df[~((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR)))]

    # RAPOR İÇİN ÖZEL GÖRSEL ÜRETİMİ (Sürüm 2 Çıktıları)
    if generate_report_visual:
        # Görsel kayıt yolunu senin belirttiğin meltem_v2 klasörüne göre ayarlıyoruz
        output_dir = os.path.join('version2', 'meltem_v2')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        plt.figure(figsize=(10, 6))
        sns.boxplot(x='Motor_Health', y='power_loss', data=df, palette='Set2')
        plt.title('Motor Sağlık Durumuna Göre Tahmini Güç Kaybı (Power Loss) Dağılımı')
        plt.xlabel('Motor Durumu (0: Sağlıklı, 1: Arızalı)')
        plt.ylabel('Hesaplanan Güç Kaybı')

        save_path = os.path.join(output_dir, 'ozellik_muhendisligi_kaniti.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"[BILGI] Sürüm 2 analizi '{save_path}' adresine başarıyla kaydedildi.")

    # 5. Min-Max Normalizasyon
    for col in cols_to_fix:
        df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())

    return df


if __name__ == "__main__":
    get_cleaned_data()