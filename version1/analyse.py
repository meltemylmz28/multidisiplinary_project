import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def get_cleaned_data():
    # Veri Yükleme
    df = pd.read_csv('../newdata.csv')

    # Aşama 0: Temizlik
    df = df.drop_duplicates()

    # Eksik Veriler: Medyan ile doldurma
    df.fillna(df.median(), inplace=True)

    # Aykırı Değer Analizi (IQR)
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    df_cleaned = df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

    # Normalizasyon (Min-Max Ölçeklendirme)
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_cleaned), columns=df_cleaned.columns)

    # Hedef Değişken Manipülasyonu (Aşırı ısınma = Arıza)
    threshold = df_normalized['stator_winding'].quantile(0.90)
    df_normalized['Motor_Health'] = (df_normalized['stator_winding'] > threshold).astype(int)

    return df_normalized


if __name__ == "__main__":
    data = get_cleaned_data()
    print(f"Veri başarıyla hazırlandı. Toplam satır: {len(data)}")