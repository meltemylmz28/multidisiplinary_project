import matplotlib.pyplot as plt
import seaborn as sns
from version1.analyse import get_cleaned_data

def run_visualizations():
    df = get_cleaned_data()
    sns.set_theme(style="white") # Arka plan- beyaz

    # Korelasyon Isı Haritası
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt='.2f', linewidths=0.5)
    plt.title("Sensör Özellikleri Korelasyon Matrisi")
    plt.savefig('korelasyon_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Sıcaklık Dağılımı
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Motor_Health', y='stator_winding', data=df, palette="spring")
    plt.title("Sağlık Durumuna Göre Sıcaklık Dağılımı")
    plt.savefig('saglik_sicaklik_violin.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Akım-Hız İlişkisi
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='i_q', y='motor_speed', hue='Motor_Health', data=df, palette="flare", alpha=0.5)
    plt.title("Akım ve Hızın Motor Sağlığı Üzerindeki Etkisi")
    plt.savefig('akim_hiz_scatter.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    run_visualizations()