import matplotlib.pyplot as plt
import seaborn as sns
import os
from analyse import get_cleaned_data

def generate_advanced_visuals():
    # Çıktı klasörünü tanımla
    output_dir = os.path.join('version2', 'meltem_v2')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("[1/3] Veri görselleştirme için hazırlanıyor (9 Sütunlu Yapı)...")
    # analyse.py üzerinden temizlenmiş ve power_loss eklenmiş veriyi alıyoruz
    df = get_cleaned_data(generate_report_visual=False)

    sns.set_theme(style="whitegrid", palette="muted")

    # 1. GÜÇ KAYBI VE MOTOR SAĞLIĞI ANALİZİ (Violin Plot)
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Motor_Health', y='power_loss', data=df, inner="quart", palette="Set1")
    plt.title("Motor Sağlık Durumuna Göre Güç Kaybı (Power Loss) Dağılımı")
    plt.xticks([0, 1], ['Sağlıklı (0)', 'Arızalı (1)'])
    plt.xlabel("Motor Durumu")
    plt.ylabel("Normalize Edilmiş Güç Kaybı")
    plt.savefig(os.path.join(output_dir, 'v2_power_loss_analizi.png'), dpi=300)
    plt.close()

    # 2. TORK - GÜÇ KAYBI - SICAKLIK İLİŞKİSİ (Scatter)
    plt.figure(figsize=(12, 7))
    scatter = plt.scatter(df['torque'], df['stator_winding'],
                          c=df['power_loss'], cmap='coolwarm', alpha=0.6)
    plt.colorbar(scatter, label='Güç Kaybı (Power Loss)')
    plt.title("Tork ve Sıcaklık Ekseninde Güç Kaybı Dağılımı")
    plt.xlabel("Normalize Edilmiş Tork")
    plt.ylabel("Normalize Edilmiş Stator Sıcaklığı")
    plt.savefig(os.path.join(output_dir, 'v2_torque_power_temp_bubble.png'), dpi=300)
    plt.close()

    # 3. GENİŞLETİLMİŞ KORELASYON ISI HARİTASI (9 Parametre)
    plt.figure(figsize=(12, 10))
    cols = ['i_q', 'i_d', 'u_q', 'u_d', 'motor_speed', 'ambient', 'coolant', 'torque', 'power_loss', 'Motor_Health']
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='RdYlBu', fmt='.2f', linewidths=0.5)
    plt.title("9 Parametreli Genişletilmiş Korelasyon Matrisi")
    plt.savefig(os.path.join(output_dir, 'v2_kapsamli_heatmap.png'), dpi=300)
    plt.close()

    print(f"[3/3] Görseller '{output_dir}' klasörüne 'v2_' ön ekiyle başarıyla kaydedildi.")

if __name__ == "__main__":
    generate_advanced_visuals()