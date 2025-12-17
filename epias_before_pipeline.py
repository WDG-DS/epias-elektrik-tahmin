import locale
import datetime
from statistics import quantiles
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import yfinance as yf
from sklearn.exceptions import ConvergenceWarning
from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.outliers_influence import variance_inflation_factor
import pandas as pd

# ---------------------------
# AYARLAR
# ---------------------------


warnings.filterwarnings("ignore")
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action="ignore", category=ConvergenceWarning)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# ---------------------------
# VERİ OKUMA
# ---------------------------
"""df_final = pd.read_csv("data_s/data_set.csv")"""
df_final = pd.read_excel("data_s/data_set_ex.xlsx")
df_final_c = df_final.copy()
# ---------------------------
# DEĞİŞKEN TİPİ DÜZELTME
# --------------------------

def clean_currency(x):
    if isinstance(x, str):
        x = x.replace('.', '').replace(',', '.')
    return float(x)

object_to_float = [col for col in df_final.columns if col not in ['Tarih', 'Saat', 'Zaman']]
for col in object_to_float:
    df_final[col] = df_final[col].apply(clean_currency)

df_final['Tarih'] = pd.to_datetime(df_final['Tarih'], format='%d.%m.%Y').dt.normalize()
df_final['Zaman'] = pd.to_datetime(df_final['Tarih'].astype(str) + ' ' + df_final['Saat'].astype(str))



df_final['Tarih'] = pd.to_datetime(df_final['Tarih'], format='%d.%m.%Y').dt.normalize()
df_final['Saat'] = pd.to_datetime(df_final['Saat']).dt.time
df_final.head()
df_final.info()


# ---------------------------
# DOLAR VE BOTAŞ EKLEME
# ---------------------------

# ---------------------------
# DOLAR KURU (YAHOO)
# ---------------------------
start_date = df_final['Tarih'].min()
end_date = df_final['Tarih'].max()
usd_data = yf.download('TRY=X', start=start_date, end=end_date + pd.Timedelta(days=5))
usd_data = usd_data[['Close']].reset_index()
usd_data.columns = ['Tarih', 'Dolar_Kuru']
usd_data['Tarih'] = pd.to_datetime(usd_data['Tarih']).dt.normalize()
usd_data['Tarih'] = usd_data['Tarih'].dt.tz_localize(None)
# Eksik günleri doldur
all_dates = pd.DataFrame({'Tarih': pd.date_range(start=start_date, end=end_date, freq='D')})
all_dates['Tarih'] = all_dates['Tarih'].dt.normalize()
usd_data = pd.merge(all_dates, usd_data, on='Tarih', how='left')
usd_data['Dolar_Kuru'] = usd_data['Dolar_Kuru'].ffill().bfill()
# Ana veriye ekle
df_final = pd.merge(df_final, usd_data, on='Tarih', how='left')

# ---------------------------
# BOTAŞ DOĞALGAZ VERİSİ
# ---------------------------
#
df_final['dogalgaz_fiyatlari_Mwh'] = 1692.00
df_final.loc[df_final['Tarih'] >= '2024-07-02', 'dogalgaz_fiyatlari_Mwh'] = 1127.82
df_final.loc[df_final['Tarih'] >= '2025-07-01', 'dogalgaz_fiyatlari_Mwh'] = 1409.77

df_final.describe().T

##############################################################
#---------------------------
# EDA
#---------------------------

def data_summary(dataframe, head=5):
    print("######### Shape ########")
    print(dataframe.shape)
    print("######### Type ########")
    print(dataframe.dtypes)
    print("######### Head #######")
    print(dataframe.head(head))
    print("######### Tail #######")
    print(dataframe.tail(head))
    print("######### Nan #######")
    print(dataframe.isnull().sum())

data_summary(df_final)


df_final.loc[df_final['Güneş'] < 0, 'Güneş'] = 0

print("Güneş değeri dağılımı:")
print("Negatif (<0):", (df_final['Güneş'] < 0).sum())
print("Sıfır (=0):", (df_final['Güneş'] == 0).sum())
print("Pozitif (>0):", (df_final['Güneş'] > 0).sum())
df_final['Güneş'] = df_final['Güneş'].clip(lower=0)
def degisken_analiz(dataframe, cat_th=2, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car
cat_cols, num_cols, cat_but_car = degisken_analiz(df_final)

def numeric_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)
    if plot:
        dataframe[numerical_col].hist(bins=20)
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show()

print("\n--- NUMERİK DEĞİŞKENLERİN DAĞILIMI ---")
for col in num_cols:
    numeric_summary(df_final, numerical_col=col, plot=True)


#---------------------------
# Target
#---------------------------

def target_summary_with_numeric(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")

for col in num_cols:
    target_summary_with_numeric(df_final, "PTF (TL/MWH)", col)

#---------------------------
# Korelasyon
#---------------------------

df_final[num_cols].corr()

f, ax = plt.subplots(figsize=[18,13])
sns.heatmap(df_final[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block = True)

#---------------------------
# Grafik ile Analiz
#---------------------------
def check_physical_integrity(df):
    print("🕵️‍♂️ Fiziksel Tutarlılık Kontrolü Yapılıyor...")

    # 1. Negatif Üretim Kontrolü (İmkansız Olay)
    prod_cols = ['Rüzgar', 'Güneş', 'Doğalgaz', 'Barajlı', 'Linyit']
    # Veri setinde olanları seç
    existing_cols = [c for c in prod_cols if c in df.columns]

    for col in existing_cols:
        negatives = df[df[col] < 0]
        if len(negatives) > 0:
            print(f"⚠️ UYARI: {col} sütununda {len(negatives)} adet negatif değer var! 0'a eşitleniyor.")
            df.loc[df[col] < 0, col] = 0
        else:
            print(f"✅ {col}: Temiz (Negatif yok).")

    # 2. PTF Kontrolü (Hata vs Gerçek Ayrımı)
    # Tavan Fiyatı manuel belirleyebiliriz (Örn: 2025 için 5000 TL diyelim, teyit etmen lazım)
    MAX_PRICE_LIMIT = 6000
    MIN_PRICE_LIMIT = 0

    errors = df[(df['PTF (TL/MWH)'] > MAX_PRICE_LIMIT) | (df['PTF (TL/MWH)'] < MIN_PRICE_LIMIT)]
    if len(errors) > 0:
        print(f"🚨 KRİTİK: PTF sütununda {len(errors)} adet mantıksız (Tavan üstü veya Negatif) değer var!")
        # Bunları baskılamıyoruz, SİLİYORUZ. Çünkü gerçek mi hata mı bilemeyiz.
        # df = df.drop(errors.index) # İstersen silebilirsin
    else:
        print("✅ PTF: Mantıksız uç değer (Error) görünmüyor.")

    print("-" * 30)
    return df
check_physical_integrity(df_final)


def plot_all_boxplots(df):
    # Stil ayarları
    sns.set_theme(style="whitegrid")

    # 1. GRUP: Fiyat Değişkenleri (Küçük Ölçekli)
    # PTF, Dolar ve Doğalgaz Fiyatları benzer ölçeklerdedir.
    price_cols = ['PTF (TL/MWH)', 'Dolar_Kuru', 'dogalgaz_fiyatlari_Mwh']

    # 2. GRUP: Büyük Ölçekli Üretim ve Yük
    # Yük tahmini ve ana üretim kalemleri (Barajlı, Doğalgaz Üretimi)
    large_scale_cols = ['Yük Tahmin Planı (MWh)', 'Doğalgaz', 'Barajlı', 'İthal Kömür']

    # 3. GRUP: Yenilenebilir ve Diğer Üretimler
    # Rüzgar, Güneş, Akarsu, Jeotermal gibi daha orta ölçekli üretimler
    renewable_cols = ['Rüzgar', 'Güneş', 'Akarsu', 'Linyit', 'Jeotermal', 'Biyokütle', 'Fuel Oil']

    # Grafiklerin çizilmesi
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))

    # Plot 1: Fiyatlar
    sns.boxplot(data=df[price_cols], ax=axes[0], palette="Set2")
    axes[0].set_title('Grup 1: Fiyat Bazlı Değişkenler', fontsize=15)

    # Plot 2: Büyük Ölçekli Veriler
    sns.boxplot(data=df[large_scale_cols], ax=axes[1], palette="Set1")
    axes[1].set_title('Grup 2: Yük ve Büyük Ölçekli Üretimler', fontsize=15)

    # Plot 3: Yenilenebilir ve Diğerleri
    sns.boxplot(data=df[renewable_cols], ax=axes[2], palette="Pastel1")
    axes[2].set_title('Grup 3: Yenilenebilir Enerji ve Diğer Üretimler', fontsize=15)
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

plot_all_boxplots(df_final)

df_final['PTF (TL/MWH)'].describe().T
sayi = (df_final['PTF (TL/MWH)'] == 99999.000).sum()
print(f"99999.000 değeri {sayi} kez geçiyor.")

#---------------------------
# NORMALLİK TESTİ
#---------------------------
# Veriyi normalize ederek K-S testi yapalım
ptf_clean = df_final['PTF (TL/MWH)'].dropna()
ks_stat, p_value_ks = stats.kstest((ptf_clean - ptf_clean.mean()) / ptf_clean.std(), 'norm')

print(f"K-S Testi p-değeri: {p_value_ks}")

plt.figure(figsize=(10, 6))
# Gerçek verinin dağılımı
sns.histplot(df_final['PTF (TL/MWH)'], kde=True, stat="density", color='skyblue', label='Gerçek Dağılım')

# İdeal Normal Dağılım eğrisi (Karşılaştırma için)
mu, std = df_final['PTF (TL/MWH)'].mean(), df_final['PTF (TL/MWH)'].std()
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'r', linewidth=2, label='Teorik Normal Dağılım')

plt.title('PTF Dağılımı vs Teorik Normal Dağılım')
plt.legend()
plt.show()

fig = sm.qqplot(df_final['PTF (TL/MWH)'].dropna(), line='s')
plt.title('PTF İçin Q-Q Plot')
plt.show()

print(f"Skewness (Çarpıklık): {df_final['PTF (TL/MWH)'].skew()}")
print(f"Kurtosis (Basıklık): {df_final['PTF (TL/MWH)'].kurt()}")

# ADF Testini çalıştır
# autolag='AIC' parametresi en iyi gecikme (lag) sayısını otomatik seçer
adf_test = adfuller(df_final['PTF (TL/MWH)'].dropna(), autolag='AIC')

print(f"ADF İstatistiği: {adf_test[0]}")
print(f"p-değeri: {adf_test[1]}")
print("Kritik Değerler:")
for key, value in adf_test[4].items():
    print(f"\t{key}: {value}")

if adf_test[1] <= 0.05:
    print("\nSonuç: p <= 0.05. H0 reddedilir. Seri DURAĞANDIR.")
else:
    print("\nSonuç: p > 0.05. H0 reddedilemez. Seri DURAĞAN DEĞİLDİR (Trend var).")

# Bağımsız değişken listesi (PTF hariç, sadece sayısal olanlar)
independent_cols = [col for col in df_final.columns if
                    col not in ['PTF (TL/MWH)', 'Tarih', 'Zaman'] and df_final[col].dtype in ['float64', 'int64']]

adf_results = []

for col in independent_cols:
    # NaN değerleri temizleyerek testi çalıştır
    series = df_final[col].dropna()
    result = adfuller(series, autolag='AIC')

    p_value = result[1]
    is_stationary = "Evet" if p_value <= 0.05 else "Hayır"

    adf_results.append({
        'Değişken': col,
        'ADF İstatistiği': round(result[0], 4),
        'p-değeri': p_value,
        'Durağan mı?': is_stationary
    })

# Sonuçları DataFrame olarak görselleştir
adf_df = pd.DataFrame(adf_results)
print(adf_df)




# Durağan olmayan ve sınırda olan değişkenleri görselleştirelim
cols_to_plot = ['Dolar_Kuru', 'dogalgaz_fiyatlari_Mwh', 'Akarsu', 'Jeotermal']

fig, axes = plt.subplots(len(cols_to_plot), 1, figsize=(12, 15))

for i, col in enumerate(cols_to_plot):
    axes[i].plot(df_final.index, df_final[col], color='tab:blue')
    axes[i].set_title(f'{col} - Zaman Serisi Grafiği (Durağanlık Kontrolü)')
    axes[i].set_ylabel('Değer')
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()




# 1. Grup: Fiyat ve Maliyet Volatilitesi (PTF, Dolar, Gaz Fiyatı)
# Not: Doların ham fiyatı trend izlese de, volatilitesi ekonomik risk dönemlerini gösterir.
price_maliye_cols = ['PTF (TL/MWH)', 'Dolar_Kuru', 'dogalgaz_fiyatlari_Mwh']

# 2. Grup: Esnek ve Baz Yük Üretim Volatilitesi (Fosil Yakıtlar)
# Doğalgaz ve Kömür santrallerindeki ani oynaklıklar sistemdeki arz şoklarını temsil eder.
fosil_cols = ['Doğalgaz', 'Linyit', 'İthal Kömür']

# 3. Grup: Yenilenebilir Enerji Volatilitesi
# Akarsu ve Rüzgar'ın yanına Güneş'i de ekliyoruz (Bulutluluk etkisi oynaklık yaratır).
yenilenebilir_cols = ['Akarsu', 'Rüzgar', 'Güneş']

# Fonksiyon: Gruplar için hareketli standart sapma çizdirme
def plot_grouped_volatility(df, columns, title):
    vol_data = df[columns].rolling(window=24).std()
    vol_data.plot(figsize=(12, 5), title=title)
    plt.ylabel("Standart Sapma (24s)")
    plt.grid(True, alpha=0.3)
    plt.show()

# Uygulama
plot_grouped_volatility(df_final, price_maliye_cols, "Fiyat ve Döviz Volatilitesi")
plot_grouped_volatility(df_final, fosil_cols, "Fosil Yakıt Üretim Volatilitesi")
plot_grouped_volatility(df_final, yenilenebilir_cols, "Yenilenebilir Enerji Üretim Volatilitesi")




# 1. Sayısal sütunları seçelim
numerical_cols = df_final.select_dtypes(include=[np.number]).columns

# 2. Spearman Korelasyon Matrisini hesaplayalım
spearman_corr = df_final[numerical_cols].corr(method='spearman')

# 3. Sadece PTF ile olan ilişkileri alıp sıralayalım
ptf_corr = spearman_corr[['PTF (TL/MWH)']].sort_values(by='PTF (TL/MWH)', ascending=False)

# 4. Görselleştirme
plt.figure(figsize=(8, 12))
sns.heatmap(ptf_corr, annot=True, cmap='RdYlGn', fmt=".2f", center=0)
plt.title("Değişkenlerin PTF ile Spearman Korelasyonu")
plt.show()





# 1. Sadece bağımsız değişkenleri seçelim (Bağımlı değişken PTF ve Tarih hariç)
X = df_final.drop(['PTF (TL/MWH)'], axis=1).select_dtypes(include=[np.number])

# 2. VIF için sabit (constant) eklenmesi önerilir (opsiyonel ama sağlıklı sonuç verir)
# Ancak VIF kütüphanesi genelde ham veriyle de çalışır.
vif_data = pd.DataFrame()
vif_data["Değişken"] = X.columns

# 3. Her değişken için VIF değerini hesapla
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

print(vif_data.sort_values(by="VIF", ascending=False))




# Korelasyonu düşük ama VIF'i çok yüksek olanları eleyerek testi tekrarlayalım
# Örn: Biyokütle, Jeotermal ve Akarsu'yu çıkarıyoruz
drop_list = ['Biyokütle', 'Jeotermal', 'Akarsu']
X_reduced = X.drop(columns=drop_list)

vif_reduced = pd.DataFrame()
vif_reduced["Değişken"] = X_reduced.columns
vif_reduced["VIF"] = [variance_inflation_factor(X_reduced.values, i) for i in range(len(X_reduced.columns))]

print("Gereksiz Değişkenler Elendikten Sonra VIF:")
print(vif_reduced.sort_values(by="VIF", ascending=False))



from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X) # Orijinal X verisi (tüm bağımsız değişkenler)

vif_scaled = pd.DataFrame()
vif_scaled["Değişken"] = X.columns
vif_scaled["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]

print("StandardScaler Sonrası VIF Değerleri:")
print(vif_scaled.sort_values(by="VIF", ascending=False))



# Durağan olmayan ve yüksek VIF verenlerin 1. derece farkını alalım
X_diff = X.diff().dropna()

vif_diff = pd.DataFrame()
vif_diff["Değişken"] = X_diff.columns
vif_diff["VIF"] = [variance_inflation_factor(X_diff.values, i) for i in range(len(X_diff.columns))]

print("Fark Alma (Differencing) Sonrası VIF Değerleri:")
print(vif_diff.sort_values(by="VIF", ascending=False))













