from statistics import quantiles

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import glob
import yfinance as yf

warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

#---------------------------
# VERİ SETİ OKUMA
#---------------------------


ptf_df = pd.read_csv("data_s/Piyasa_Takas_Fiyati(PTF).csv", sep=";")
yuk_df = pd.read_csv("data_s/Yuk_Tahmin_Plani.csv", sep=";")
kgup_files = [
    "data_s/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-01012025-01042025.csv",
    "data_s/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-02042025-02072025.csv",
    "data_s/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-03072025-03102025.csv",
    "data_s/Kesinlesmis_Gunluk_Uretim_Plani_(KGUP)-03102025-30112025.csv"
]
kgup_dfs = []
for file in kgup_files:
    df = pd.read_csv(file,sep=";")
    kgup_dfs.append(df)
kgup_df = pd.concat(kgup_dfs)
kgup_df = kgup_df.drop_duplicates(subset=['Tarih', 'Saat']).reset_index(drop=True)

#---------------------------
# Tarih Formatı Değiştirme ve Merge İşlemleri
#---------------------------

# 1.Merge İşlemi
merge_1 = pd.merge(ptf_df,yuk_df, on=["Tarih","Saat"] ,how="inner" )

# 1.Tarih Format Değişimi
merge_1['Tarih'] = pd.to_datetime(merge_1['Tarih'], dayfirst=True, errors='coerce').dt.normalize()
kgup_df['Tarih'] = pd.to_datetime(kgup_df['Tarih'], dayfirst=True, errors='coerce').dt.normalize()

# Saat Formatı Eşitleme
merge_1['Saat'] = merge_1['Saat'].astype(str).str.strip().str[:5]
kgup_df['Saat'] = kgup_df['Saat'].astype(str).str.strip().str[:5]

# 2.Merge İşlemi
df_final = pd.merge(merge_1, kgup_df, on=["Tarih","Saat"] ,how="inner" ).reset_index(drop=True)
df_final = df_final.sort_values(by=["Tarih", "Saat"]).reset_index(drop=True)



#---------------------------
# Değişken Tiplerini Düzeltme
#---------------------------
def clean_currency(x):
    if isinstance(x, str):
        # 1. Önce binlik ayracı olan NOKTALARI tamamen sil
        x = x.replace('.', '')
        # 2. Sonra ondalık ayracı olan VİRGÜLLERİ noktaya çevir
        x = x.replace(',', '.')
    return float(x)
obcejt_to_str = [col for col in df_final.columns if col not in ['Tarih', 'Saat']]
for col in obcejt_to_str:
    df_final[col] = df_final[col].apply(clean_currency)

# df_final.to_csv('EPIAS_Project_Dataset.csv', index=False)

#---------------------------
# Dolar Kurunu Ekleme(Yahoo)
#---------------------------
start_date = df_final['Tarih'].min()
end_date = df_final['Tarih'].max()

usd_data = yf.download('TRY=X', start=start_date, end=end_date + pd.Timedelta(days=5))
usd_data = usd_data['Close'].reset_index()
usd_data.columns = ['Tarih', 'Dolar_Kuru']

usd_data['Tarih'] = pd.to_datetime(usd_data['Tarih']).dt.normalize()
usd_data['Tarih'] = usd_data['Tarih'].dt.tz_localize(None)

#---------------------------
# BOTAS Veri Ekleme
#---------------------------

# Kıyaslama yapacağımız sınır tarihi belirliyoruz
sinir_tarih = pd.Timestamp('2025-07-01')

# 2. ADIM: List Comprehension ile yeni değişkeni oluşturma
# Mantık: [ (Koşul sağlanırsa değer) if (koşul) else (sağlanmazsa değer) for x in (sütun) ]

df_final['dogalgaz_fiyatlari_Mwh'] = [
    1127.82 if tarih <= sinir_tarih else 1409.77  # 1500 yerine sonraki tarihlerin fiyatını yazmalısın
    for tarih in df_final['Tarih']
]

print(df_final)
# CHECK
df_final[df_final['Tarih'] == '2025-08-20']['dogalgaz_fiyatlari_Mwh'].values[0]
df_final[df_final['Tarih'] == '2025-05-20']['dogalgaz_fiyatlari_Mwh'].values[0]
df_final["Tarih"]

# Datelerdeki boşluk dolar değerlerini doldurduk
all_dates = pd.DataFrame({'Tarih': pd.date_range(start=start_date, end=end_date, freq='D')})
all_dates['Tarih'] = all_dates['Tarih'].dt.normalize().dt.tz_localize(None)

usd_data = pd.merge(all_dates, usd_data, on='Tarih', how='left')

usd_data['Dolar_Kuru'] = usd_data['Dolar_Kuru'].ffill().bfill()

# Ana veriye ekle
df_final = pd.merge(df_final, usd_data, on='Tarih', how='left')




#---------------------------
# Gereksiz Değişkenleri Veri Setinden Atma
#---------------------------
drop_list = [
    'PTF (USD/MWh)', 'PTF (EUR/MWh)',
    'Toplam(MWh)',
    'Nafta', 'Fueloil',
    'Taş Kömür', 'Diğer'
]
existing_drop = [col for col in drop_list if col in df_final.columns]
if existing_drop:
    df_final.drop(columns=existing_drop, inplace=True)

#---------------------------
# KAYIT
#---------------------------
df_final.to_csv('EPIAS_.csv', index=False)
df_final.head()


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

def degisken_analiz(dataframe, cat_th=10, car_th=20):
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
    target_summary_with_numeric(df_final, "PTF (TL/MWh)", col)
#---------------------------
# Korelasyon
#---------------------------

df_final[num_cols].corr()

f, ax = plt.subplots(figsize=[18,13])
sns.heatmap(df_final[num_cols].corr(), annot=True, fmt=".2f", ax=ax, cmap="magma")
ax.set_title("Correlation Matrix", fontsize=20)
plt.show(block = True)









