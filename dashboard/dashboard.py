import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

df = pd.read_csv('bike_sharing_dataset_day_clean.csv')

min_date = df['dteday'].min()
max_date = df['dteday'].max()
 
with st.sidebar:
    st.image('logo_sepeda.png')
    st.subheader('Rentang Waktu')
    start_date = st.date_input(
        label='Dari',
        min_value=min_date,
        value=min_date
    )

    end_date = st.date_input(
        label='Sampai',
        max_value=max_date,
        value=max_date
    )

main_df = df[(df['dteday'] >= str(start_date)) & (df['dteday'] <= str(end_date))]

st.header('Peminjaman Sepeda Tahun 2011 - 2012')

tab1, tab2 = st.tabs(["Dashboard", "Analisis"])
 
with tab1:
    st.subheader('Total Peminjaman Sepeda')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        total_casual = main_df.casual.sum()
        st.metric("Total Peminjaman Casual", value=total_casual)
    
    with col2:
        total_registered = main_df.registered.sum() 
        st.metric("Total Peminjaman Registered", value=total_registered)

    with col3:
        total_count = main_df.cnt.sum() 
        st.metric("Total Peminjaman Keseluruhan", value=total_count)

    st.subheader('Peminjaman Sepeda per Bulan')

    pivot_table = main_df.groupby(['yr', 'mnth'])['cnt'].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.lineplot(x='mnth', y='cnt', hue='yr', data=pivot_table, marker='o', ax=ax)
    ax.set_xlabel('Bulan')
    ax.set_ylabel('')
    ax.set_xticks(np.arange(1, 13))
    ax.legend(title='Tahun', labels=['2011', '2012'])

    st.pyplot(fig)

    with st.expander("Penjelasan"):
        st.write("Jumlah peminjaman sepeda terbanyak terjadi pada bulan September 2012, sedangkan jumlah peminjaman sepeda terkecil terjadi pada bulan Januari 2011")
        
 
with tab2:
    st.subheader('Korelasi')

    corr_df = main_df.select_dtypes(include=['number'])
    fig1, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(corr_df.corr(), annot=True, cmap='coolwarm', fmt=".2f")

    st.pyplot(fig1)

    with st.expander("Penjelasan"):
        st.write("Korelasi yang terbesar terhadap kolom 'cnt' adalah pada kolom 'tmp' dan 'atmp' dimana hasilnya menunjukan korelasi positif yang artinya semakin tinggi tmp dan atmp maka semakin banyak pula peminjaman sepeda yang terjadi")

    st.subheader('Scatter Plot')

    fig3, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    sns.scatterplot(x='temp', y='cnt', data=df, ax=ax[0])
    ax[0].set_ylabel('')
    ax[0].set_xlabel('Temp')
    ax[0].set_title('Hubungan antara temp dan jumlah peminjaman', loc="center")

    sns.scatterplot(x='temp', y='cnt', data=df, ax=ax[1])
    ax[1].set_ylabel('')
    ax[1].set_xlabel('Atemp')
    ax[1].set_title('Hubungan antara atemp dan jumlah peminjaman', loc="center")
    
    st.pyplot(fig3)

    with st.expander("Penjelasan"):
        st.write("Temp dan Atmp mempengaruhi Jumlah Peminjaman yang menunjukan korelasi positif, artinya semakin tinggi temp dan atemp maka kemungkinan semakin banyak pula peminjaman sepeda yang terjadi")

    low_temp_threshold = df['temp'].quantile(0.2)
    high_temp_threshold = df['temp'].quantile(0.8)

    main_df['temperature_group'] = pd.cut(
        main_df['temp'], 
        bins=[-1, low_temp_threshold, high_temp_threshold, 1], 
        labels=['Rendah', 'Sedang', 'Tinggi'], 
        include_lowest=True
    )

    st.subheader("Analisis Klaster Berdasarkan Temperature")

    st.write('Analisis Clustering dilakukan dengan tujuan untuk memahami lebih dalam bagaimana kategori temperatur memengaruhi pola peminjaman sepeda dengan membagi data temperatur ke dalam tiga kelompok: rendah, sedang, dan tinggi.')

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=main_df, x='temp', y='cnt', hue='temperature_group', palette='viridis', ax=ax)
    ax.set_xlabel('Temperature')
    ax.set_ylabel('')
    ax.legend(title='Temperature', labels=['Rendah', 'Sedang', 'Tinggi'])

    st.pyplot(fig)

    with st.expander("Detail"):
        st.write(main_df.groupby("temperature_group").agg({
            'temp': ['min', 'max'],
            'atemp': ['min', 'max'],
            'cnt': sum
        }))

        st.write('Analisis ini dapat mengidentifikasi apakah peminjaman lebih tinggi pada temperatur rendah, sedang, atau tinggi. Dan hasilnya ternyata peminjaman lebih banyak dilakukakan disaat temperaturenya di keadaan sedang atau sekitar 0.3165 - 0.6858 satuan suhu.')