import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import numpy as np
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Memuat data curah hujan
rainfall_data = pd.read_csv('rainfall_data.csv')

# Memuat shapefile yang berisi area rawan banjir
flood_areas = gpd.read_file('flood_areas.shp')

# Memfilter data untuk wilayah NTB
ntb_rainfall_data = rainfall_data[rainfall_data['province'] == 'Nusa Tenggara Barat']

# Mengonversi latitude dan longitude menjadi objek Point
ntb_rainfall_data['geometry'] = ntb_rainfall_data.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
ntb_rainfall_data = gpd.GeoDataFrame(ntb_rainfall_data, geometry='geometry')

# Menggabungkan spasial untuk mengidentifikasi titik-titik di dalam area rawan banjir
ntb_rainfall_data = gpd.sjoin(ntb_rainfall_data, flood_areas, how='left', op='within')

# Menghapus titik-titik di luar area rawan banjir
ntb_rainfall_data = ntb_rainfall_data[~ntb_rainfall_data['index_right'].isnull()]

# Pra-pemrosesan data curah hujan
rainfall_features = ntb_rainfall_data[['rainfall', 'temperature', 'humidity']]
scaler = MinMaxScaler(feature_range=(0, 1))
rainfall_features_scaled = scaler.fit_transform(rainfall_features)

# Membagi data menjadi set pelatihan dan pengujian
X_train, X_test, y_train, y_test = train_test_split(rainfall_features_scaled, rainfall_features_scaled[:, 0], test_size=0.2, shuffle=False)

# Membuat urutan untuk pelatihan LSTM
def create_sequences(data, seq_length):
    X = []
    y = []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10
X_train, y_train = create_sequences(X_train, seq_length)
X_test, y_test = create_sequences(X_test, seq_length)

# Membangun model LSTM
model = Sequential()
model.add(LSTM(units=256, input_shape=(seq_length, X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(BatchNormalization())
model.add(LSTM(units=256, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Mendefinisikan callback untuk early stopping dan model checkpoint
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

# Melatih model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, verbose=1, validation_data=(X_test, y_test), callbacks=[early_stopping, model_checkpoint])

# Memuat model terbaik
model.load_weights('best_model.h5')

# Mendefinisikan fungsi untuk mengambil data cuaca dari API yang diberikan
def fetch_weather_data(latitude, longitude, api_key):
    url = f'http://api.openweathermap.org/data/2.5/weather?lat={latitude}&lon={longitude}&appid={api_key}'
    response = requests.get(url)
    if response.status_code == 200:
        weather_data = response.json()
        return weather_data
    else:
        return None

# Kunci API dari API yang diberikan
api_key = '642a205c2a212f2e0752ce580367dc06'

# Mendapatkan data cuaca untuk setiap lokasi di NTB
for index, row in ntb_rainfall_data.iterrows():
    latitude = row['latitude']
    longitude = row['longitude']
    weather_data = fetch_weather_data(latitude, longitude, api_key)
    if weather_data is not None:
        # Mengambil data curah hujan, suhu, dan kelembaban dari data cuaca
        rainfall = weather_data['rain']['1h'] if 'rain' in weather_data and '1h' in weather_data['rain'] else 0
        temperature = weather_data['main']['temp']
        humidity = weather_data['main']['humidity']
        # Memperbarui ntb_rainfall_data dengan data cuaca baru
        ntb_rainfall_data.at[index, 'rainfall'] = rainfall
        ntb_rainfall_data.at[index, 'temperature'] = temperature
        ntb_rainfall_data.at[index, 'humidity'] = humidity

# Pra-pemrosesan data curah hujan yang diperbarui
rainfall_features_updated = ntb_rainfall_data[['rainfall', 'temperature', 'humidity']]
rainfall_features_scaled_updated = scaler.transform(rainfall_features_updated)

# Membuat urutan untuk prediksi LSTM
X_pred, _ = create_sequences(rainfall_features_scaled_updated, seq_length)

# Melakukan prediksi menggunakan model LSTM
predictions = model.predict(X_pred)
predictions = scaler.inverse_transform(predictions)

# Menambahkan prediksi ke ntb_rainfall_data
ntb_rainfall_data['predicted_rainfall'] = np.nan
ntb_rainfall_data.iloc[seq_length:seq_length+len(predictions), ntb_rainfall_data.columns.get_loc('predicted_rainfall')] = predictions.flatten()

# Memfilter titik-titik dengan curah hujan yang diprediksi rendah
ntb_rainfall_data_filtered = ntb_rainfall_data[ntb_rainfall_data['predicted_rainfall'] >= 5]

# Mencetak prediksi curah hujan di area rawan banjir di NTB
print(ntb_rainfall_data_filtered[['latitude', 'longitude', 'predicted_rainfall']])

# Memvisualisasikan area rawan banjir dan prediksi curah hujan di NTB
fig, ax = plt.subplots(figsize=(12, 8))
flood_areas.plot(ax=ax)
ntb_rainfall_data.plot(column='predicted_rainfall', cmap='Blues', markersize=10, legend=True, ax=ax)

plt.show()
