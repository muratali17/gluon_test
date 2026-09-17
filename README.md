# gluon_test

CSV dosyalarından AutoGluon ile model eğitmek, kaydedilmiş modelleri yüklemek
ve tahmin üretmek için kullanılan Streamlit uygulaması.

## Gereksinimler

- Python 3.12.x
- `pip`
- AutoGluon 1.6.1

Python 3.14 şu an desteklenmez. Bağımlılıklar proje kökündeki
`requirements.txt` dosyasından kurulur ve sistem Python kurulumunu değiştirmemek
için sanal ortam kullanılması gerekir.

## Kurulum

Proje kök dizininde aşağıdaki komutları çalıştırın:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Ubuntu/Debian sisteminde `venv` oluşturma sırasında `ensurepip is not
available` hatası alırsanız önce şu paketi kurun:

```bash
sudo apt update
sudo apt install -y python3.12-venv
```

Ardından kurulum komutlarını tekrar çalıştırın.

## Uygulamayı Çalıştırma

Sanal ortam aktifken:

```bash
streamlit run app/web_app/app.py --server.address 0.0.0.0 --server.port 8501
```

Tarayıcıdan [http://localhost:8501](http://localhost:8501) adresini açın.

Uygulamada:

1. **Upload Data** sekmesinden bir CSV dosyası yükleyin.
2. **Train** sekmesinde hedef sütunu, görev adını ve süre sınırını seçin.
3. Model eğitildikten sonra **Predict** sekmesinden kayıtlı modeli yükleyip tekli
	veya toplu tahmin üretin.

Modeller proje kökündeki `trained_models/` dizinine kaydedilir. Bu dizin Git
tarafından takip edilmez.

## Sanal Ortamı Kapatma

```bash
deactivate
```