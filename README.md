# SoftGNSS — GPS L1 C/A Software Receiver

Программный GNSS-приёмник для обработки GPS L1 C/A сигналов из сырых IQ-отсчётов. Поддерживает как комплексные baseband-данные (IF=0, int16), так и реальные IF-сигналы (int8) — классический режим SoftGNSS.

---

## Структура репозитория

```
SoftGNSS/
├── main.py                 # Оригинальный пайплайн (Python 3)
├── initialize.py           # Settings, C/A-коды, чтение файлов
├── acquisition.py          # Обнаружение спутников (PCPS)
├── tracking.py             # Слежение (DLL + PLL)
├── ephemeris.py            # Декодирование навигационного сообщения
├── postNavigation.py       # Навигационное решение (псевдодальности, позиция)
├── geoFunctions/
│   └── __init__.py         # Геодезические функции (WGS-84, UTM, тропосфера)
│
├── gnss_receiver/          # Новый модульный приёмник (CLI + JSON + CSV + графики)
│   ├── config.py           # Настройки, CLI (argparse), парсинг Zenodo JSON
│   ├── data_loader.py      # Загрузка IQ из .bin (int8/int16/float32)
│   ├── prn_generator.py    # Генерация C/A-кодов PRN 0–31
│   ├── detection.py        # Адаптивный порог: SNR = (peak − μ) / σ
│   ├── acquisition.py      # Обнаружение (PCPS, когерентное/некогерентное)
│   ├── tracking.py         # Слежение (DLL + PLL)
│   ├── nav_decoder.py      # Синхронизация LNAV и декодирование эфемерид
│   ├── pvt.py              # Псевдодальности, спутниковые орбиты и расчёт координат
│   ├── output.py           # Таблицы, JSON, CSV, matplotlib-графики
│   ├── main.py             # Оркестрация: acquisition → tracking
│   └── __main__.py         # Точка входа: python -m gnss_receiver
│
├── Clearsky_signal_C_-7_crop.bin   # Тестовый IQ-файл (24 МБ, int16, 50 МГц)
├── Info_about_dataset.json         # Метаданные Zenodo
├── Readme.pdf                      # Параметры записи (USRP-2945R, GPS L1)
└── Technical_specification.txt     # Техническое задание
```

---

## Быстрый старт

### Установка

```bash
pip install numpy scipy matplotlib
```

### Обнаружение спутников (gnss_receiver)

Одна строка (bash, PowerShell, cmd):

```bash
python -m gnss_receiver --data Clearsky_signal_C_-7_crop.bin --json Info_about_dataset.json --data-format int16 --threshold 2.5 --plot --output results.json
```

С переносом строк (bash / macOS):

```bash
python -m gnss_receiver \
  --data Clearsky_signal_C_-7_crop.bin \
  --json Info_about_dataset.json \
  --data-format int16 \
  --threshold 2.5 \
  --plot \
  --output results.json
```

С переносом строк (Windows PowerShell):

```powershell
python -m gnss_receiver `
  --data Clearsky_signal_C_-7_crop.bin `
  --json Info_about_dataset.json `
  --data-format int16 `
  --threshold 2.5 `
  --plot `
  --output results.json
```

### Полный пайплайн с навигационным решением (координаты)

```bash
python -m gnss_receiver \
  --data path/to/record.bin \
  --json Info_about_dataset.json \
  --data-format float32 \
  --mode full \
  --ms-to-process 37000 \
  --nav-sol-period-ms 500 \
  --output position.json
```

Режим `full` выполняет acquisition, tracking, декодирование GPS LNAV-эфемерид и расчёт PVT. В JSON будут координаты WGS-84 (широта, долгота, высота и ECEF), DOP и список спутников для каждой эпохи. Нужны минимум 4 спутника с корректными эфемеридами и обычно не менее 36 секунд непрерывной записи; полный файл датасета — interleaved `float32`, приложенный crop — `int16` и слишком короткий для навигационного решения.

Для короткого crop можно запустить обнаружение спутников, но в `full` координаты будут помечены как недоступные, если данных недостаточно.

### Оригинальный пайплайн (Settings → acquire → track)

```python
from initialize import Settings
from acquisition import AcquisitionResult
import tracking

s = Settings()  # настроен на int16, 50 МГц, IF=0
s.fileName = 'Clearsky_signal_C_-7_crop.bin'

with open(s.fileName, 'rb') as fid:
    signal = s.readComplexSignal(fid, 11 * s.samplesPerCode)

acq = AcquisitionResult(s)
acq.acquire(signal)
acq.preRun()
acq.showChannelStatus()

s.msToProcess = 37
tr = tracking.TrackingResult(acq)
with open(s.fileName, 'rb') as fid:
    tr.track(fid)
```

---

## Два режима работы

### 1. Комплексный baseband (isComplexIQ=True)

Данные от SDR-приёмника в формате interleaved IQ с IF=0. Типичный формат — int16 little-endian.

```
Файл:  I₀ Q₀ I₁ Q₁ I₂ Q₂ ...  (каждый отсчёт — 2 байта int16)
```

Алгоритм обнаружения использует комплексное умножение на пробную несущую:

```
baseband = signal × exp(−j·2π·fd·t)
corr = |IFFT(FFT(baseband) × conj(FFT(C/A_code)))|²
```

Настройки по умолчанию:

| Параметр            | Значение        |
|---------------------|-----------------|
| `dataType`          | `int16` (LE)    |
| `samplingFreq`      | 50 МГц          |
| `IF`                | 0 Гц            |
| `isComplexIQ`       | `True`          |

### 2. Реальный IF-сигнал (isComplexIQ=False)

Данные от антенны с промежуточной частотой. Классический режим SoftGNSS v3.0.

```
Файл:  s₀ s₁ s₂ ...  (каждый отсчёт — 1 байт int8, вещественный сигнал)
```

| Параметр            | Значение            |
|---------------------|---------------------|
| `dataType`          | `int8`              |
| `samplingFreq`      | 38.192 МГц          |
| `IF`                | 9.548 МГц           |
| `isComplexIQ`       | `False`             |

---

## Параметры CLI (gnss_receiver)

```
python -m gnss_receiver --data <файл> [опции]
```

| Параметр           | По умолчанию | Описание                                      |
|--------------------|--------------|-----------------------------------------------|
| `--data`           | —            | Путь к .bin файлу (обязательно)               |
| `--json`           | —            | Путь к Zenodo JSON (извлекает fs)             |
| `--mode`           | `acquisition`| `acquisition` или `full` (acq + tracking + PVT) |
| `--fs`             | 50000000     | Частота дискретизации, Гц                     |
| `--if-freq`        | 0            | Промежуточная частота, Гц                     |
| `--data-format`    | авто         | `int8`, `int16`, `float32`                    |
| `--doppler-range`  | 10000        | Диапазон Доплера, ±Гц                         |
| `--doppler-step`   | 500          | Шаг поиска, Гц                                |
| `--threshold`      | 2.5          | Порог SNR                                     |
| `--coherent-ms`    | 1            | Когерентное накопление, мс                    |
| `--noncoherent`    | 1            | Некогерентные суммирования                    |
| `--prn`            | 1-32         | Список PRN через запятую                      |
| `--offset`         | 0            | Начальный комплексный отсчёт                  |
| `--ms-to-process`  | 37000        | Максимальная длительность слежения, мс         |
| `--nav-sol-period-ms` | 500        | Интервал между решениями, мс                  |
| `--channels`       | 8            | Максимум спутниковых каналов в режиме full    |
| `--output`         | авто         | Путь к выходному JSON                        |
| `--save-correlation| —            | Сохранить корреляционные матрицы (.npy)       |
| `--plot`           | —            | Сохранить графики (PNG)                       |
| `--log-level`      | `INFO`       | `ERROR`, `WARNING`, `INFO`, `DEBUG`           |

---

## Алгоритм обнаружения (Acquisition)

**Метод:** Parallel Code Phase Search (PCPS) с FFT-корреляцией.

Для каждого спутника (PRN 0–31):
1. Для каждой пробной частоты Доплера (шаг 500 Гц, ±10 кГц):
   - Умножить сигнал на пробную несущую
   - Вычислить корреляцию с C/A-кодом через FFT
2. Выбрать лучший из двух последовательных 1-мс блоков (защита от границы бита навигационного сообщения)
3. Вычислить SNR: `(peak − noise_mean) / noise_std`
4. Если SNR > порога → спутник обнаружен

| Порог        | Режим            | Примечание                         |
|-------------|------------------|------------------------------------|
| 2.0 – 2.5   | Чувствительный   | Возможны ложные срабатывания       |
| 2.5 – 3.5   | Стандартный      | Рекомендуется                      |
| 3.5 – 5.0   | Консервативный   | Только уверенные обнаружения       |

---

## Алгоритм слежения (Tracking)

Для каждого обнаруженного спутника:
- **PLL** (Phase Lock Loop): Costas-дискриминатор, устойчивый к 180° инверсии навигационных битов
- **DLL** (Delay Lock Loop): отслеживание кода, нормированный Early-Minus-Late

---

## Примеры вывода

### Таблица обнаружения (gnss_receiver)

```
====================================================================
ACQUISITION RESULTS
====================================================================
 PRN Doppler (Hz)  Code Phase   Code Phase     SNR  Flag  Peak Ratio
                    (samples)      (chips)                 (1st/2nd)
--------------------------------------------------------------------
  16          0.0        7135       145.98   71.74    ok        6.46
  18          0.0       14249       291.53   48.98    ok        4.28
  26      -2000.0       28666       586.51   24.36    ok        2.37
  29      -3500.0       21306       435.92   23.29    ok        2.62
  27       3500.0       48918      1000.86   20.93    ok        2.39
--------------------------------------------------------------------
Total detected satellites: 32
  Strong detections (ok): 16
  Weak detections:        16
```

### CSV-вывод (results.csv)

```csv
PRN;Doppler (Hz);Code Phase (samples);Code Phase (chips);SNR;Flag;Peak Ratio (1st/2nd)
16;0,0;7135;145,98;71,74;ok;6,46
18;0,0;14249;291,53;48,98;ok;4,28
```

Формат: разделитель `;`, десятичный разделитель `,` (EU-формат для корректного открытия в Excel).

---

## Результаты тестирования

Датасет: `Clearsky_signal_C_-7_crop.bin` (C-7 Clear-Sky, int16, 50 МГц, IF=0)

| Пайплайн              | Обнаружено | Tracking         | Время   |
|-----------------------|-----------|------------------|---------|
| Оригинальный pipeline | 3 спутн.  | 3 канала         | ~7 с    |
| `gnss_receiver`       | 32 спутн. | 8 каналов        | ~23 с   |

---

## Источник данных

Набор: **Tuni2025 Datasets for GNSS — GPS Spoofing**
DOI: [10.5281/zenodo.17249795](https://doi.org/10.5281/zenodo.17249795)
Лицензия: CC BY 4.0
Сценарий: C-7 GPS Static, True Position, Clear-Sky (no spoofers)

Параметры записи:
- Приёмник: USRP-2945R
- Сигнал: GPS L1 C/A (1575.42 МГц)
- Частота дискретизации: 50 МГц
- Формат: interleaved IQ, 16-bit int (crop)
- Усиление: 20 дБ

---

## Требования

- Python 3.10+
- numpy ≥ 1.24
- scipy ≥ 1.10 (для gnss_receiver)
- matplotlib ≥ 3.7 (для графиков)

---

## Лицензия

Основан на SoftGNSS v3.0 (Darius Plausinaitis, Dennis M. Akos).
GNU General Public License v2.
