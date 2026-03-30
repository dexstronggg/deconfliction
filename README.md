# Speed-Only Deconfliction Benchmark

Бенчмарк классических и нейросетевых алгоритмов деконфликтации (разрешения конфликтов) для беспилотных летательных аппаратов.

**Управление только скоростью** — дрон не может менять направление, только ускоряться/замедляться вдоль фиксированного маршрута.

## Алгоритмы

| Алгоритм | Тип | Сложность |
|---|---|---|
| No-Control | Baseline | O(1) |
| FCFS | Классический | O(N log N) |
| VO (Velocity Obstacle) | Классический | O(N^2) |
| VO-SSD (Speed Separation Detection) | Классический | O(N log N) |
| RVO (Reciprocal VO) | Классический | O(N^2) |
| ORCA (Optimal Reciprocal Collision Avoidance) | Классический | O(N log N) |
| CSORCA (Cooperative Speed ORCA) | Классический | O(N^2) |
| MILP (Mixed-Integer Linear Programming) | Оптимизационный | O(N^2 + LP) |
| MAPPO (Multi-Agent PPO) | MARL | O(N\*K\*d) |
| MADDPG (Multi-Agent DDPG) | MARL | O(N\*K\*d^2) |
| MASAC (Multi-Agent SAC) | MARL | O(N\*K\*d^2) |

MARL-алгоритмы представлены в виде необученных нейросетей (random weights) — базовый уровень для последующего обучения.

## Требования

- Python 3.10+
- Windows / Linux / macOS

## Установка

```bash
git clone https://github.com/dexstronggg/deconfliction.git
cd deconfliction
pip install -r requirements.txt
```

## Запуск

Полный бенчмарк (1000 агентов, все алгоритмы):

```bash
python main.py
```

С выводом прогресса:

```bash
python main.py --verbose
```

Быстрый тест (50 агентов):

```bash
python main.py --agents 50 -v
```

Без ветра:

```bash
python main.py --agents 1000 --no-wind -v
```

Другой seed:

```bash
python main.py --seed 123
```

## Параметры симуляции

| Параметр | Значение |
|---|---|
| Размер мира | 10 000 x 10 000 м |
| Количество агентов | 1000 |
| Шаг симуляции | 0.1 с |
| Макс. время | 600 с (10 мин) |
| Номинальная скорость | 15 м/с |
| Диапазон скоростей | 0.5 — 20 м/с |
| Радиус агента | 1.5 м |
| Дистанция коллизии | 3 м |
| Ветер (база / порывы) | 2 / 4 м/с |

Параметры настраиваются в `config.py`.

## Результаты

После запуска в папке `results/` появятся:

- `benchmark_YYYYMMDD_HHMMSS.json` — результаты в JSON
- `benchmark_YYYYMMDD_HHMMSS.xlsx` — Excel-таблица с тремя листами:
  1. **Benchmark Results** — метрики по каждому алгоритму
  2. **Параметры симуляции** — условия эксперимента
  3. **Описание алгоритмов** — принцип работы каждого алгоритма

### Метрики

- **Всего коллизий** — количество уникальных пар агентов, сблизившихся менее чем на 3 м
- **Частота коллизий** — коллизии / кол-во агентов
- **Успешность** — доля агентов, долетевших до цели за отведённое время
- **Makespan** — время прибытия последнего агента
- **Утилизация скорости** — средняя (фактическая скорость / номинальная)
- **Время алгоритма** — чистое время на вызовы `compute_speeds()`

## Структура проекта

```
deconfliction/
├── main.py              — точка входа
├── config.py            — параметры симуляции
├── environment.py       — 2D-среда (агенты, ветер, коллизии через KDTree)
├── benchmark.py         — запуск алгоритмов, сбор метрик, Excel-экспорт
├── requirements.txt
├── algorithms/
│   ├── base.py          — интерфейс BaseAlgorithm
│   ├── no_control.py    — без управления (baseline)
│   ├── fcfs.py          — FCFS приоритетное торможение
│   ├── vo_classic.py    — Velocity Obstacle
│   ├── vo_speed_only.py — VO-SSD
│   ├── rvo.py           — Reciprocal VO
│   ├── orca_projected.py — ORCA с проекцией на heading
│   ├── csorca.py        — Cooperative Speed ORCA
│   ├── milp.py          — линейное программирование
│   ├── mappo.py         — MAPPO (необученный)
│   ├── maddpg.py        — MADDPG (необученный)
│   └── masac.py         — MASAC (необученный)
└── results/             — JSON + XLSX результаты
```
