# Market Simulator

Детерминированный сервис воспроизведения исторического L2 и OHLCV с виртуальным
шагом 100 мс. Simulator отделён от `ML_service`: он отвечает только за время,
порядок событий и HTTP-транспорт, а модели, признаки и позиции остаются внутри
ML-контура.

## Зачем нужен отдельный сервис

- Повторяемый прогон одного и того же участка рынка.
- Проверка causal-контракта без доступа к будущим свечам.
- Сравнение стратегий на идентичном потоке.
- Управляемая скорость replay без изменения временных меток.
- Тестирование сетевого API так же, как при live WebSocket feed.

## Подготовка replay

```bash
python -m market_simulator.prepare \
  --l2 ../ML_service/market_replay_data/raw_last_3d/l2_raw.parquet \
  --ohlcv ../ML_service/market_replay_data/raw_last_3d/ohlcv_raw.parquet \
  --output ../ML_service/simulator_state \
  --max-pairs 8 \
  --min-concurrent-seconds 900 \
  --overwrite
```

Подготовщик проверяет не только пересечение `min_ts/max_ts`, а настоящий
одновременный поток обеих ног. Для каждой выбранной пары регистрируются оба
направления сделки.

```text
simulator_state/
├── l2_selected_sorted.parquet
├── ohlcv_selected_sorted.parquet
├── pairs.json
└── manifest.json
```

OHLCV публикуется только после закрытия свечи.

## Запуск

Самостоятельно:

```bash
uvicorn market_simulator.main:app --host 0.0.0.0 --port 8090
```

В составе проекта:

```bash
cd ../ML_service
docker compose --profile replay up -d --build
```

Запуск 15-минутного replay:

```bash
curl -X POST http://localhost:8090/v1/replay/start \
  -H 'content-type: application/json' \
  -d '{"speed": 1, "duration_seconds": 900}'
```

Для smoke-теста можно использовать `speed: 10`. Измерять реальную
производительность моделей следует при `speed: 1`.

## API

```text
GET  /health/live
GET  /v1/replay/status
GET  /v1/replay/pairs
POST /v1/replay/start
POST /v1/replay/pause
POST /v1/replay/resume
POST /v1/replay/stop
```

Swagger UI: [http://localhost:8090/docs](http://localhost:8090/docs).

## Переменные окружения

| Переменная | Значение по умолчанию |
|---|---|
| `SIM_DATA_DIR` | `../ML_service/market_replay_data/raw_last_3d` |
| `SIM_PREPARED_DIR` | `state/prepared` |
| `SIM_ML_URL` | `http://127.0.0.1:8080` |
| `SIM_DEFAULT_SPEED` | `1` |
| `SIM_DEFAULT_DURATION_SECONDS` | `900` |
| `SIM_OHLCV_WARMUP_MINUTES` | `720` |

## Тесты

```bash
python -m pytest -q
```

Тесты проверяют выбор непрерывного интервала, оба направления пары, causal
OHLCV warm-up, паузу/возобновление и отправку 100-мс batch-событий.
