COMPOSE = docker compose
CUDA_COMPOSE = docker compose -f compose.yaml -f compose.cuda.yaml

.PHONY: help up up-gpu replay replay-gpu down logs config test

help:
	@printf '%s\n' \
		'make up          # ML API и dashboard на CPU/auto' \
		'make up-gpu      # ML API и dashboard с CUDA' \
		'make replay      # добавить исторический market simulator' \
		'make replay-gpu  # replay и CUDA' \
		'make logs        # следить за логами сервисов' \
		'make down        # остановить контейнеры' \
		'make test        # запустить unit-тесты'

up:
	$(COMPOSE) up -d --build ml-service dashboard-service

up-gpu:
	$(CUDA_COMPOSE) up -d --build ml-service dashboard-service

replay:
	$(COMPOSE) --profile replay up -d --build

replay-gpu:
	$(CUDA_COMPOSE) --profile replay up -d --build

down:
	$(COMPOSE) --profile replay down

logs:
	$(COMPOSE) --profile replay logs -f --tail=200

config:
	$(CUDA_COMPOSE) --profile replay config

test:
	python -m pytest -q
