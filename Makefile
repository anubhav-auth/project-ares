# Makefile
.PHONY: help up down restart logs clean build

help:
	@echo "Available commands:"
	@echo "  make up        - Start all services"
	@echo "  make down      - Stop all services"
	@echo "  make restart   - Restart all services"
	@echo "  make logs      - View logs"
	@echo "  make clean     - Clean up volumes"
	@echo "  make build     - Build all services"

up:
	docker-compose --env-file .env up -d

down:
	docker-compose down

restart:
	docker-compose down
	docker-compose --env-file .env up -d

logs:
	docker-compose logs -f

clean:
	docker-compose down -v
	rm -rf postgres_data redis_data n8n_data

build:
	docker-compose --env-file .env build --no-cache

# Service-specific commands
analytics-logs:
	docker-compose logs -f analytics-service

ai-logs:
	docker-compose logs -f ai-service

automation-logs:
	docker-compose logs -f automation-service

db-shell:
	docker exec -it ares-postgres psql -U ${POSTGRES_USER} -d ${POSTGRES_DB}

redis-cli:
	docker exec -it ares-redis redis-cli