default: help

dev: ## Start development server with auto-reload at 0.0.0.0:8000
	@echo "Starting development server..."
	 uvicorn src.api.main:app --host 0.0.0.0 --port 8000  --reload

start: ## Start production server at 0.0.0.0:8000
	@echo "Starting application..."
	 uvicorn src.api.main:app --host 0.0.0.0 --port 8000 

help: ## Display this help screen
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m\033[0m\n"} /^[$$()% a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)