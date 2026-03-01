.PHONY: format lint test

format:
	docker run --rm \
		--pull=always \
		-v .:/app \
		hmcvlab/format:latest

lint:
	docker run --rm \
		--pull=always \
		-v .:/app \
		hmcvlab/lint:latest

test:
	docker run --rm  \
		--user ubuntu \
		-w /app \
		-v .:/app \
		-t hmcvlab/computer-vision:3.2.7 \
		bash -c "pip install -e . && pytest"

install-hooks:
	@echo "make format && make lint" > .git/hooks/pre-commit
	@echo "make test" > .git/hooks/pre-push
	chmod +x .git/hooks/pre-*
