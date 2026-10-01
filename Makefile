.PHONY: format lint test check reproduce

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

check:
	python scripts/check_setup.py

reproduce:
	python scripts/analysis/fog.py
	python scripts/analysis/gt_2d.py
	python scripts/analysis/gt_3d.py
	python scripts/analysis/metadata_2d.py
	python scripts/analysis/metadata_3d.py
	python scripts/analysis/eval_2d.py
	python scripts/analysis/eval_3d.py
	python scripts/figures/reprojection_error.py
	python scripts/figures/fog.py
	python scripts/figures/weather_impact.py
	python scripts/figures/samples.py
	python scripts/table/correlation_metrics.py
