.PHONY: install test clean uninstall

install:
	@echo "🔧 Creating virtual environment and installing the package..."
	pip install --upgrade pip setuptools wheel && pip install .

test:
	@echo "🧪 Running tests..."
	pytest

clean:
	@echo "🧹 Cleaning up..."
	conda remove --all -n ALGen
	find . -type d -name "__pycache__" -exec rm -r {} +

uninstall:
	@echo "❌ Uninstalling package..."
	conda activate ALGen && pip uninstall -y ALGen
