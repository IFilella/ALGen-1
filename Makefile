.PHONY: install test clean uninstall

install:
	@echo "🔧 Creating virtual environment and installing the package..."
	conda create -n ALGen
	. conda activate ALGen && pip install --upgrade pip && pip install -e .

test:
	@echo "🧪 Running tests..."
	. conda activate ALGen && pip install -r requirements.txt && pytest

clean:
	@echo "🧹 Cleaning up..."
	conda remove --all -n ALGen
	find . -type d -name "__pycache__" -exec rm -r {} +

uninstall:
	@echo "❌ Uninstalling package..."
	. conda activate ALGen && pip uninstall -y ALGen