.PHONY: install test clean uninstall

install:
	@echo "🔧 Creating virtual environment and installing the package..."
<<<<<<< HEAD
	pip install --upgrade pip setuptools wheel && pip install .
=======
	pip install --upgrade pip && pip install -e .
>>>>>>> 68a9c762ca3c35b729043e9c5dafe47f810b3719

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
