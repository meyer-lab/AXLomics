# flist = 1 2

all: .venv

.venv: pyproject.toml uv.lock
	uv sync
	touch .venv

test: .venv
	uv run pytest -s -v -x msresist

testcover: .venv
	uv run pytest --junitxml=junit.xml --cov=msresist --cov-report xml:coverage.xml

msresist/data/RNAseq/AXLmutants_RNAseq_merged.feather: msresist/data/RNAseq/AXLmutants_RNAseq_merged.feather.xz
	xz -vk -d $<

%.pdf: %.ipynb .venv
	uv run jupyter nbconvert --execute --ExecutePreprocessor.timeout=6000 --to pdf $< --output $@

lint: .venv
	uv run ruff check .

notebooks: .venv msresist/data/RNAseq/AXLmutants_RNAseq_merged.feather
	uv run jupyter nbconvert --execute --inplace *.ipynb

clean:
	rm -rf *.pdf .venv pylint.log
