flist = 0

all: $(patsubst %, output/figure%.svg, $(flist))

# Figure rules
output/figure%.svg: venv genFigure.py msresist/figures/figure%.py
	. venv/bin/activate && ./genFigure.py $*

venv: venv/bin/activate

venv/bin/activate: requirements.txt msresist/data/RNAseq/AXLmutants_RNAseq_merged.feather
	test -d venv || virtualenv venv
	. venv/bin/activate && pip install --prefer-binary -Uqr requirements.txt
	touch venv/bin/activate

test: venv
	. venv/bin/activate && pytest -s -v -x msresist

testcover: venv
	. venv/bin/activate && pytest --junitxml=junit.xml --cov=msresist --cov-report xml:coverage.xml

msresist/data/RNAseq/AXLmutants_RNAseq_merged.feather: msresist/data/RNAseq/AXLmutants_RNAseq_merged.feather.xz
	xz -vk -d $<

%.pdf: %.ipynb
	. venv/bin/activate && jupyter nbconvert --execute --ExecutePreprocessor.timeout=6000 --to pdf $< --output $@

clean:
	rm -rf *.pdf venv pylint.log
