SHELL=/bin/bash
LINT_PATHS=src/physilearning tests/ docs/source/conf.py setup.py

pytest:
	./scripts/run_tests.sh

lint:
	ruff ${LINT_PATHS} --select=E9,F63,F7,F82 --show-source
	ruff ${LINT_PATHS} --exit-zero

doc:
	cd docs && make html

raven:
	cd ./src/PhysiCell_src && make raven && make

mela:
	cd ./src/PhysiCell_src && make mela && make
