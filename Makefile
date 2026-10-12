.PHONY: setup smoke corpus run analyze test reproduce

setup:      ## install dependencies
	uv sync
smoke:      ## end-to-end check on a few clips
	uv run python scripts/smoke_test.py
corpus:     ## human + TTS corpus, then rate-matched voices
	uv run python scripts/01_make_corpus.py
	uv run python scripts/01b_rate_matched.py
run:        ## full grid (cached, resumable)
	uv run python scripts/04_run_experiment.py
analyze:    ## tables, figures, summary.json
	uv run python scripts/05_analyze.py
test:       ## statistics unit tests
	uv run --with pytest python -m pytest tests
reproduce:  ## the exact published run
	./scripts/reproduce.sh
