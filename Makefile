.PHONY: demo run review clean

demo:
	python scripts/make_demo_data.py

run:
	python src/ingest.py && \
	python src/embed.py && \
	python src/index.py --seed 1001 && \
	python src/score.py

review:
	streamlit run app/review_ui.py

clean:
	rm -rf data/working data/output
