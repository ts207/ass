run:
	python3 -m app.chat

lint:
	python3 -m py_compile app/*.py

db-tables:
	sqlite3 data/assistant.sqlite3 ".tables"

