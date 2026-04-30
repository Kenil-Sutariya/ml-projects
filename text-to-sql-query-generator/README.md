# AI-Powered SQL Query Generator

Convert natural language questions into safe SQL queries, run them against a sample sales database, and explore uploaded CSV schemas in a clean Streamlit interface.

This project is designed as a practical beginner-friendly AI Engineering and Data Engineering portfolio project. It demonstrates prompt engineering, database handling, SQL safety validation, lightweight evaluation, and simple app deployment.

## Highlights

- Natural language to SQLite `SELECT` query generation
- OpenAI, Ollama, and no-key demo rule modes
- Built-in SQLite sales database with customers, products, and orders
- Safety guardrail that blocks unsafe SQL such as `DROP`, `DELETE`, `UPDATE`, and `INSERT`
- Read-only database execution for defense in depth
- Multi-CSV upload panel for schema discovery
- Streamlit interface for an easy demo
- Pytest test suite covering database, safety, and upload logic

## Demo Flow

1. Ask a business question, such as `Show total sales by month`.
2. The app sends the database schema and question to the generator.
3. The generated SQL is displayed for transparency.
4. A safety layer validates that the SQL is a single `SELECT` query.
5. The query runs against SQLite using a read-only connection.
6. Results are shown in an interactive table.

Unsafe requests are blocked before reaching the database.

## Tech Stack

| Layer | Tool |
| --- | --- |
| App | Streamlit |
| Language | Python |
| Database | SQLite |
| Data handling | Pandas |
| LLM providers | OpenAI, Ollama, demo rules |
| Testing | Pytest |
| Safety | SQL validation plus read-only SQLite execution |

## Project Structure

```text
ai-sql-query-generator/
├── app.py
├── data/
│   └── sales.db              # generated locally
├── examples/
│   ├── example_questions.md
│   └── sample_upload.csv
├── scripts/
│   └── create_database.py
├── src/
│   └── sql_generator/
│       ├── database.py
│       ├── datasets.py
│       ├── llm.py
│       └── safety.py
├── tests/
│   ├── test_database.py
│   ├── test_datasets.py
│   └── test_safety.py
├── requirements.txt
└── README.md
```

## Quick Start

```bash
cd ai-sql-query-generator
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/create_database.py
streamlit run app.py
```

Open the local Streamlit URL shown in your terminal.

The default `Demo rules` provider works immediately without an API key.

## Example Questions

Try these prompts in the app:

- Show total sales by month
- Which products have the highest sales?
- Show total revenue by customer
- Show sales by product category
- Show the latest orders
- Drop the customers table

The last example is intentionally unsafe and should be blocked by the safety layer.

## Dataset Upload

The upload panel supports adding multiple CSV files at once. For each uploaded file, the app shows:

- File name
- Row count
- Column count
- Column names
- Inferred data types

You can test this feature with:

```text
examples/sample_upload.csv
```

## OpenAI Mode

Set your API key before starting Streamlit:

```bash
export OPENAI_API_KEY="your_api_key_here"
streamlit run app.py
```

Then select `OpenAI` in the sidebar.

## Ollama Mode

Install and start Ollama locally, then pull a model:

```bash
ollama pull llama3.1
ollama serve
streamlit run app.py
```

Then select `Ollama` in the sidebar.

## Safety Design

The app uses multiple protections before SQL is executed:

- Removes code fences and normalizes generated SQL
- Allows only one SQL statement
- Requires the statement to start with `SELECT`
- Blocks dangerous keywords such as `DROP`, `DELETE`, `UPDATE`, `INSERT`, `CREATE`, `ALTER`, and `PRAGMA`
- Runs approved SQL through a read-only SQLite connection
- Uses SQLite authorizer callbacks to deny non-read operations

Allowed example:

```sql
SELECT product_name, category FROM products;
```

Blocked examples:

```sql
DROP TABLE customers;
DELETE FROM orders;
UPDATE products SET unit_price = 0;
INSERT INTO customers VALUES (99, 'Test', 'Nowhere', '2025-01-01');
```

## Run Tests

```bash
pytest
```

Current coverage focuses on:

- Sample database creation and query execution
- SQL safety validation
- Read-only database protection
- CSV upload parsing and schema profiling

## What This Project Shows

This project is useful for AI Engineer and Data Engineer portfolios because it combines:

- LLM application design
- Prompt construction with schema context
- Structured data handling
- Database querying
- Safety validation for generated code
- Simple UI deployment
- Testable Python modules

## Future Improvements

- Convert uploaded CSV files into temporary SQLite tables
- Let users ask questions across uploaded datasets
- Add query explanation in plain English
- Add evaluation cases comparing expected SQL to generated SQL
- Add optional FastAPI endpoint for programmatic use
- Add Docker support for deployment

## Interview Talking Points

- The LLM receives schema context, not raw database rows.
- Generated SQL is treated as untrusted output and validated before execution.
- The app uses both static validation and read-only execution for safety.
- Demo mode makes the project easy to run without external services.
- The code is split into small modules so database, LLM, dataset, and safety logic can be tested independently.
