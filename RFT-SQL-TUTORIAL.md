# ✈️ Natural-Language → SQL with Reinforcement-Fine-Tuning (RFT)

Welcome! This tutorial will show you how to fine-tune a 7B parameter model to answer natural language (NL) questions by writing SQL to execute against your database, without using real production data in fine-tuning the process.

### Why this matters  
Off-the-shelf LLM copilots often guess column names, ignore schema quirks, or hallucinate tables. **Reinforcement Fine-Tuning (RFT)** fixes this by teaching the model the shape of your data _and_ the patterns in your queries, boosting exact-match accuracy.

---

## In this tutorial you will

| You'll practice … | … and walk away with |
| --- | --- |
| ✅ **Generate a synthetic DuckDB** that mirrors your schema | `synthetic_openflights.db` (<20 MB) served via an MCP endpoint |
| ✅ **Create a MECE query set** & compute ground-truth rows | `generated_queries.json` & `ground_truth_results.json` |
| ✅ **Build NL ↔ SQL result pairs** for fine-tuning and eval | `final_rft_sql_train_data.jsonl` & `final_rft_sql_test_data.jsonl` |
| ✅ **Run an RFT job on Fireworks AI** | A tuned **Qwen 2.5-7B** checkpoint |
| ✅ **Benchmark baseline vs. tuned model** and a larger baseline | > 15% exact-match improvement over baseline and on-par with SoTA base models |


## Agenda

1. 🛠️ Development Environment Setup
2. 🗄️ Simulate the "Production" Database  
3. 📋 Acquire the Schema (No Real Data!)
4. 🧪 Create the Synthetic Training Sandbox with an LLM
5. ✅ Validate the Sandbox
6. 📝 Generate Example SQL Queries
7. 🎯 Execute Queries to Get Ground-Truth Answers
8. 💬 Generate Natural Language Questions for Final RFT Training Data
9. 🛰️ Deploy an MCP Server for the Synthetic Data
10. ☁️ Set Up Google Cloud CLI & .gcloudignore
11. 📦 Containerize & Deploy the MCP Server
12. 🔍 Define an evaluation function for RFT
13. 🧪 Test English -> SQL of a base model without fine-tuning
14. 🚀 Launch the Fine-Tuning Job & Deploy via the UI
15. ⚖️ Evaluate Model Performance
16. ✨ Cleanup & Conclusion


> **Demo vs Real World 🌍**  
> Look for these call-outs to see the difference between the self-contained demo steps in this notebook and the equivalent actions you'd perform on your own private schema, logs, and query store.

---

## 1. 🛠️ Development Environment Setup

**Complete these steps once in your terminal, *outside* this notebook.**

1.  **Get a Fireworks AI API Key**
    - Go to [fireworks.ai](https://fireworks.ai) and sign up.
    - Create an API key from your settings page.
    - Create a file named `.env` in your project directory and add your key:
      ```
      FIREWORKS_API_KEY="YOUR_API_KEY_HERE"
      ```

2.  **Install `uv`**
    - `uv` is a fast Python package manager from Astral. Follow the official installation instructions at [docs.astral.sh/uv/](https://docs.astral.sh/uv/).
    - It's significantly faster than pip and handles dependency resolution more reliably.

3.  **Create a Virtual Environment and Install Packages**
    - Once `uv` is installed, initialize a project.
    ```bash
    # Run this in your terminal
    uv init --python 3.12
    ```
    - Install all required packages using `uv add`.
    ```bash
    # Run this in your terminal
    uv add duckdb tabulate pandas pyarrow requests \
           pydantic python-dotenv \
           jsonlines fireworks-ai \
           mcp-server-motherduck
    ```
    - Create and activate a virtual environment
    ```bash
    # Run this in your terminal
    uv sync
    source .venv/bin/activate
    ```

After running these commands, your environment is ready. You can proceed with the cells inside this notebook.

---

## 2. 🗄️ Simulate the "Production" Database

First, we'll create a database that represents your real, populated production database. We'll download the public OpenFlights dataset and load it into a DuckDB file.

### What is DuckDB?
DuckDB is an in-process SQL OLAP database management system. Think of it as "SQLite for analytics". It's perfect for this tutorial because:
- It's embedded (no server setup required)
- It's fast for analytical queries
- It has excellent SQL compatibility
- The entire database is just a single file
- It has an existing MCP server we can use ([mcp-server-motherduck](https://github.com/motherduckdb/mcp-server-motherduck))

> **Real World 🌍**: You already have this! It's your live production database (or a replica). You would skip this entire step.

```python
import urllib.request
import pathlib
import pandas as pd
import duckdb

# --- Download the raw data files ---
DATA_DIR = pathlib.Path("data")
DATA_DIR.mkdir(exist_ok=True)
BASE_URL = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/"
FILES_TO_DOWNLOAD = {
    "airports": "airports.dat",
    "airlines": "airlines.dat",
    "routes": "routes.dat",
    "countries": "countries.dat",
    "planes": "planes.dat"
}
# Define column names as the files don't have headers
COLUMN_NAMES = {
    "airports": ["airport_id", "name", "city", "country", "iata", "icao", "latitude", "longitude", "altitude", "timezone", "dst", "tz_db", "type", "source"],
    "airlines": ["airline_id", "name", "alias", "iata", "icao", "callsign", "country", "active"],
    "routes": ["airline", "airline_id", "source_airport", "source_airport_id", "destination_airport", "destination_airport_id", "codeshare", "stops", "equipment"],
    "countries": ["name", "iso_code", "dafif_code"],
    "planes": ["name", "iata", "icao"]
}

PROD_DB_PATH = "data/prod_openflights.db"

# --- Load the real data into our "production" DuckDB ---
with duckdb.connect(PROD_DB_PATH) as con:
    for name, filename in FILES_TO_DOWNLOAD.items():
        url = f"{BASE_URL}{filename}"
        path = DATA_DIR / filename
        if not path.exists():
            urllib.request.urlretrieve(url, path)
            print(f"✅ Downloaded: {path}")

        # Load data using pandas to handle missing headers and null values
        df = pd.read_csv(path, header=None, names=COLUMN_NAMES[name], na_values=["\\N"])
        con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM df")

    print(f"\n✅ 'Production' database simulated at: {PROD_DB_PATH}")
    print("Tables created:", con.sql("SHOW TABLES;").fetchall())
```

---

## 3. 📋 Acquire the Schema (No Real Data!)

This is a critical step. We connect to our "production" database and extract **only its schema** (the table structure, column names, and data types). We do not touch or read any of the data rows. This schema is the only artifact we need from the production environment.

### Why Schema-Only?
This approach is powerful because:
- **Privacy**: No actual customer data leaves your production environment
- **Security**: No risk of exposing sensitive data during fine-tuning
- **Efficiency**: Schema information is tiny compared to actual data

The `DESCRIBE` command in DuckDB gives us comprehensive schema information without accessing any rows.

```python
import duckdb

# Connect to the "production" database we just created
with duckdb.connect(PROD_DB_PATH, read_only=True) as con:
    # The DESCRIBE command gives us the schema information for all tables
    schema_df = con.sql("DESCRIBE;").df()

print("✅ Schema successfully extracted from 'production' database:")
print(schema_df.to_markdown(index=False))

# We can also store this for later use in prompts
schema_for_prompt = schema_df.to_markdown(index=False)
```

## 4. 🧪 Create the Synthetic Training Sandbox with an LLM

Now that we have the schema, we will use a large language model to generate a complete, contextually-aware synthetic dataset.

### Key Concepts in This Step:

**Dynamic Pydantic Model Generation**: We dynamically create Pydantic models based on your database schema. This ensures the LLM's output is structured and parseable, adapting to any database schema automatically.

**Chunked Generation Strategy**: Instead of asking for all data at once (which could overwhelm the LLM or hit token limits), we generate data in small chunks of 2 rows per API call. This approach:
- Ensures high-quality, coherent data
- Avoids token limit issues

**Contextual Awareness**: Each generation request includes previously generated data as context, preventing duplicates and ensuring variety.

To fine-tune our model with RFT, **we will only interact with this synthetic database.**

> **Real World 🌍**: This pattern is directly applicable. You would use the same approach with your production schema to generate synthetic data that maintains the statistical properties and relationships of your real data without exposing any actual records.

```python
import pandas as pd
import os
from pydantic import create_model, BaseModel
from fireworks import LLM
import duckdb
import json
from dotenv import load_dotenv
from typing import List, Optional, Any, Dict, Type
import datetime
import decimal
import uuid
import math
import time


TARGET_ROW_COUNT = 100  # The number of rows to generate for each table.

# --- 1. Dynamically Create Pydantic Models from the SQL Schema ---
def map_sql_type_to_python(sql_type: str) -> Type:
    """Maps SQL data types to Python types for Pydantic models."""
    sql_type_upper = str(sql_type).upper()
    if 'DECIMAL' in sql_type_upper: return decimal.Decimal
    if 'DOUBLE' in sql_type_upper or 'FLOAT' in sql_type_upper or 'REAL' in sql_type_upper: return float
    if 'BIGINT' in sql_type_upper or 'INT' in sql_type_upper: return int
    if 'VARCHAR' in sql_type_upper or 'TEXT' in sql_type_upper or 'STRING' in sql_type_upper: return str
    if 'TIMESTAMP' in sql_type_upper: return datetime.datetime
    if 'DATE' in sql_type_upper: return datetime.date
    if 'TIME' in sql_type_upper: return datetime.time
    if 'BOOLEAN' in sql_type_upper: return bool
    if 'BLOB' in sql_type_upper or 'BYTEA' in sql_type_upper: return bytes
    if 'UUID' in sql_type_upper: return uuid.UUID
    return object

pydantic_models: Dict[str, Type[BaseModel]] = {}
table_names = schema_df['name'].unique()

for table_name in table_names:
    table_schema = schema_df[schema_df['name'] == table_name].iloc[0]
    fields: Dict[str, Any] = {}
    col_names = table_schema['column_names']
    col_types = table_schema['column_types']
    for i, col_name in enumerate(col_names):
        python_type = map_sql_type_to_python(col_types[i])
        fields[col_name] = (Optional[python_type], None)
    model_name = table_name.capitalize() + "Model"
    pydantic_models[table_name] = create_model(model_name, **fields)

dataset_fields: Dict[str, Any] = {
    table_name: (List[model], ...) for table_name, model in pydantic_models.items()
}
SyntheticDataset = create_model('SyntheticDataset', **dataset_fields)
print("✅ Dynamically created Pydantic models for all tables.")


# --- 2. Define Total Row Counts and Chunking Strategy ---
TOTAL_ROW_COUNTS = {name: TARGET_ROW_COUNT for name in table_names}
ROWS_PER_API_CALL = 2 # Ask for data in small, safe chunks
print("\n✅ Data Generation Plan:")
print(f" - Target rows per table: {list(TOTAL_ROW_COUNTS.values())[0]}")
print(f" - Will make API calls asking for {ROWS_PER_API_CALL} rows/call until target is met.")


# --- 3. Setup LLM and Loop to Generate Data in Chunks ---
SYNTHETIC_DB_PATH = "data/synthetic_openflights.db"
load_dotenv()
llm = LLM(model="accounts/fireworks/models/deepseek-v3", deployment_type="serverless", api_key=os.getenv("FIREWORKS_API_KEY"))

all_synthetic_data: Dict[str, List[Dict]] = {name: [] for name in table_names}
chunk_row_counts = {name: ROWS_PER_API_CALL for name in table_names}

base_generation_prompt = f"""
You are a highly intelligent AI data generator. Your task is to create a realistic, synthetic dataset based on the provided database schema.
The data you generate must be internally consistent. For example, an `airline_id` in a `routes` table must correspond to an existing `airline_id` in an `airlines` table within this same generated chunk.
This applies to any schema you might be working with, not just airline-related data.
You must generate a single JSON object that strictly adheres to the provided JSON schema.

The database schema is as follows:
{schema_for_prompt}
"""

call_count = 0
# Loop until all tables have at least the desired number of rows
while not all(len(rows) >= TOTAL_ROW_COUNTS[name] for name, rows in all_synthetic_data.items()):
    call_count += 1
    print(f"\n📞 --- Generating data chunk #{call_count} ---")
    
    # --- Create a summary of existing data to guide the LLM ---
    existing_data_summary = ""
    if any(len(rows) > 0 for rows in all_synthetic_data.values()):
        summary_parts = ["\nYou have already generated the following data. Do NOT generate rows that are substantially similar to these examples. Create new, unique data.\n"]
        for table_name, rows in all_synthetic_data.items():
            if rows:
                summary_parts.append(f"\n--- Existing data in '{table_name}' table ---")
                sample_rows = rows[-100:] if len(rows) > 100 else rows  # sample the last 100 rows
                df = pd.DataFrame(sample_rows)
                if len(df.columns) > 10:
                    df = df.iloc[:, :10]
                markdown_summary = df.to_markdown(index=False, tablefmt="grid")
                if markdown_summary:
                    summary_parts.append(markdown_summary)
        existing_data_summary = "\n".join(summary_parts)


    # --- Construct the final prompt for this iteration ---
    final_prompt = (
        base_generation_prompt +
        existing_data_summary +
        f"\n\nNow, generate a NEW JSON object with a key for each table. The number of new rows for each table should be:\n" +
        json.dumps(chunk_row_counts, indent=2)
    )

    response = llm.chat.completions.create(
        messages=[{"role": "user", "content": final_prompt}],
        response_format={"type": "json_schema", "json_schema": {"name": "SyntheticDataset", "schema": SyntheticDataset.model_json_schema()}},
        temperature=0.7
    )

    choice = response.choices[0]
    response_content = choice.message.content

    if choice.finish_reason == "length":
        print(f"⚠️ WARNING: Chunk #{call_count} was truncated. Skipping.")
        continue
    if not response_content:
        print(f"⚠️ WARNING: Received empty content for chunk #{call_count}. Skipping.")
        continue

    try:
        chunk_data = json.loads(response_content)
        print(f"✅ Received and parsed chunk #{call_count}.")
        for table_name, rows in chunk_data.items():
            if table_name in all_synthetic_data and rows:
                all_synthetic_data[table_name].extend(rows)
        # Log progress
        for name, rows in all_synthetic_data.items():
             print(f"   - '{name}': {len(rows)} / {TOTAL_ROW_COUNTS[name]} rows")
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Failed to parse JSON for chunk #{call_count}. Reason: {e}. Skipping.")
    
    time.sleep(1)

# --- 4. Deduplicate and Write to DB ---
print("\n✨ Data generation complete. Aggregating, deduplicating, and saving to database...")

synthetic_data = all_synthetic_data
print("\n--- Deduplicating generated data ---")
for table_name, rows in synthetic_data.items():
    if not rows: continue
    initial_count = len(rows)
    df = pd.DataFrame(rows).drop_duplicates()
    final_count = len(df)
    synthetic_data[table_name] = df.to_dict('records')
    print(f" - Table '{table_name}': Removed {initial_count - final_count} duplicates ({initial_count} -> {final_count}).")

# Final trim to ensure exact counts
for table_name, total_rows_needed in TOTAL_ROW_COUNTS.items():
    if table_name in synthetic_data:
        synthetic_data[table_name] = synthetic_data[table_name][:total_rows_needed]

with duckdb.connect(SYNTHETIC_DB_PATH) as con:
    for table_name, rows in synthetic_data.items():
        if rows:
            df = pd.DataFrame(rows)
            schema_cols = schema_df[schema_df['name'] == table_name].iloc[0]['column_names']
            for col in schema_cols:
                if col not in df.columns: df[col] = None
            df = df[schema_cols]
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM df")
    
    print(f"\n✅ Synthetic training sandbox created at: {SYNTHETIC_DB_PATH}")
    print("Tables created:", con.sql("SHOW TABLES;").fetchall())
```

---

## 5. ✅ Validate the Sandbox

Let's run a few queries against our new synthetic database to ensure the LLM did a good job generating plausible, interconnected data.

We expect to see non-empty, realistic-looking data that follows the schema constraints.

```python
import duckdb
from tabulate import tabulate

SYNTHETIC_DB_PATH = "data/synthetic_openflights.db"

# Connect to the synthetic database
with duckdb.connect(SYNTHETIC_DB_PATH, read_only=True) as con:
    
    # Get the list of all tables created
    all_tables = [table[0] for table in con.sql("SHOW TABLES;").fetchall()]
    
    # Select the first 5 tables to display (or all if fewer than 5)
    tables_to_validate = all_tables[:5]

    print("--- Validating the first few tables in the synthetic sandbox ---\n")

    # Execute and print results for the selected tables
    for table_name in tables_to_validate:
        print(f"--- SELECT * FROM {table_name} LIMIT 10; ---")
        try:
            result_df = con.sql(f"SELECT * FROM {table_name} LIMIT 10;").df()
            if not result_df.empty:
                print(tabulate(result_df, headers='keys', tablefmt='psql'))
            else:
                print(f"(Table '{table_name}' is empty)")
        except Exception as e:
            print(f"Query failed for table '{table_name}': {e}")
        print("\n")
```

## 6. 📝 Generate Example SQL Queries

With our synthetic database in place, the next step is to create a set of synthetic SQL queries. These SQL queries will be executed against our database of synthetic data to get the ground truth labels for RFT. Furthermore, these same SQL queries will be used as input to an LLM to generate queries in natural language. This will enable us to form our final RFT dataset, which pairs natural language queries with ground truth results from the database.

### Query Generation Strategy:
- **Diversity**: We want queries covering different SQL features (JOINs, GROUP BY, aggregates)
- **Complexity Range**: From simple SELECT statements to complex multi-table joins
- **Deterministic Results**: Queries include ORDER BY clauses where necessary to break ties and ensure consistent results
- **MECE Principle**: Mutually Exclusive, Collectively Exhaustive - covering all major query patterns

> **Real World 🌍**: You would use a historical log of real SQL queries that have been run against your production database. These logs are the most valuable source of training data because they represent the *actual* way your users query your data.

```python
import pandas as pd
import json
import time
from pydantic import BaseModel, Field
from typing import List
from fireworks import LLM
import os
import duckdb
from dotenv import load_dotenv

load_dotenv()

# --- 1. Define Generation Parameters and Pydantic Model ---
llm = LLM(model="accounts/fireworks/models/qwen3-coder-480b-a35b-instruct", deployment_type="serverless", api_key=os.getenv("FIREWORKS_API_KEY"))  # Use Qwen3-coder for SQL queries
TOTAL_QUERIES_TO_GENERATE = 1000  # Note, some of these queries will likely be duplicates or invalid, reducing the final number used for fine-tuning
QUERIES_PER_API_CALL = 30

class SqlQueryBatch(BaseModel):
    queries: List[str] = Field(description=f"A list of exactly {QUERIES_PER_API_CALL} unique and diverse SQL queries.")

print(f"🎯 Goal: Generate {TOTAL_QUERIES_TO_GENERATE} unique queries in batches of {QUERIES_PER_API_CALL}.")

# --- 2. Get Clean Schema From Synthetic DB ---
with duckdb.connect(SYNTHETIC_DB_PATH, read_only=True) as con:
    schema_df = con.sql("DESCRIBE;").df()
    schema_for_prompt = schema_df.to_markdown(index=False)

# --- 3. Setup Base Prompt and Generation Loop ---
base_query_generation_prompt = f"""
You are an expert SQL data analyst. Your task is to generate unique and diverse SQL queries based on the database schema provided.
The queries should be realistic and cover a range of complexities and SQL features (JOINS, GROUP BY, aggregates, etc.).
Ensure you break ties with ORDER BY clauses so that the same queries produce the same results when executed against the database.
Write on the SQL query and nothing else.
Ensure the generated SQL is valid for DuckDB.

**Database Schema:**
{schema_for_prompt}
"""

all_generated_queries = []
# Loop until we have enough queries
while len(all_generated_queries) < TOTAL_QUERIES_TO_GENERATE:
    print(f"\n📞 --- Generating batch #{len(all_generated_queries) // QUERIES_PER_API_CALL + 1} ---")

    # Create a summary of queries generated so far to prevent duplicates
    existing_queries_summary = ""
    if all_generated_queries:
        summary_parts = ["\nYou have already generated the following queries:\n"]
        for i, q in enumerate(all_generated_queries):
            summary_parts.append(f"{i+1}. {q}")
        existing_queries_summary = "\n".join(summary_parts)

    # Construct the final prompt for this iteration
    final_prompt = (
        base_query_generation_prompt +
        existing_queries_summary +
        f"\n\nNow, generate {QUERIES_PER_API_CALL} new, unique SQL queries, which cover different analytic scenarios and are not already in the list above. Return your response as a single JSON object adhering to the specified schema."
    )

    response = llm.chat.completions.create(
        messages=[{"role": "user", "content": final_prompt}],
        response_format={"type": "json_schema", "json_schema": {"name": "SqlQueryBatch", "schema": SqlQueryBatch.model_json_schema()}},
        temperature=0.8
    )

    response_content = response.choices[0].message.content
    if response_content:
        try:
            new_queries = json.loads(response_content).get("queries", [])
            all_generated_queries.extend(new_queries)
            print(f"   - Received {len(new_queries)} new queries. Total now: {len(all_generated_queries)} / {TOTAL_QUERIES_TO_GENERATE}")
        except json.JSONDecodeError as e:
            print(f"❌ ERROR: Failed to parse generated queries in this batch: {e}")
    
    time.sleep(1) # Be nice to the API

# --- 4. Deduplicate, Trim, and Save --- 
print("\n✨ Generation complete. Deduplicating and saving...")
initial_count = len(all_generated_queries)
# Simple, fast deduplication preserving order
unique_queries = list(dict.fromkeys(all_generated_queries))
final_count = len(unique_queries)
print(f" - Removed {initial_count - final_count} duplicates ({initial_count} -> {final_count}).")

# Trim to the exact number we need
final_queries = unique_queries[:TOTAL_QUERIES_TO_GENERATE]

# Save the final list to a file
QUERIES_FILE_PATH = "data/generated_queries.json"
with open(QUERIES_FILE_PATH, 'w') as f:
    json.dump({"queries": final_queries}, f, indent=2)

print(f"\n✅ Successfully saved {len(final_queries)} unique queries to `{QUERIES_FILE_PATH}`.")
print("\n--- Here are a few examples: ---")
for query in final_queries[:5]:
    print(f"- {query}")
```

---

## 7. 🎯 Execute Queries to Get Ground-Truth Answers

Now we will act as the "system" and run the queries we just generated against our synthetic sandbox. The output of each query is the **ground-truth result**. During Reinforcement Fine-Tuning, our model will be rewarded if the SQL it writes produces this exact same result.

### Why RFT is a good choice for a text-to-SQL use-case:
In RFT, the model explores the space of possible SQL queries during fine-tuning; the reward signal comes from comparing the result of executing the model's output SQL queries against the ground truth expected results. This is fundamentally different from SFT, where the model learns to mimic the exact SQL syntax. With RFT:
- Multiple SQL queries can be "correct" if they produce the same result
- The model learns to reason about the problem rather than memorize solutions
- Edge cases and query optimization patterns can emerge naturally

> **Real World 🌍**: You would run your real historical queries against the synthetic database we previously created. The correctness of the data is not a concern here, as our aim is to see what a correct query would have generated, so we can compare it to our LLM's generations during the RFT process.

```python
import duckdb
import json
import pandas as pd

# --- 1. Define File Paths ---
SYNTHETIC_DB_PATH = "data/synthetic_openflights.db"
QUERIES_FILE_PATH = "data/generated_queries.json"
GROUND_TRUTH_FILE_PATH = "data/ground_truth_results.jsonl"

# --- 2. Load Generated Queries ---
with open(QUERIES_FILE_PATH, 'r') as f:
    queries_data = json.load(f)
    queries_to_execute = queries_data.get("queries", [])

print(f"Loaded {len(queries_to_execute)} queries to execute.")

# --- 3. Execute Queries and Store Results ---
ground_truth_results = []
successful_executions = 0
failed_executions = 0

print("Executing queries against the synthetic database...")
with duckdb.connect(SYNTHETIC_DB_PATH, read_only=True) as con:
    for query in queries_to_execute:
        try:
            # Execute the query and convert the result to a pandas DataFrame
            result_df = con.sql(query).df()

            # Replace any NaN/NaT values with None, which serializes to JSON `null`
            result_df = result_df.astype(object).where(pd.notna(result_df), None)
            
            result_records = result_df.to_dict('records')
            
            # Pair the query with its result
            ground_truth_results.append({
                "query": query,
                "result": result_records
            })
            successful_executions += 1
        except Exception as e:
            # The LLM might have occasionally generated a slightly invalid query
            print(f"⚠️  Skipping query due to execution error: {query}\n   Error: {e}\n")
            failed_executions += 1

print(f"\nExecution complete. Success: {successful_executions}, Failed: {failed_executions}.")

# --- 4. Save the Ground-Truth Data ---
with open(GROUND_TRUTH_FILE_PATH, 'w') as f:
    for entry in ground_truth_results:
        f.write(json.dumps(entry) + '\n')

print(f"\n✅ Successfully saved {len(ground_truth_results)} ground-truth results to `{GROUND_TRUTH_FILE_PATH}`.")

# --- 5. Print an Example ---
if ground_truth_results:
    print("\n--- Example ground_truth_results dataset entry ---")
    print(json.dumps(ground_truth_results[0], indent=2))
```

---

## 8. 💬 Generate Natural Language Questions for Final RFT Training Data

We now have pairs of `(SQL Query, Ground-Truth Result)`. The final piece missing from our training data is the user's input: a question in natural language. This is because our final goal is to use RFT to tune an LLM to map from a natural language question to a SQL query, having the reward signal be the actual result of the query, rather than just the query itself. This is important because there are many ways to write the same SQL query that yield the same, correct result.

### Thus, the complete training loop will look like this:
1. User asks: *"Which countries have the most airlines?"*
2. Model generates: *SQL query*
3. System executes: *Query against database*
4. Reward calculation: *Does result match ground truth?*
5. Model update: *Reinforce successful strategies*

Thus, we will use an LLM once again to translate our "historical" SQL queries into plausible questions a business user might ask, corresponding to that query. This will yield our final training dataset in the format: `(Natural Language Question, SQL Query, Ground-Truth Result)`. Note that the SQL queries themselves will not be used as part of the RFT job itself, but are useful for debugging our evaluation function (more details in a later section).

> **Real World 🌍**: You might not need this step! If you have logs that already link user questions to the queries they ran (e.g., from a BI tool's search bar), you can use those directly. If not, this LLM-based translation is a powerful technique to bootstrap your training data.

```python
import json
import time
import jsonlines
from typing import List
import random
from fireworks import LLM

# --- 1. Define File Paths and Parameters ---
llm = LLM(model="accounts/fireworks/models/qwen3-coder-480b-a35b-instruct", deployment_type="serverless", api_key=os.getenv("FIREWORKS_API_KEY"))
GROUND_TRUTH_FILE_PATH = "data/ground_truth_results.jsonl"
FINAL_TRAINING_DATA_PATH = "data/final_rft_sql_train_data.jsonl"
FINAL_TEST_DATA_PATH = "data/final_rft_sql_test_data.jsonl"

# --- 2. Load Ground-Truth Data ---
query_result_pairs = []
with jsonlines.open(GROUND_TRUTH_FILE_PATH) as reader:
    for obj in reader:
        query_result_pairs.append(obj)

print(f"Loaded {len(query_result_pairs)} query-result pairs.")

# --- 3. Use LLM to Generate Natural Language Questions ---
nl_generation_prompt_template = f"""
You are an expert data analyst who is great at translating SQL queries into plain English.
Based on the database schema and the provided SQL query, what is a natural language question a business user would ask to get this information?
Ensure that the question is precise enough to accurately map to the corresponding SQL query.

**Database Schema:**
{schema_for_prompt}

**SQL Query:**
{query}

Provide only the user's question, without any preamble or explanation.
"""

# The system prompt that will be included in the final training data for the RFT job.
# It gives the model its instructions at inference time.
rft_system_prompt = f"""
You are an expert SQL data analyst. Your task is to write a single, valid DuckDB SQL query to answer the user's question, based on the provided database schema. Do not provide any explanation or text other than the SQL query itself.

**Database Schema:**
{schema_for_prompt}
"""

final_generated_data = []
print(f"Generating natural language questions and formatting for RFT for {len(query_result_pairs)} queries...")

for i, pair in enumerate(query_result_pairs):
    print(f" - Processing query {i+1}/{len(query_result_pairs)}...")
    query = pair['query']
    ground_truth = pair['result']
    nl_generation_prompt = nl_generation_prompt_template.format(query=query)
    
    response = llm.chat.completions.create(
        messages=[{"role": "user", "content": nl_generation_prompt}],
        temperature=0.5
    )
    
    nl_question = response.choices[0].message.content
    if nl_question:  # Only include the entry if the LLM generated a question
        # Assemble the final data structure
        rft_entry = {
            "messages": [
                {"role": "system", "content": rft_system_prompt},
                {"role": "user", "content": nl_question.strip()},
                {"role": "assistant", "content": query}
            ],
            "ground_truth": ground_truth  # The ground-truth result for the evaluator
        }
        final_generated_data.append(rft_entry)
    
    time.sleep(1) # Be nice to the API

# --- 4. Shuffle and Split the Dataset ---
print(f"\nGenerated {len(final_generated_data)} total examples. Now splitting into train and test sets.")
random.seed(42)
random.shuffle(final_generated_data)

split_index = int(len(final_generated_data) * 0.8)
train_data = final_generated_data[:split_index]
test_data = final_generated_data[split_index:]

print(f"Train set size: {len(train_data)}")
print(f"Test set size: {len(test_data)}")

# --- 5. Save the Final RFT-Ready Datasets ---
with jsonlines.open(FINAL_TRAINING_DATA_PATH, mode='w') as writer:
    writer.write_all(train_data)
print(f"\n✅ Successfully saved training dataset to `{FINAL_TRAINING_DATA_PATH}`.")

with jsonlines.open(FINAL_TEST_DATA_PATH, mode='w') as writer:
    writer.write_all(test_data)
print(f"✅ Successfully saved test dataset to `{FINAL_TEST_DATA_PATH}`.")

# --- 6. Print an Example ---
if train_data:
    print("\n--- Example RFT training entry ---")
    print(json.dumps(train_data[0], indent=2))
```

## 9. 🛰️ Deploy an MCP Server for the Synthetic Data

Now, we'll start a remote server that speaks the Model Context Protocol (MCP). This server will wrap our synthetic DuckDB database, providing a standardized way for any external tool—in our case, the Fireworks RFT evaluator—to interact with it.

### What is MCP?
The Model Context Protocol is an open standard that standardizes how applications provide context to LLMs. Think of MCP like a USB-C port for AI applications. Just as USB-C provides a standardized way to connect your devices to various peripherals, MCP provides a standardized way to connect AI models to various data sources and tools.

Key benefits:
- **Flexibility**: Works with any data source or tool
- **Standardization**: One protocol for all integrations instead of custom APIs for each tool; MCP servers for many applications are readily available

> Real World 🌍: This pattern is directly applicable. You would run a similar MCP server to provide a secure, read-only interface to a production database replica or a data warehouse, allowing the fine-tuning process to happen without granting direct database credentials to the training environment.

9. a) Create a server script in this project's root directory (`run_mcp_server.py`). This Python script starts our database server. It is configured to be read-only.

```python
import os, contextlib, uvicorn
from starlette.applications import Starlette
from starlette.routing import Mount
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp_server_motherduck import build_application

DB = "data/synthetic_openflights.db"          # ← path from previous steps
PORT = int(os.environ.get("PORT", 8080))        # Cloud Run injects $PORT

# 1️⃣ Build the core SQL-aware MCP server (read-only for safety).
server, _ = build_application(db_path=DB, read_only=True)

# 2️⃣ Wrap it so HTTP clients can talk to it (ASGI handler).
sess = StreamableHTTPSessionManager(app=server, event_store=None, stateless=True)

async def handler(scope, receive, send):
    await sess.handle_request(scope, receive, send)

@contextlib.asynccontextmanager
async def lifespan(app):
    async with sess.run():
        yield                                        # keep sessions alive

# 3️⃣ Starlette turns that handler into a full ASGI app Uvicorn can serve.
app = Starlette(routes=[Mount("/mcp", app=handler)], lifespan=lifespan)

if __name__ == "__main__":
    print(f"🔥 MCP endpoint → http://0.0.0.0:{PORT}/mcp")
    uvicorn.run(app, host="0.0.0.0", port=PORT)
```

---

## 10. ☁️ Set Up Google Cloud CLI & .gcloudignore

We'll first set up the Google Cloud CLI and authenticate. Google Cloud Run provides an easy way to deploy containerized applications without managing infrastructure.

> **Real World 🌍**  
> You would follow along here in the same way. Cloud Run is ideal for MCP servers because it auto-scales based on demand (down to zero when not in use, thus charging only for actual usage).

10. a) **Install** the SDK (macOS/Linux):

```bash
curl -sSL https://sdk.cloud.google.com | bash
exec -l $SHELL  # reload shell so 'gcloud' is available
```


10. b) **Log in** (creates local access token):
```bash
gcloud auth login
```


10. c) **Set your active project desired gcloud project**:
```bash
gcloud config set project < YOUR_PROJECT_ID >  # set up project in gcloud console before running this if not already done
```

---

## 11. 📦 Containerize & Deploy the MCP Server

We'll build a Docker image and push it straight to Cloud Run.  
Remember to replace **`YOUR_PROJECT_ID`** with the project you actually want to bill.

> **Real World 🌍**  
> You would follow along in the same way here.

11. a) Create `mcp_requirements.txt` containing the following:


```
mcp
mcp-server-motherduck
duckdb
uvicorn
starlette
```


11. b) Create a `Dockerfile` (no extension) containing the following:
```dockerfile
FROM python:3.11-slim
WORKDIR /app

COPY mcp_requirements.txt .
RUN pip install --no-cache-dir -r mcp_requirements.txt

COPY run_mcp_server.py .
COPY data/synthetic_openflights.db ./data/

EXPOSE 8080

CMD ["python", "run_mcp_server.py"]
```


11. c) Create a .gcloudignore file in your root dir (to only deploy files needed for MCP server) containing:
```
# .gcloudignore

# 1. Ignore EVERYTHING in the directory by default.
*

# 2. Now, create exceptions for ONLY the files needed by the Dockerfile.
# The "!" character means "do not ignore this file".

# The Dockerfile itself is needed for the build process.
!Dockerfile

# The files explicitly copied by your Dockerfile:
!mcp_requirements.txt
!run_mcp_server.py

# 3. To include a specific file in a subdirectory, use this
#    three-line pattern to un-ignore the directory, re-ignore its
#    contents, and then un-ignore the specific file.
!data/
data/*
!data/synthetic_openflights.db
```


11. d) Deploy your MCP server as a Cloud Run app by running (from your project root):
```bash
FIREWORKS_API_KEY=$(grep FIREWORKS_API_KEY .env | cut -d '=' -f2) reward-kit deploy-mcp \
--id mcp-sql-rft-server \
--dockerfile Dockerfile \
--port 8080 \
--gcp-project < YOUR_GCP_PROJECT_ID > \
--gcp-region < YOUR_GCP_REGION >
```


11. e) Test that your MCP server is working as expected by running the following from your terminal:
11. e) i. To get your MCP server's URL:
```bash
gcloud run services describe mcp-sql-rft-server \
--project < YOUR_GCP_PROJECT_ID > \
--region < YOUR_GCP_REGION > \
--format="value(status.url)"
```

11. e) ii. (optional) To check the names of the MCP server's available tools:
```bash
curl -X POST "< YOUR_MCP_SERVER_URL_FROM_STEP_i >/mcp/" \
-H "Content-Type: application/json" \
-H "Accept: application/json, text/event-stream" \
-d '{
    "id": "list-tools-1",
    "jsonrpc": "2.0",
    "method": "tools/list",
    "params": {
        "session": {"id": "test-from-my-laptop"}
    }
}'
```
>Note that the above is a generally useful way to check an MCP server's tools.
>In this case, the tool of interest is the "query" tool.

11. e) iii. To send a test request to the MCP server:
```bash
curl -X POST "< YOUR_MCP_SERVER_URL_FROM_STEP_i >/mcp/" \
-H "Content-Type: application/json" \
-H "Accept: application/json, text/event-stream" \
-d '{
    "id": "query-1",
    "jsonrpc": "2.0",
    "method": "tools/call",
    "params": {
        "session": {"id": "test-from-my-laptop"},
        "name": "query",
        "arguments": {
            "query": "SELECT COUNT(*) FROM airlines;"
        }
    }
}'
```

---

## 12. 🔍 Define an evaluation function for RFT

Here, we define an `evaluate` function for RFT, which will interface with our MCP server. Note that you will not directly execute the function here, but will use it as part of the Fireworks Evaluations UI.

### Understanding the Evaluation Function:
The evaluation function is the heart of RFT. It:
1. Receives the model's generated SQL query
2. Executes it against the real database (via MCP)
3. Compares the result with ground truth
4. Returns a reward score (0 or 1)

This binary reward signal drives the reinforcement learning process. The model learns through trial and error which SQL patterns lead to correct results.

Key design decisions:
- **Exact match comparison**: We normalize values and sort rows to handle different but equivalent result orderings
- **Robust error handling**: SQL syntax errors or execution failures return a score of 0
- **Detailed reasoning**: The function returns explanatory messages for debugging

Ensure that you set MCP_SERVER_URL to be your actual MCP server URL from step 11. e) i.

> **Real World 🌍**  
> You would follow along in the same way here. The evaluation function could also be further customized, with, for example:
> - Partial credit for near-correct answers
> - Performance-based rewards (faster queries get higher scores)

```python
import requests
import json
import math

MCP_SERVER_URL = None  # <--- PUT MCP SERVER URL HERE without the /mcp/ suffix at the end

def evaluate(messages: list[dict], ground_truth: list[dict], **kwargs) -> dict:
    """
    Evaluates the model's generated SQL query by executing it against a live
    MCP server and comparing the result with the ground_truth.
    """
    
    def parse_duckdb_ascii_table(table_string: str) -> list[dict]:
        """
        Parses a DuckDB-style ASCII table string into a list of dictionaries.
        This version robustly handles 'NULL' values and empty strings.
        """
        lines = table_string.strip().split('\n')
        content_lines = [line for line in lines if line.strip() and not line.startswith('+')]
        if len(content_lines) < 2:
            return []
        
        header_raw = [h.strip() for h in content_lines[0].split('|')[1:-1]]
        data_lines = content_lines[1:]
        
        if len(data_lines) > 0:
            try:
                first_data_values = [v.strip() for v in data_lines[0].split('|')[1:-1]]
                if len(first_data_values) == len(header_raw) and all(v.isupper() for v in first_data_values):
                    data_lines = data_lines[1:]
            except IndexError:
                pass

        rows = []
        for line in data_lines:
            try:
                values_raw = [v.strip() for v in line.split('|')[1:-1]]
                if len(values_raw) == len(header_raw):
                    row_dict = {}
                    for i, header in enumerate(header_raw):
                        value_str = values_raw[i]
                        if value_str.upper() == 'NULL' or value_str == '':
                            row_dict[header] = None
                            continue
                        
                        try:
                            if '.' in value_str:
                                row_dict[header] = float(value_str)
                            else:
                                row_dict[header] = int(value_str)
                        except (ValueError, TypeError):
                            row_dict[header] = value_str
                    rows.append(row_dict)
            except IndexError:
                continue
        return rows

    # --- 1. Get MCP Server URL from Environment Variables ---
    mcp_server_url = MCP_SERVER_URL
    if not mcp_server_url:
        return {"score": 0, "is_score_valid": False, "reason": "FATAL: MCP_SERVER_URL environment variable is not set."}

    # --- 2. Get the SQL query from the model's response ---
    sql_query = messages[-1]['content'].strip()
    if not sql_query:
        return {"score": 0, "reason": "Model returned an empty response."}

    # --- 3. Execute the Query against the MCP Server ---
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/event-stream"
    }
    payload = {
        "id": "eval-query-1", "jsonrpc": "2.0", "method": "tools/call",
        "params": {"session": {"id": "stateless-eval-session"}, "name": "query", "arguments": {"query": sql_query}}
    }
    try:
        with requests.post(f"{mcp_server_url}/mcp/", headers=headers, json=payload, timeout=15, stream=True) as response:
            response.raise_for_status()
            response_data = None
            for line in response.iter_lines():
                if line:
                    decoded_line = line.decode('utf-8')
                    if decoded_line.startswith('data:'):
                        json_part = decoded_line[len('data:'):].strip()
                        if json_part:
                            response_data = json.loads(json_part)
                            break
            if response_data is None:
                return {"score": 0, "reason": "Could not find JSON data in event stream response from MCP server."}

        if "error" in response_data:
            return {"score": 0, "reason": f"SQL execution failed. Error: {response_data['error'].get('message', 'Unknown')}"}

        ascii_table = response_data['result']['content'][0]['text']
        predicted_rows = parse_duckdb_ascii_table(ascii_table)

    except requests.exceptions.RequestException as e:
        return {"score": 0, "reason": f"Network error calling MCP server: {e}"}
    except json.JSONDecodeError as e:
        return {"score": 0, "reason": f"JSON decode error from server response: {e}"}
    except (KeyError, IndexError):
        return {"score": 0, "reason": f"Failed to parse predicted result from MCP server response structure. Data found: {json.dumps(response_data)}"}
    except Exception as e:
        return {"score": 0, "reason": f"An unexpected error occurred during query execution: {e}"}

    # --- 4. Process Ground Truth ---
    if not isinstance(ground_truth, list):
        return {"score": 0, "is_score_valid": False, "reason": f"FATAL: ground_truth is not a list as expected. Got type: {type(ground_truth)}"}
    ground_truth_rows = ground_truth


    # --- 5. Comparison Logic ---
    def normalize_and_stringify(v):
        """
        Normalizes numbers and None before string conversion.
        """
        if v is None:
            return str(v)
        
        if isinstance(v, float) and not math.isinf(v) and not math.isnan(v) and v == int(v):
            v = int(v)
        return str(v)

    try:
        gt_values = sorted([sorted(map(normalize_and_stringify, row.values())) for row in ground_truth_rows])
        predicted_values = sorted([sorted(map(normalize_and_stringify, row.values())) for row in predicted_rows])

        if gt_values == predicted_values:
            score = 1
            reason = "Success: The SQL query produced the exact expected result."
        else:
            score = 0
            gt_json = json.dumps(ground_truth_rows)
            pred_json = json.dumps(predicted_rows)
            reason = f"Incorrect result. Expected (from ground_truth): {gt_json}. Got (from query): {pred_json}."
    
    except Exception as e:
        return {"score": 0, "reason": f"Error during result comparison: {e}"}

    return {"score": score, "reason": reason}
```

---

## 13. 🧪 Test English -> SQL of a base model without fine-tuning

Here, we test a base model's ability to generate SQL from a natural language question on a single example from our training data.

This is a quick sanity check that:
1. **Verifies your MCP server is working**: Ensures the server is accessible and can execute queries
2. **Tests the full pipeline**: Confirms that the flow from natural language → SQL generation → execution → result parsing works end-to-end
3. **Shows a concrete example**: Demonstrates what happens when an off-the-shelf model tries to answer a question about your specific database

The test process:
1. Load one example from your training data (by default, the first row)
2. Feed the natural language question to a base model (e.g., Llama 3.1 8B)
3. Execute whatever SQL the model generates against your MCP server
4. Compare the result to the ground truth
5. Print whether it succeeded or failed

What to expect:
- The base model might get it right! Simple queries often work.
- Or, you'll see some kind of failure: wrong column names, missing aliases, incorrect syntax, etc.
- Either outcome is fine; this is just a quick test to see the model in action before fine-tuning.

To try different examples, change `ROW_INDEX_TO_TEST` to test other rows from your dataset.

Ensure that you set MCP_SERVER_URL to be your actual MCP server URL from step 11. e) i.

> **Real World 🌍**  
> You can follow along in the same way here. This single-example test is just a quick way to verify everything is wired up correctly before launching the more expensive fine-tuning job.

```python
import requests
import json
import os
from fireworks import LLM

# --- 1. SETUP: Define API keys, server URLs, and the model to use ---

# IMPORTANT: Make sure your FIREWORKS_API_KEY is set as an environment variable.
# You can get one from https://fireworks.ai
if "FIREWORKS_API_KEY" not in os.environ:
    print("FATAL: FIREWORKS_API_KEY environment variable not set.")
    # If not set, you can hardcode it here for testing, but this is not recommended:
    # os.environ["FIREWORKS_API_KEY"] = "YOUR_API_KEY_HERE"

# The model we'll use to generate the SQL. This acts as our "base" model.
LLM_MODEL = "accounts/fireworks/models/llama-v3p1-8b-instruct"
llm = LLM(model=LLM_MODEL, deployment_type="auto", api_key=os.getenv("FIREWORKS_API_KEY"))

# The URL for your running MCP server.
MCP_SERVER_URL = None  # PUT MCP SERVER URL HERE without the /mcp/ suffix at the end


# --- 2. LOAD THE EXAMPLE DATA ---

# This is the example data you provided.
DATASET_FILE_PATH = "data/final_rft_sql_train_data.jsonl"
ROW_INDEX_TO_TEST = 0  # 0 is the first row, 1 is the second row, etc.

EXAMPLE_DATA = None
try:
    with open(DATASET_FILE_PATH, 'r') as f:
        for i, line in enumerate(f):
            if i == ROW_INDEX_TO_TEST:
                EXAMPLE_DATA = json.loads(line)
                break
    
    if EXAMPLE_DATA is None:
        with open(DATASET_FILE_PATH, 'r') as f:
            line_count = sum(1 for line in f)
        raise IndexError(f"row index {ROW_INDEX_TO_TEST} is out of bounds for file with {line_count} rows.")

    print(f"Successfully loaded row {ROW_INDEX_TO_TEST} from '{DATASET_FILE_PATH}'.\n")

except Exception as e:
    print(f"Warning: Could not load from file. Reason: {e}")

# If loading from file failed for any reason, use the hardcoded fallback data.
if EXAMPLE_DATA is None:
    print("Using hardcoded fallback EXAMPLE_DATA.\n")
    EXAMPLE_DATA = {
        "messages": [
            {"role": "system", "content": "\nYou are an expert SQL data analyst. Your task is to write a single, valid DuckDB SQL query to answer the user's question, based on the provided database schema. Do not provide any explanation or text other than the SQL query itself.\n\n**Database Schema:**\n| database              | schema   | name      | column_names                                                               | column_types                                                          | temporary   |\n|:----------------------|:---------|:----------|:---------------------------------------------------------------------------|:----------------------------------------------------------------------|:------------|\n| synthetic_openflights | main     | airlines  | ['airline_id' 'name' 'alias' 'iata' 'icao' 'callsign' 'country' 'active']  | ['BIGINT' 'VARCHAR' 'VARCHAR' 'VARCHAR' 'VARCHAR' 'VARCHAR' 'VARCHAR' | False       |\n|                       |          |           |                                                                            |  'VARCHAR']                                                           |             |\n| synthetic_openflights | main     | airports  | ['airport_id' 'name' 'city' 'country' 'iata' 'icao' 'latitude' 'longitude' | ['BIGINT' 'VARCHAR' 'VARCHAR' 'VARCHAR' 'VARCHAR' 'VARCHAR' 'DOUBLE'  | False       |\n|                       |          |           |  'altitude' 'timezone' 'dst' 'tz_db' 'type' 'source']                      |  'DOUBLE' 'BIGINT' 'DOUBLE' 'VARCHAR' 'VARCHAR' 'VARCHAR' 'VARCHAR']  |             |\n| synthetic_openflights | main     | countries | ['name' 'iso_code' 'dafif_code']                                           | ['VARCHAR' 'VARCHAR' 'VARCHAR']                                       | False       |\n| synthetic_openflights | main     | planes    | ['name' 'iata' 'icao']                                                     | ['VARCHAR' 'VARCHAR' 'VARCHAR']                                       | False       |\n| synthetic_openflights | main     | routes    | ['airline' 'airline_id' 'source_airport' 'source_airport_id'               | ['VARCHAR' 'BIGINT' 'VARCHAR' 'BIGINT' 'VARCHAR' 'BIGINT' 'VARCHAR'   | False       |\n|                       |          |           |  'destination_airport' 'destination_airport_id' 'codeshare' 'stops'        |  'BIGINT' 'VARCHAR']                                                  |             |\n|                       |          |           |  'equipment']                                                              |                                                                       |             |\n"},
            {"role": "user", "content": "Which countries have the most airlines, and how many airlines are there in each country, listed in descending order by the number of airlines and then alphabetically by country name?"},
            {"role": "assistant", "content": "SELECT country, COUNT(*) AS airline_count FROM airlines GROUP BY country ORDER BY airline_count DESC, country ASC"}
        ],
        "ground_truth": [{"country": "Canada", "airline_count": 10}, {"country": "Sweden", "airline_count": 10}, {"country": "Kenya", "airline_count": 9}, {"country": "United States", "airline_count": 9}, {"country": "Australia", "airline_count": 8}, {"country": "Spain", "airline_count": 6}, {"country": "Italy", "airline_count": 4}, {"country": "Switzerland", "airline_count": 4}, {"country": "Finland", "airline_count": 3}, {"country": "France", "airline_count": 3}, {"country": "Mexico", "airline_count": 3}, {"country": "Costa Rica", "airline_count": 2}, {"country": "Germany", "airline_count": 2}, {"country": "Iceland", "airline_count": 2}, {"country": "Ireland", "airline_count": 2}, {"country": "Japan", "airline_count": 2}, {"country": "Norway", "airline_count": 2}, {"country": "Singapore", "airline_count": 2}, {"country": "United Kingdom", "airline_count": 2}, {"country": "Argentina", "airline_count": 1}, {"country": "Brazil", "airline_count": 1}, {"country": "China", "airline_count": 1}, {"country": "Egypt", "airline_count": 1}, {"country": "Fiji", "airline_count": 1}, {"country": "Greece", "airline_count": 1}, {"country": "India", "airline_count": 1}, {"country": "Jordan", "airline_count": 1}, {"country": "Netherlands", "airline_count": 1}, {"country": "New Zealand", "airline_count": 1}, {"country": "Portugal", "airline_count": 1}, {"country": "Saudi Arabia", "airline_count": 1}, {"country": "South Africa", "airline_count": 1}, {"country": "Thailand", "airline_count": 1}, {"country": "United Arab Emirates", "airline_count": 1}]
    }

# Extract the prompts and ground truth from the data
system_prompt = EXAMPLE_DATA["messages"][0]["content"]
user_prompt = EXAMPLE_DATA["messages"][1]["content"]
GROUND_TRUTH_ROWS = EXAMPLE_DATA["ground_truth"]

# --- 3. HELPER FUNCTION: To parse the server's ASCII table response ---

def parse_duckdb_ascii_table(table_string: str) -> list[dict]:
    """
    Parses a DuckDB-style ASCII table string into a list of dictionaries.
    This version robustly handles 'NULL' values and empty strings.
    """
    lines = table_string.strip().split('\n')
    content_lines = [line for line in lines if line.strip() and not line.startswith('+')]
    if len(content_lines) < 2:
        return []
    
    header_raw = [h.strip() for h in content_lines[0].split('|')[1:-1]]
    data_lines = content_lines[1:]
    
    if len(data_lines) > 0:
        try:
            first_data_values = [v.strip() for v in data_lines[0].split('|')[1:-1]]
            if len(first_data_values) == len(header_raw) and all(v.isupper() for v in first_data_values):
                data_lines = data_lines[1:]
        except IndexError:
            pass

    rows = []
    for line in data_lines:
        try:
            values_raw = [v.strip() for v in line.split('|')[1:-1]]
            if len(values_raw) == len(header_raw):
                row_dict = {}
                for i, header in enumerate(header_raw):
                    value_str = values_raw[i]
                    if value_str.upper() == 'NULL' or value_str == '':
                        row_dict[header] = None
                        continue
                    
                    try:
                        if '.' in value_str:
                            row_dict[header] = float(value_str)
                        else:
                            row_dict[header] = int(value_str)
                    except (ValueError, TypeError):
                        row_dict[header] = value_str
                rows.append(row_dict)
        except IndexError:
            continue
    return rows

# --- 4. GENERATE SQL QUERY USING THE LLM ---

print("="*20)
print("LLM QUERY GENERATION")
print("="*20)

model_generated_sql = ""
try:
    print(f"Calling model '{LLM_MODEL}' to generate SQL query...")
    
    messages_for_llm = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=messages_for_llm,
        temperature=0.0  # Set to 0 for deterministic output
    )
    
    model_generated_sql = response.choices[0].message.content.strip()
    print("\nModel Generated SQL Query:")
    print(model_generated_sql)
    
except Exception as e:
    print(f"\nAN ERROR OCCURRED during LLM call: {e}")


# --- 5. EXECUTE GENERATED QUERY ON MCP SERVER ---

predicted_rows = []
if model_generated_sql:
    try:
        print("\n" + "="*20)
        print("MCP SERVER EXECUTION")
        print("="*20)
        print(f"Sending query to MCP server...")
        
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        payload = {
            "id": "eval-query-1", "jsonrpc": "2.0", "method": "tools/call",
            "params": {"session": {"id": "stateless-eval-session"}, "name": "query", "arguments": {"query": model_generated_sql}}
        }

        with requests.post(f"{MCP_SERVER_URL}/mcp/", headers=headers, json=payload, timeout=20, stream=True) as response:
            response.raise_for_status()
            response_data = None
            for line in response.iter_lines():
                if line and line.decode('utf-8').startswith('data:'):
                    json_part = line.decode('utf-8')[len('data:'):].strip()
                    if json_part:
                        response_data = json.loads(json_part)
                        break
            
            if response_data is None: raise RuntimeError("No JSON data in event stream.")
            if "error" in response_data: raise RuntimeError(f"SQL Error: {response_data['error'].get('message', 'Unknown')}")

            ascii_table = response_data['result']['content'][0]['text']
            predicted_rows = parse_duckdb_ascii_table(ascii_table)
            print("\nParsed Result from Server:")
            print(json.dumps(predicted_rows, indent=2))

    except Exception as e:
        print(f"\nAN ERROR OCCURRED during MCP call: {e}")

# --- 6. FINAL COMPARISON ---
print("\n" + "="*20)
print("COMPARISON")
print("="*20)

if not predicted_rows:
    print("Skipping comparison: no rows returned from query or an error occurred.")
else:
    gt_values = sorted([sorted(map(str, row.values())) for row in GROUND_TRUTH_ROWS])
    predicted_values = sorted([sorted(map(str, row.values())) for row in predicted_rows])

    if gt_values == predicted_values:
        print("\n✅ GOOD RESULT: The base model generated SQL that produced the correct data.\n")
    else:
        print("\n❌ BAD RESULT: The base model's SQL produced different data than expected.\n")
        print("This is often the intended outcome when testing a base model, as it highlights what fine-tuning needs to correct.")
```

## 14. 🚀 Launch the Fine-Tuning Job & Deploy via the UI

Now we'll use the Fireworks AI web interface to take our prepared dataset and fine-tune a model. This process uses your custom `evaluate` function to teach a base model how to generate SQL correctly.

### RFT vs Traditional Fine-Tuning:
Traditional supervised fine-tuning (SFT) would:
- Require thousands of examples
- Teach the model to mimic exact SQL syntax
- Often overfit to specific query patterns

Reinforcement fine-tuning (RFT) instead:
- Works with just hundreds of examples
- Rewards correct results regardless of SQL syntax
- Discovers novel solutions through exploration
- Generalizes better to unseen queries

> **Real World 🌍**  
> This is the core of the RFT process. You're teaching a general-purpose model a very specific and valuable new skill using a powerful, UI-driven workflow. You may follow along as described below

As described in the [Fireworks RFT documentation](https://fireworks.ai/docs/fine-tuning/reinforcement-fine-tuning-models), the process involves uploading your data, creating an evaluator, running the job, and deploying.


**14. a) Upload Your Dataset**

1.  Navigate to the **Datasets** tab in your [https://app.fireworks.ai](https://app.fireworks.ai) dashboard.
2.  Click **"Create Dataset"**.
3.  Upload your training file: `data/final_rft_sql_train_data.jsonl`.
4.  Give it a memorable name, like `rft-sql-train-data-v1`, and save it.


**14. b) Create the Evaluator**

1.  Navigate to the **Evaluations** tab in the dashboard.
2.  Click **"Create Evaluator"**. This will open the web IDE.
3.  In the editor on the left, replace the template code with your full `evaluate` function from step 12 above. This function already contains the logic to connect to your MCP server and compare the results. You just need to add your MCP server URL to the MCP_SERVER_URL line.
4.  Save the evaluator with a name like `rft-sql-mcp-evaluator-v1`.


**14. c) Launch the Fine-Tuning Job**

1.  Navigate to the **Fine-Tuning** tab.
2.  Click **"Fine-Tune a Model"** and select **Reinforcement**.
3.  Configure the job:
    *   **Model Selection:** Select a model, for example `qwen2p5-7b` (may appear as `Qwen2.5 7B`).
    *   **Dataset:** Select the `rft-sql-train-data-v1` you uploaded.
    *   **Evaluator:** Select the `rft-sql-mcp-evaluator-v1` you just created.
    *   **Rollout:** You can leave these as the default values.
    *   **Optional Settings:** You can leave the Model Output Name blank and get the default name, or enter a name of your choosing.
4.  You can leave most other hyperparameters as their defaults, though fine-tuning for 32 epochs (i.e., setting `Epochs` to `32`) is recommended due to the complexity of the task.
5.  Click **"Create Job"**.


**14. d) Monitor and Deploy**

1.  You can monitor the progress of your job in the **Fine-Tuning** tab.
2.  Once the job status is `Completed`, you can deploy your model. To deploy, click "Deploy" on the top right of your fine-tuning job's page. Please note:
    -  The Model under "Select base model*" should be the one from your Reinforcement Fine-Tuning job (this should be populated automatically)
    -  Speculative decoding is an advanced technique that can improve latency, but is not needed for this use-case
    -  Feel free to make the other selections (Performance, Scaling, and Metadata) as needed; enabling autoscaling is recommended to reduce costs
3.  Find this new model and click the **Deploy** button to create an API endpoint.


**14. e) Test Your New Model!**
Once deployed, copy your new model's ID and paste it into the `LLM_MODEL` variable in the testing cell (step #13) to make sure it works as expected, along with your MCP server URL (i.e., `LLM_MODEL = "accounts/<your-account-id>/models/<your-model-id>"` and `MCP_SERVER_URL = "<your-mcp-server-url>"`).

---

## 15. ⚖️ Evaluate Model Performance

Now for the moment of truth. We will systematically compare the performance of the original base model against our newly fine-tuned model, as well as a much larger base model, to quantify the improvement and general accuracy.

We'll run both models against every entry in our test dataset (final_rft_sql_test_data.jsonl). For each entry, we will:
1. Provide the same system and user prompt to both the base model and the fine-tuned model.
2. Capture the SQL query generated by each.
3. Execute each query against our live MCP server.
4. Compare the query result to the ground_truth from our dataset.
5. Keep a running score for each model.

This process will give us a clear, data-driven view of how much more accurate our model became after reinforcement fine-tuning.
> **Real World 🌍**
> This is a critical step in any MLOps loop. Evaluating a model on a consistent test set is the only way to prove that your efforts have resulted in a tangible improvement. In production, you'd also want to:
> - Track latency and cost metrics
> - Monitor for drift over time
> - A/B test against your current solution
> - Collect user feedback on query quality

```python
import requests
import json
import os
import time
from fireworks import LLM
from tqdm.auto import tqdm
from dotenv import load_dotenv

load_dotenv()

# --- 1. SETUP: Define the models to compare, server URL, and dataset path ---

# IMPORTANT: Make sure your FIREWORKS_API_KEY is set as an environment variable.
if "FIREWORKS_API_KEY" not in os.environ:
    print("FATAL: FIREWORKS_API_KEY environment variable not set.")

# The base model you used for the fine-tuning job.
BASE_MODEL_ID = "accounts/fireworks/models/qwen2p5-7b"  # <--- Replace if you used a different base model
LARGE_BASE_MODEL_ID = "accounts/fireworks/models/qwen3-235b-a22b-instruct-2507"

# IMPORTANT: Replace this with the model ID of your new fine-tuned model.
#FINE_TUNED_MODEL_ID = "accounts/<your-account-id>/models/<your-base-model-id>"  # <--- Replace with your fine-tuned model ID
FINE_TUNED_MODEL_ID = "accounts/<your-account-id>/models/<your-base-model-id>"

MCP_SERVER_URL = None  # <--- PUT MCP SERVER URL HERE without the /mcp/ suffix at the end
DATASET_FILE_PATH = "data/final_rft_sql_test_data.jsonl"

# --- 2. Create LLM Objects ---
base_model_llm = None
large_base_model_llm = None
fine_tuned_model_llm = None
try:
    base_model_llm = LLM(model=BASE_MODEL_ID, deployment_type="auto", api_key=os.getenv("FIREWORKS_API_KEY"))
    large_base_model_llm = LLM(model=LARGE_BASE_MODEL_ID, deployment_type="auto", api_key=os.getenv("FIREWORKS_API_KEY"))
    fine_tuned_model_llm = LLM(model=FINE_TUNED_MODEL_ID, deployment_type="auto", api_key=os.getenv("FIREWORKS_API_KEY"))
    print("LLM objects for all three models created successfully.")
except Exception as e:
    print(f"FATAL: Could not create LLM objects. Error: {e}")

# --- 3. Load Dataset ---
dataset = []
if all([base_model_llm, large_base_model_llm, fine_tuned_model_llm]):
    try:
        with open(DATASET_FILE_PATH, 'r') as f:
            dataset = [json.loads(line) for line in f]
        print(f"Loaded {len(dataset)} evaluation examples from '{DATASET_FILE_PATH}'.")
    except Exception as e:
        print(f"FATAL: Could not load dataset. Error: {e}")
        dataset = []

# --- 4. HELPER AND EVALUATION FUNCTIONS ---

def parse_duckdb_ascii_table(table_string: str) -> list[dict]:
    """
    Parses a DuckDB-style ASCII table string into a list of dictionaries.
    This version robustly handles 'NULL' values and empty strings.
    """
    lines = table_string.strip().split('\n')
    content_lines = [line for line in lines if line.strip() and not line.startswith('+')]
    if len(content_lines) < 2:
        return []
    
    header_raw = [h.strip() for h in content_lines[0].split('|')[1:-1]]
    data_lines = content_lines[1:]
    
    if len(data_lines) > 0:
        try:
            first_data_values = [v.strip() for v in data_lines[0].split('|')[1:-1]]
            if len(first_data_values) == len(header_raw) and all(v.isupper() for v in first_data_values):
                data_lines = data_lines[1:]
        except IndexError:
            pass

    rows = []
    for line in data_lines:
        try:
            values_raw = [v.strip() for v in line.split('|')[1:-1]]
            if len(values_raw) == len(header_raw):
                row_dict = {}
                for i, header in enumerate(header_raw):
                    value_str = values_raw[i]
                    if value_str.upper() == 'NULL' or value_str == '':
                        row_dict[header] = None
                        continue
                    
                    try:
                        if '.' in value_str:
                            row_dict[header] = float(value_str)
                        else:
                            row_dict[header] = int(value_str)
                    except (ValueError, TypeError):
                        row_dict[header] = value_str
                rows.append(row_dict)
        except IndexError:
            continue
    return rows

def are_results_equal(predicted_rows: list[dict], ground_truth_rows: list[dict]) -> bool:
    """
    Compares datasets by converting all values to strings and sorting them,
    which ignores row order, column order, and data types (e.g., int vs float).
    """
    try:
        gt_values = sorted([sorted(map(str, row.values())) for row in ground_truth_rows])
        predicted_values = sorted([sorted(map(str, row.values())) for row in predicted_rows])
        return gt_values == predicted_values
    except Exception:
        return False

def get_sql_and_evaluate(llm_obj, system_prompt: str, user_prompt: str, ground_truth: list[dict]) -> int:
    """
    Calls a pre-configured LLM object to get a SQL query, executes it, and compares to ground truth.
    Returns 1 for a correct result, 0 for an incorrect one.
    """
    try:
        # Step 1: Get SQL from the model
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        response = llm_obj.chat.completions.create(messages=messages, temperature=0.0)
        sql_query = response.choices[0].message.content.strip()

        # Step 2: Execute SQL on MCP server
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        payload = {"id": "eval-query-1", "jsonrpc": "2.0", "method": "tools/call", "params": {"session": {"id": "full-eval-session"}, "name": "query", "arguments": {"query": sql_query}}}

        response_data = None
        with requests.post(f"{MCP_SERVER_URL}/mcp/", headers=headers, json=payload, timeout=30, stream=True) as mcp_response:
            mcp_response.raise_for_status()
            for line in mcp_response.iter_lines():
                if line and line.decode('utf-8').startswith('data:'):
                    json_part = line.decode('utf-8')[len('data:'):].strip()
                    if json_part:
                        response_data = json.loads(json_part)
                        break

        if response_data is None or "error" in response_data:
            return 0

        # Step 3: Parse and compare results
        ascii_table = response_data['result']['content'][0]['text']
        predicted_rows = parse_duckdb_ascii_table(ascii_table)

        return 1 if are_results_equal(predicted_rows, ground_truth) else 0
    except Exception as e:
        print(f"--> Error during evaluation for model {llm_obj.model}: {e}")
        return 0

# --- 5. RUN THE FULL EVALUATION ---

base_model_score = 0
large_base_model_score = 0
fine_tuned_model_score = 0

if dataset:
    print("\nStarting evaluation...")
    for item in tqdm(dataset, desc="Evaluating models"):
        system_prompt = item["messages"][0]["content"]
        user_prompt = item["messages"][1]["content"]
        ground_truth = item["ground_truth"]

        # Evaluate base model
        base_model_score += get_sql_and_evaluate(base_model_llm, system_prompt, user_prompt, ground_truth)
        time.sleep(1)  # Be nice to the API

        # Evaluate large base model
        large_base_model_score += get_sql_and_evaluate(large_base_model_llm, system_prompt, user_prompt, ground_truth)
        time.sleep(1)

        # Evaluate fine-tuned model
        fine_tuned_model_score += get_sql_and_evaluate(fine_tuned_model_llm, system_prompt, user_prompt, ground_truth)
        time.sleep(1)

# --- 6. REPORT RESULTS ---

if dataset:
    total = len(dataset)
    base_accuracy = (base_model_score / total) * 100
    large_base_accuracy = (large_base_model_score / total) * 100
    tuned_accuracy = (fine_tuned_model_score / total) * 100

    print("\n" + "="*25)
    print("  EVALUATION COMPLETE")
    print("="*25)
    print(f"Total Examples: {total}\n")
    print("--- BASE MODEL ---")
    print(f"Model ID: {BASE_MODEL_ID}")
    print(f"Correct: {base_model_score}/{total}")
    print(f"Accuracy: {base_accuracy:.2f}%\n")

    print("--- LARGE BASE MODEL ---")
    print(f"Model ID: {LARGE_BASE_MODEL_ID}")
    print(f"Correct: {large_base_model_score}/{total}")
    print(f"Accuracy: {large_base_accuracy:.2f}%\n")

    print("--- FINE-TUNED MODEL ---")
    print(f"Model ID: {FINE_TUNED_MODEL_ID}")
    print(f"Correct: {fine_tuned_model_score}/{total}")
    print(f"Accuracy: {tuned_accuracy:.2f}%\n")
    
    print("="*25)
    print("  PERFORMANCE LIFT")
    print("="*25)
    print(f"Fine-Tuned vs. Base: {tuned_accuracy - base_accuracy:+.2f}%")
    print(f"Fine-Tuned vs. Large Base: {tuned_accuracy - large_base_accuracy:+.2f}%")

else:
    print("Evaluation skipped because the dataset or LLM objects could not be loaded.")
```

### Evaluation Results

**Total Examples:** 90

#### Model Performance

| Model | Model ID | Correct | Accuracy |
|-------|----------|---------|----------|
| **Base Model** | `accounts/fireworks/models/qwen2p5-7b` | 38/90 | 42.22% |
| **Large Base Model** | `accounts/fireworks/models/qwen3-235b-a22b-instruct-2507` | 48/90 | 53.33% |
| **Fine-Tuned Model** | `accounts/pyroworks/models/qwen2p5-mcp-sql-rft-tune-bigger` | 52/90 | 57.78% |

#### Performance Lift

- **Fine-Tuned vs. Base:** +15.56%

---

## 16. ✨ Cleanup & Conclusion

Congratulations! You've successfully completed the entire Reinforcement Fine-Tuning loop. You started with just a database schema and ended with a highly specialized, performant, and data-aware AI model.

### Cleanup
Cloud resources and model deployments can incur costs, so it's good practice to clean up any resources you no longer need.

*   **Check your Deployments:** Navigate to the [Deployments tab](https://app.fireworks.ai/dashboard/deployments) in your Fireworks AI dashboard. Here you can monitor and manage all your deployed models.
*   **Delete Unneeded Models:** Feel free to delete any deployments you no longer need. For example, you might have deployed the base or large-base models during the evaluation step to compare against your fine-tuned model. These can now be safely removed to save costs.
*   **Delete Cloud Run service and container image:** Feel free to delete your MCP server Cloud Run service and container image to avoid stray storage costs.

You can, of course, continue using your new fine-tuned SQL generation model for any application you see fit!

### Conclusions
The evaluation results from the previous step highlight the power of this approach.

*   **Performance on par with massive models:** Our fine-tuned 7B parameter model performs on par, or even better, than a much larger model like `qwen3-235b-a22b-instruct-2507` on this specific dataset. This is because it has been fine-tuned to understand the data schema via real query generation and execution.
*   **Efficiency Gains:** A 7B model is significantly faster and cheaper to run than a 235B one, offering production-grade performance at a fraction of the cost and latency.
*   **High-Level Capability on Complex Tasks:** The queries in this dataset are relatively complex, which is reflected in the final accuracy score of around 60%. This is a strong result, demonstrating that for a specialized domain, a smaller model can be tuned to achieve a level of performance comparable to much larger models like `qwen3-235b-a22b-instruct-2507`. Specifically, the final accuracy scores we measured for this dataset were:
    - Qwen2.5 7B (base): **42.22%** accuracy (**38/90** correct on the held-out test set)
    - Qwen3 235B Instruct (base): **53.33%** accuracy (**48/90** correct on the held-out test set)
    - Qwen2.5 7B (RFT tuned): **57.78%** accuracy (**52/90** correct on the held-out test set)

---

Throughout this tutorial, we demonstrated a complete, end-to-end workflow for creating a fine-tuned text-to-SQL model. We began with the absolute minimum requirement, a database schema, and used a series of LLM-driven steps to generate a safe, synthetic data sandbox. From there, we generated a rich dataset of queries and answers, which we used to fine-tune a model using the Fireworks RFT platform. The final result is a small, efficient model that can accurately query data it has never seen, a task that was previously only possible with vastly larger and more expensive models.

This pattern of **schema → synthetic data → RFT** is a secure, effective, and repeatable methodology for teaching language models to become expert users of your private data and custom APIs, without ever exposing the underlying sensitive information.