
# 📊 Financial Data Parser
This project is a Python-based tool for parsing Excel files containing financial data.

## Features
- Detects data types automatically
- Parses Excel files into structured JSON-like format
- Extensible architecture

##  Automatic Data Type Detection
Automatically detects cell-level data types including:
Numbers (integers, floats)
Strings
Dates and timestamps
Booleans and empty cells
Uses intelligent heuristics and regex-based checks for accuracy.

## Excel File Parsing
Reads .xlsx files using openpyxl.
Supports multi-sheet parsing.
Extracts and processes all rows and columns into structured formats.
Preserves header-row relationships and column order.

## Requirements
```bash
pip install -r requirements.txt
```
