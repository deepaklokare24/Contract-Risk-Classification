# Contract Risk Classification

A machine learning system that classifies contract deals as high/medium/low risk by analyzing descriptive text fields using LLM and RAG technology.

## Features

- Analyzes descriptive contract text using PDF knowledge sources
- Converts detailed descriptions into Yes/No values based on adherence to guidelines
- Uses RAG (Retrieval-Augmented Generation) to search PDF guidelines for relevant information
- Outputs modified CSV with converted values suitable for ML model training

## Requirements

- Python 3.10+
- OpenAI API key

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/contract-risk-classification.git
   cd contract-risk-classification
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

4. Create a `knowledge` directory at the project root and place your PDF contract guidelines there.

## Usage

Run the classification tool:
```
python contract_risk_classifier.py
```

Follow the prompts to:
- Enter the path to your CSV file
- Select which text columns to analyze
- Save the resulting CSV with Yes/No values

## How It Works

The system:
1. Reads PDF guidelines into a knowledge base
2. Processes each descriptive text field in the input CSV
3. For each text entry, the LLM agent:
   - Queries the knowledge base for relevant guidelines
   - Determines if the text adheres to those guidelines
   - Returns Yes (low risk) or No (higher risk)
4. The processed CSV can then be used to train a machine learning model for risk classification

## Example

Input CSV structure:
```
id,contract_type,description,terms,value
1,Service,"Long descriptive text about contract scope...",More detailed terms...,50000
```

After processing, the descriptive columns will be replaced with Yes/No values:
```
id,contract_type,description,terms,value
1,Service,Yes,No,50000
```

These Yes/No values indicate whether the contract text adheres to the guidelines and best practices in the PDF files, which can then be used for risk classification.

## License

MIT 