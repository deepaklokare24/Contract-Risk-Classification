import os
import pandas as pd
from dotenv import load_dotenv
from crewai import Agent, Task, Crew, Process, LLM
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource

# Load environment variables
load_dotenv()

# Set up OpenAI API key - make sure it's set in your .env file
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Use relative paths for PDF files in the knowledge directory
# CrewAI expects files to be in a 'knowledge' directory at the project root
pdf_files = [
    "2015_UGC_09-16-15.pdf",
    "3560-1Chapter09.pdf",
    "ContractTypes.pdf",
    "Contracts-for-Contractors.pdf",
    "Guide-Construction-Contracts.pdf",
    "construction-contracting-options-4-pg.pdf"
]

# Create PDF knowledge source with relative paths
pdf_source = PDFKnowledgeSource(
    file_paths=pdf_files,
    chunk_size=1000,
    chunk_overlap=100
)
    
def create_contract_risk_agent():
    """Create a CrewAI agent for contract risk classification."""
    
    # Set up GPT-4o LLM
    llm = LLM(
        model="gpt-4o",
        api_key=OPENAI_API_KEY,
        temperature=0.1
    )
    
    # Create the contract risk classifier agent
    risk_classifier_agent = Agent(
        role="Contract Risk Assessor",
        goal="Classify contract deals as high/medium/low risk based on contract guidelines.",
        backstory="""You are an expert in contract risk assessment with years of experience
        in evaluating contract deals. You have deep knowledge of contract best practices,
        legal requirements, and risk factors. Your job is to analyze descriptive text
        in contracts and determine if they adhere to guidelines and best practices.""",
        verbose=True,
        allow_delegation=False,
        llm=llm,
        knowledge_sources=[pdf_source],
        embedder={
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-large",
                "api_key": OPENAI_API_KEY
            }
        }
    )
    
    return risk_classifier_agent

def analyze_text(agent, text, column_name):
    """Create a task for the agent to analyze a single text entry."""
    
    task = Task(
        description=f"""
        Analyze the following contract text from the '{column_name}' field:
        
        "{text}"
        
        Based on your knowledge of contract guidelines and best practices,
        determine if this text adheres to guidelines and represents low risk.
        
        Respond with only "Yes" if it adheres to guidelines (low risk),
        or "No" if it does not adhere to guidelines (higher risk).
        """,
        expected_output="Yes or No",
        agent=agent
    )
    
    # Create and run a crew for this specific analysis
    crew = Crew(
        agents=[agent],
        tasks=[task],
        verbose=False,
        process=Process.sequential,
        knowledge_sources=[pdf_source],
    )
    
    # Execute the crew
    result = crew.kickoff()
    
    # Clean the result - extract just Yes or No
    result_str = str(result).strip()
    if "yes" in result_str.lower():
        return "Yes"
    elif "no" in result_str.lower():
        return "No"
    else:
        # Default to No if unclear
        print(f"Warning: Unclear result from agent: '{result_str}'. Defaulting to 'No'")
        return "No"

def process_csv_file(csv_path, text_columns):
    """Process a CSV file and classify the contract risks."""
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Create the agent
    risk_agent = create_contract_risk_agent()
    
    # Process each row and column
    for col in text_columns:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in CSV file.")
            continue
            
        print(f"\nProcessing column: {col}")
        
        # Process each text value in the column
        for idx, row in df.iterrows():
            print(f"  Analyzing row {idx+1}/{len(df)}: {col}")
            text_value = row[col]
            
            # Skip empty values
            if pd.isna(text_value) or text_value == "":
                continue
                
            # Analyze the text and get Yes/No result
            result = analyze_text(risk_agent, text_value, col)
            
            # Update the value in the dataframe
            df.at[idx, col] = result
    
    # Return the modified dataframe
    return df

def main():
    """Main function to run the contract risk classification."""
    
    # Example usage
    print("Contract Risk Classification System")
    print("----------------------------------")
    print("This system analyzes contract text and classifies risk based on guidelines.")
    
    # Get input CSV file
    csv_path = input("Enter the path to your CSV file: ")
    
    # Check if file exists
    if not os.path.exists(csv_path):
        print(f"Error: File '{csv_path}' not found.")
        return
    
    try:
        # Load the CSV to display column names
        df = pd.read_csv(csv_path)
        print("\nAvailable columns:")
        for i, col in enumerate(df.columns):
            print(f"{i+1}. {col}")
        
        # Get columns to analyze
        col_input = input("\nEnter the numbers of text columns to analyze (comma-separated): ")
        col_indices = [int(idx.strip()) - 1 for idx in col_input.split(",")]
        text_columns = [df.columns[idx] for idx in col_indices if 0 <= idx < len(df.columns)]
        
        if not text_columns:
            print("No valid columns selected.")
            return
        
        print(f"\nAnalyzing columns: {', '.join(text_columns)}")
        
        # Process the CSV file
        result_df = process_csv_file(csv_path, text_columns)
        
        # Output results
        print("\nClassification Complete!")
        print(result_df)
        
        # Ask if user wants to save results
        save_path = input("\nEnter a file path to save results (or press Enter to skip): ")
        if save_path:
            result_df.to_csv(save_path, index=False)
            print(f"Results saved to {save_path}")
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main() 