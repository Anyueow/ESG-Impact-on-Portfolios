import pandas as pd
import os

def clean_and_merge_data(price_changes_file, financials_file, esg_file):
    """
    Clean and merge price changes data with financial and ESG data
    
    Parameters:
    -----------
    price_changes_file : str
        Path to the price changes CSV file
    financials_file : str
        Path to the financials CSV file
    esg_file : str
        Path to the ESG data CSV file
    
    Returns:
    --------
    pandas.DataFrame
        Cleaned and merged DataFrame with price changes, financial, and ESG data
    """
    # Read the data files
    price_changes_df = pd.read_csv(price_changes_file)
    financials_df = pd.read_csv(financials_file)
    esg_df = pd.read_csv(esg_file)
    
    # Define columns to remove from financials
    columns_to_remove = [
        '52 Week High', '52 Week Low', '52_Week_High', '52_Week_Low',
        'Price/Book', 'Price to Book', 'P/B',
        'Price/Sales', 'Price to Sales', 'P/S',
        'SEC Filings', 'SEC filings', 'SEC_Filings',
        'Industry', 'industry'  # Remove industry column from financials
    ]
    
    # Remove specified columns if they exist
    financials_df = financials_df.drop(columns=[col for col in columns_to_remove if col in financials_df.columns])
    
    # Ensure Ticker column exists and is in the same format
    if 'Ticker' not in financials_df.columns and 'Symbol' in financials_df.columns:
        financials_df = financials_df.rename(columns={'Symbol': 'Ticker'})
    
    # Rename ESG columns to match our expected format
    esg_df = esg_df.rename(columns={
        'Symbol': 'Ticker',
        'GICS Sector': 'GIS Sector',
        'GICS Sub-Industry': 'Sub Industry'
    })
    
    # Remove 'Full Name' and 'marketCap' columns from ESG data
    if 'Full Name' in esg_df.columns or 'marketCap' in esg_df.columns:
        esg_df = esg_df.drop(columns=['Full Name', 'marketCap'])
    
    # Clean data - remove rows with null values
    price_changes_df = price_changes_df.dropna()
    financials_df = financials_df.dropna()
    esg_df = esg_df.dropna()
    
    # Convert numeric columns to appropriate types
    numeric_columns = price_changes_df.columns[1:]  # All columns except Ticker
    for col in numeric_columns:
        price_changes_df[col] = pd.to_numeric(price_changes_df[col], errors='coerce')
    
    # First merge price changes with financials
    merged_df = pd.merge(
        price_changes_df,
        financials_df,
        on='Ticker',
        how='inner'  # Only keep tickers that exist in both datasets
    )
    
    # Then merge with ESG data
    merged_df = pd.merge(
        merged_df,
        esg_df,  # Now keeping all ESG columns except 'Full Name' and 'marketCap'
        on='Ticker',
        how='inner'  # Only keep tickers that exist in all datasets
    )
    
    # Remove any remaining null values after merge
    merged_df = merged_df.dropna()
    
    return merged_df

def main():
    # File paths
    price_changes_file = 'price_changes.csv'
    financials_file = 'financials.csv'
    esg_file = 'sp500_esg_data.csv'
    
    # Check if files exist
    if not os.path.exists(price_changes_file):
        print(f"Error: {price_changes_file} not found. Please run price_data_processor.py first.")
        return
    
    if not os.path.exists(financials_file):
        print(f"Error: {financials_file} not found.")
        return
        
    if not os.path.exists(esg_file):
        print(f"Error: {esg_file} not found.")
        return
    
    # Clean and merge the data
    merged_df = clean_and_merge_data(price_changes_file, financials_file, esg_file)
    
    # Save the merged data
    output_file = 'merged_data.csv'
    merged_df.to_csv(output_file, index=False)
    print(f"Merged data saved to {output_file}")
    
    # Display information about the merged dataset
    print("\nMerged Dataset Information:")
    print(f"Number of companies: {len(merged_df)}")
    print("\nColumns in the merged dataset:")
    print(merged_df.columns.tolist())
    print("\nSample of merged data:")
    print(merged_df.head())
    
    # Display data quality information
    print("\nData Quality Information:")
    print(f"Number of rows with null values: {merged_df.isnull().sum().sum()}")
    print("\nNumber of null values per column:")
    print(merged_df.isnull().sum())

if __name__ == "__main__":
    main() 