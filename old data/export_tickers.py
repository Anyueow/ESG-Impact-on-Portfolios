import pandas as pd

# Read the Excel file
file_name = 'ESG Data With Company Performance.xlsx'
df = pd.read_excel(file_name, sheet_name=0)

# Extract ticker symbols and save to CSV
tickers = df[['Company Name', 'Ticker']]
tickers.to_csv('ticker_symbols.csv', index=False)
print(f"Successfully exported {len(tickers)} ticker symbols to ticker_symbols.csv") 