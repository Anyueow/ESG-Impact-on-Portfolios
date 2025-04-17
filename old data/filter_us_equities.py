import pandas as pd

# Read the Excel file
file_name = 'ESG Data With Company Performance.xlsx'
df = pd.read_excel(file_name, sheet_name=0)

# Filter for US equities
us_equities = df[df['Equity'].str.contains('US Equity', na=False)]

# Export US equities to CSV
us_equities[['Company Name', 'Ticker']].to_csv('us_ticker_symbols.csv', index=False)
print(f"Found {len(us_equities)} US equities")
print(f"Successfully exported US ticker symbols to us_ticker_symbols.csv")

# Display first few US equities
print("\nFirst few US equities:")
print(us_equities[['Company Name', 'Ticker']].head()) 