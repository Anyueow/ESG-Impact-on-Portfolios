import json
import os

def create_notebook():
    notebook = {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# ESG Portfolio Analysis Visualizations\n",
                    "This notebook contains visualizations analyzing the relationship between ESG scores and stock performance."
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "import pandas as pd\n",
                    "import numpy as np\n",
                    "import matplotlib.pyplot as plt\n",
                    "import seaborn as sns\n",
                    "import plotly.express as px\n",
                    "import plotly.graph_objects as go\n",
                    "from scipy import stats\n",
                    "from plotly.subplots import make_subplots\n",
                    "\n",
                    "# Set style\n",
                    "plt.style.use('seaborn')\n",
                    "sns.set_palette('husl')\n",
                    "pd.set_option('display.max_columns', None)"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Read the data\n",
                    "df = pd.read_csv('merged_data.csv')\n",
                    "print(f\"Dataset shape: {df.shape}\")\n",
                    "df.head()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 1. Portfolio Group Creation and Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create ESG portfolio groups\n",
                    "n_groups = 3\n",
                    "df['ESG_Group'] = pd.qcut(df['totalEsg'], q=n_groups, labels=['Low ESG', 'Medium ESG', 'High ESG'])\n",
                    "\n",
                    "# Calculate group statistics\n",
                    "group_stats = df.groupby('ESG_Group').agg({\n",
                    "    'totalEsg': ['mean', 'std', 'count'],\n",
                    "    '1M_Return': ['mean', 'std'],\n",
                    "    '3M_Return': ['mean', 'std'],\n",
                    "    '6M_Return': ['mean', 'std'],\n",
                    "    '12M_Return': ['mean', 'std']\n",
                    "}).round(4)\n",
                    "\n",
                    "group_stats"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 2. Portfolio Performance Comparison"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create subplots\n",
                    "fig = make_subplots(rows=2, cols=2, \n",
                    "                   subplot_titles=('1-Month Returns', '3-Month Returns', \n",
                    "                                 '6-Month Returns', '12-Month Returns'))\n",
                    "\n",
                    "# Add box plots for each time period\n",
                    "time_periods = ['1M_Return', '3M_Return', '6M_Return', '12M_Return']\n",
                    "positions = [(1,1), (1,2), (2,1), (2,2)]\n",
                    "\n",
                    "for period, pos in zip(time_periods, positions):\n",
                    "    fig.add_trace(\n",
                    "        go.Box(\n",
                    "            y=df[period],\n",
                    "            x=df['ESG_Group'],\n",
                    "            name=period,\n",
                    "            boxpoints='all',\n",
                    "            jitter=0.3,\n",
                    "            pointpos=-1.8\n",
                    "        ),\n",
                    "        row=pos[0], col=pos[1]\n",
                    "    )\n",
                    "\n",
                    "fig.update_layout(\n",
                    "    height=800,\n",
                    "    width=1000,\n",
                    "    title_text='Return Distributions by ESG Group',\n",
                    "    showlegend=False\n",
                    ")\n",
                    "fig.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 3. Risk-Adjusted Returns Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Calculate Sharpe ratios (assuming risk-free rate of 0 for simplicity)\n",
                    "def calculate_sharpe(returns, periods=12):\n",
                    "    return np.sqrt(periods) * ((returns - 1).mean() / (returns - 1).std())\n",
                    "\n",
                    "sharpe_ratios = df.groupby('ESG_Group').agg({\n",
                    "    '1M_Return': lambda x: calculate_sharpe(x, 12),\n",
                    "    '3M_Return': lambda x: calculate_sharpe(x, 4),\n",
                    "    '6M_Return': lambda x: calculate_sharpe(x, 2),\n",
                    "    '12M_Return': lambda x: calculate_sharpe(x, 1)\n",
                    "})\n",
                    "\n",
                    "# Plot Sharpe ratios\n",
                    "plt.figure(figsize=(12, 6))\n",
                    "sharpe_ratios.T.plot(kind='bar')\n",
                    "plt.title('Sharpe Ratios by ESG Group and Time Horizon')\n",
                    "plt.ylabel('Sharpe Ratio')\n",
                    "plt.xticks(rotation=45)\n",
                    "plt.legend(title='ESG Group')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 4. Sector Composition Analysis"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Calculate sector composition\n",
                    "sector_composition = pd.crosstab(df['ESG_Group'], df['GIS Sector'], normalize='index') * 100\n",
                    "\n",
                    "# Create heatmap\n",
                    "plt.figure(figsize=(12, 8))\n",
                    "sns.heatmap(sector_composition, annot=True, fmt='.1f', cmap='YlGnBu')\n",
                    "plt.title('Sector Composition by ESG Group (%)')\n",
                    "plt.xlabel('Sector')\n",
                    "plt.ylabel('ESG Group')\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 5. ESG Score Distribution and Performance"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Create scatter plot with regression lines\n",
                    "fig = px.scatter(df, x='totalEsg', y='12M_Return', \n",
                    "                color='ESG_Group',\n",
                    "                trendline='ols',\n",
                    "                title='ESG Score vs 12-Month Returns',\n",
                    "                labels={'totalEsg': 'ESG Score', '12M_Return': '12-Month Return'})\n",
                    "\n",
                    "fig.update_layout(\n",
                    "    height=600,\n",
                    "    width=800,\n",
                    "    showlegend=True\n",
                    ")\n",
                    "fig.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 6. Performance During Market Conditions"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Calculate cumulative returns for each group\n",
                    "time_periods = ['1M_Return', '3M_Return', '6M_Return', '12M_Return']\n",
                    "dates = pd.date_range(end=pd.Timestamp.now(), periods=len(time_periods), freq='M')\n",
                    "\n",
                    "plt.figure(figsize=(12, 6))\n",
                    "for group in df['ESG_Group'].unique():\n",
                    "    group_data = df[df['ESG_Group'] == group]\n",
                    "    cumulative_returns = [group_data[period].mean() for period in time_periods]\n",
                    "    plt.plot(dates, cumulative_returns, label=f'{group} Portfolio', marker='o')\n",
                    "\n",
                    "plt.title('Cumulative Returns by ESG Group Over Time')\n",
                    "plt.xlabel('Date')\n",
                    "plt.ylabel('Cumulative Return')\n",
                    "plt.legend()\n",
                    "plt.grid(True)\n",
                    "plt.tight_layout()\n",
                    "plt.show()"
                ]
            },
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## 7. Statistical Significance Tests"
                ]
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "# Perform ANOVA test for each time period\n",
                    "results = {}\n",
                    "for period in ['1M_Return', '3M_Return', '6M_Return', '12M_Return']:\n",
                    "    groups = [df[df['ESG_Group'] == group][period] for group in df['ESG_Group'].unique()]\n",
                    "    f_stat, p_value = stats.f_oneway(*groups)\n",
                    "    results[period] = {'F-statistic': f_stat, 'p-value': p_value}\n",
                    "\n",
                    "# Display results\n",
                    "pd.DataFrame(results).T.round(4)"
                ]
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }

    # Save the notebook
    with open('ESG_Portfolio_Analysis.ipynb', 'w') as f:
        json.dump(notebook, f, indent=1)

if __name__ == "__main__":
    create_notebook() 