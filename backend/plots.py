"""
Plotting utilities for FinSight AI.

Contains 3 plot functions (revenue/income, balance sheet, cash flow)
and a helper to save matplotlib figures to base64 strings.
"""

import logging
import base64
import traceback
from io import BytesIO
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('dark_background')
import matplotlib.ticker as mtick

from edgar import Company
from edgar.xbrl import XBRLS

logger = logging.getLogger(__name__)


def get_company_filings_data(ticker: str):
    """Fetch company and XBRL data once for reuse across all three plots."""
    c = Company(ticker)
    filings = c.get_filings(form="10-K").latest(5)
    xbrs = XBRLS.from_filings(filings)
    return c, xbrs


def _save_plot_to_base64(fig, ticker: str, chart_type: str) -> str:
    try:
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)
        logger.info(f"Generated {chart_type} plot for {ticker}")
        return image_base64
    except Exception as e:
        logger.error(f"Error saving {chart_type} plot: {e}")
        plt.close(fig)
        raise


def plot_revenue(ticker: str, c, xbrs) -> Optional[str]:
    """Revenue, Gross Profit, and Net Income bar chart with margin overlays. Returns base64 PNG or None."""
    try:
        income_statement = xbrs.statements.income_statement()
        income_df = income_statement.to_dataframe()

        # Revenue: try labelled "Contract Revenue" first, then any Revenues concept
        rev_rows = income_df[income_df.label == "Contract Revenue"]
        if rev_rows.empty:
            rev_rows = income_df[income_df.concept.str.contains("Revenues|Revenue", case=False, na=False)]
        if rev_rows.empty:
            raise ValueError("Could not find Revenue row in income statement")
        revenue = rev_rows[income_statement.periods].iloc[0]

        net_rows = income_df[income_df.concept == "us-gaap_NetIncomeLoss"]
        if net_rows.empty:
            net_rows = income_df[income_df.concept.str.contains("NetIncome", case=False, na=False)]
        net_income = net_rows[income_statement.periods].iloc[0] if not net_rows.empty else pd.Series([0]*len(income_statement.periods), index=income_statement.periods)

        gp_rows = income_df[income_df.concept == "us-gaap_GrossProfit"]
        has_gross_profit = not gp_rows.empty
        gross_profit = gp_rows[income_statement.periods].iloc[0] if has_gross_profit else None

        periods = [pd.to_datetime(period).strftime('FY%y') for period in income_statement.periods][::-1]
        revenue_values = revenue.values[::-1]
        net_income_values = net_income.values[::-1]
        gross_profit_values = gross_profit.values[::-1] if has_gross_profit else None

        df_cols = {'Revenue': revenue_values, 'Net Income': net_income_values}
        if has_gross_profit:
            df_cols['Gross Profit'] = gross_profit_values

        plot_data = pd.DataFrame(df_cols, index=periods) / 1e9

        margins = pd.DataFrame({
            'Net Margin': plot_data['Net Income'] / plot_data['Revenue'] * 100,
            **({"Gross Margin": plot_data['Gross Profit'] / plot_data['Revenue'] * 100} if has_gross_profit else {})
        }, index=periods)

        fig, ax1 = plt.subplots(figsize=(12, 8))
        x = np.arange(len(periods))
        width = 0.25

        if has_gross_profit:
            ax1.bar(x - width, plot_data['Revenue'], width, label='Revenue', color='#3498db', alpha=0.8)
            ax1.bar(x, plot_data['Gross Profit'], width, label='Gross Profit', color='#2ecc71', alpha=0.8)
            ax1.bar(x + width, plot_data['Net Income'], width, label='Net Income', color='#9b59b6', alpha=0.8)
        else:
            ax1.bar(x - width/2, plot_data['Revenue'], width, label='Revenue', color='#3498db', alpha=0.8)
            ax1.bar(x + width/2, plot_data['Net Income'], width, label='Net Income', color='#9b59b6', alpha=0.8)

        ax2 = ax1.twinx()
        if has_gross_profit:
            ax2.plot(x, margins['Gross Margin'], 'o-', color='#2ecc71', linewidth=2, label='Gross Margin %')
        ax2.plot(x, margins['Net Margin'], 's-', color='#9b59b6', linewidth=2, label='Net Margin %')

        ax1.set_xlabel('Fiscal Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Billions USD', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Profit Margin (%)', fontsize=12, fontweight='bold')

        ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:.1f}B'))
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

        ax1.set_xticks(x)
        ax1.set_xticklabels(periods)

        plt.title(f'{c.name} ({ticker}) Financial Performance', fontsize=16, fontweight='bold', pad=20)
        plt.figtext(0.5, 0.01, 'Source: SEC EDGAR via edgartools', ha='center', fontsize=10)

        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, fontsize=10)

        bar_metrics = list(plot_data.columns)
        for i, metric in enumerate(bar_metrics):
            for j, value in enumerate(plot_data[metric]):
                offset = (i - len(bar_metrics)/2 + 0.5) * width
                ax1.annotate(f'${value:.1f}B',
                            xy=(j + offset, value),
                            xytext=(0, 3),
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=8, fontweight='bold')

        for metric in bar_metrics:
            for i in range(1, len(periods)):
                growth = ((plot_data[metric].iloc[i] / plot_data[metric].iloc[i-1]) - 1) * 100
                offset = (list(plot_data.columns).index(metric) - len(bar_metrics)/2 + 0.5) * width
                ax1.annotate(f'{growth:+.1f}%',
                            xy=(i + offset, plot_data[metric].iloc[i]),
                            xytext=(0, -15),
                            textcoords='offset points',
                            ha='center',
                            color='#e74c3c' if growth < 0 else '#27ae60',
                            fontsize=8, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        return _save_plot_to_base64(fig, ticker, "financial")

    except Exception as e:
        logger.error(f"Error creating financial plot for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return None


def plot_balance_sheet(ticker: str, c, xbrs) -> Optional[str]:
    """Assets, Liabilities, and Stockholders Equity bar chart with D/E ratio overlay. Returns base64 PNG or None."""
    try:
        balance_sheet = xbrs.statements.balance_sheet()
        balance_df = balance_sheet.to_dataframe()

        total_assets = balance_df[balance_df.concept == "us-gaap_Assets"][balance_sheet.periods].iloc[0]
        total_liabilities = balance_df[balance_df.concept == "us-gaap_Liabilities"][balance_sheet.periods].iloc[0]
        stockholders_equity = balance_df[balance_df.concept == "us-gaap_StockholdersEquity"][balance_sheet.periods].iloc[0]

        periods = [pd.to_datetime(period).strftime('FY%y') for period in balance_sheet.periods]
        periods = periods[::-1]

        assets_values = total_assets.values[::-1] / 1e9
        liabilities_values = total_liabilities.values[::-1] / 1e9
        equity_values = stockholders_equity.values[::-1] / 1e9

        plot_data = pd.DataFrame({
            'Total Assets': assets_values,
            'Total Liabilities': liabilities_values,
            'Stockholders Equity': equity_values
        }, index=periods)

        debt_to_equity = liabilities_values / equity_values

        fig, ax1 = plt.subplots(figsize=(12, 8))

        x = np.arange(len(periods))
        width = 0.25

        ax1.bar(x - width, plot_data['Total Assets'], width, label='Total Assets', color='#3498db', alpha=0.8)
        ax1.bar(x, plot_data['Total Liabilities'], width, label='Total Liabilities', color='#e74c3c', alpha=0.8)
        ax1.bar(x + width, plot_data['Stockholders Equity'], width, label='Stockholders Equity', color='#2ecc71', alpha=0.8)

        ax2 = ax1.twinx()
        ax2.plot(x, debt_to_equity, 'o-', color='#f39c12', linewidth=2, label='Debt/Equity Ratio')

        ax1.set_xlabel('Fiscal Year', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Billions USD', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Debt/Equity Ratio', fontsize=12, fontweight='bold')

        ax1.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:.1f}B'))

        ax1.set_xticks(x)
        ax1.set_xticklabels(periods)

        plt.title(f'{c.name} ({ticker}) Balance Sheet', fontsize=16, fontweight='bold', pad=20)
        plt.figtext(0.5, 0.01, 'Source: SEC EDGAR via edgartools', ha='center', fontsize=10)

        ax1.grid(axis='y', linestyle='--', alpha=0.7)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', frameon=True, fontsize=10)

        for i, metric in enumerate(['Total Assets', 'Total Liabilities', 'Stockholders Equity']):
            for j, value in enumerate(plot_data[metric]):
                offset = (i - 1) * width
                ax1.annotate(f'${value:.1f}B',
                            xy=(j + offset, value),
                            xytext=(0, 3),
                            textcoords='offset points',
                            ha='center', va='bottom',
                            fontsize=8, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        return _save_plot_to_base64(fig, ticker, "balance sheet")

    except Exception as e:
        logger.error(f"Error creating balance sheet plot for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return None


def plot_cash_flow(ticker: str, c, xbrs) -> Optional[str]:
    """Operating, Investing, Financing, and Free Cash Flow bar chart. Returns base64 PNG or None."""
    try:
        cash_flow = xbrs.statements.cashflow_statement()
        cashflow_df = cash_flow.to_dataframe()

        operating_cf = cashflow_df[cashflow_df.concept == "us-gaap_NetCashProvidedByUsedInOperatingActivities"][cash_flow.periods].iloc[0]
        investing_cf = cashflow_df[cashflow_df.concept == "us-gaap_NetCashProvidedByUsedInInvestingActivities"][cash_flow.periods].iloc[0]
        financing_cf = cashflow_df[cashflow_df.concept == "us-gaap_NetCashProvidedByUsedInFinancingActivities"][cash_flow.periods].iloc[0]

        periods = [pd.to_datetime(period).strftime('FY%y') for period in cash_flow.periods]
        periods = periods[::-1]

        operating_values = operating_cf.values[::-1] / 1e9
        investing_values = investing_cf.values[::-1] / 1e9
        financing_values = financing_cf.values[::-1] / 1e9
        free_cf = operating_values + investing_values

        plot_data = pd.DataFrame({
            'Operating CF': operating_values,
            'Investing CF': investing_values,
            'Financing CF': financing_values,
            'Free Cash Flow': free_cf
        }, index=periods)

        fig, ax = plt.subplots(figsize=(12, 8))

        x = np.arange(len(periods))
        width = 0.2

        ax.bar(x - 1.5*width, plot_data['Operating CF'], width, label='Operating CF', color='#2ecc71', alpha=0.8)
        ax.bar(x - 0.5*width, plot_data['Investing CF'], width, label='Investing CF', color='#e74c3c', alpha=0.8)
        ax.bar(x + 0.5*width, plot_data['Financing CF'], width, label='Financing CF', color='#f39c12', alpha=0.8)
        ax.bar(x + 1.5*width, plot_data['Free Cash Flow'], width, label='Free Cash Flow', color='#9b59b6', alpha=0.8)

        ax.set_xlabel('Fiscal Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Billions USD', fontsize=12, fontweight='bold')

        ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'${x:.1f}B'))

        ax.set_xticks(x)
        ax.set_xticklabels(periods)

        plt.title(f'{c.name} ({ticker}) Cash Flow Statement', fontsize=16, fontweight='bold', pad=20)
        plt.figtext(0.5, 0.01, 'Source: SEC EDGAR via edgartools', ha='center', fontsize=10)

        ax.grid(axis='y', linestyle='--', alpha=0.7)
        ax.legend(loc='upper left', frameon=True, fontsize=10)

        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

        for i, metric in enumerate(['Operating CF', 'Investing CF', 'Financing CF', 'Free Cash Flow']):
            for j, value in enumerate(plot_data[metric]):
                offset = (i - 1.5) * width
                y_offset = 3 if value >= 0 else -15
                va = 'bottom' if value >= 0 else 'top'
                ax.annotate(f'${value:.1f}B', 
                           xy=(j + offset, value), 
                           xytext=(0, y_offset),
                           textcoords='offset points',
                           ha='center', va=va,
                           fontsize=7, fontweight='bold')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        return _save_plot_to_base64(fig, ticker, "cash flow")

    except Exception as e:
        logger.error(f"Error creating cash flow plot for {ticker}: {e}")
        logger.error(traceback.format_exc())
        return None