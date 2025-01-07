from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.optimize import minimize
import pickle

app = Flask(__name__)

##### Credit Risk Analysis #####
# Load models
with open('decision_tree_model.pkl', 'rb') as file:
    decision_tree_model = pickle.load(file)

with open('knn_model.pkl', 'rb') as file:
    knn_model = pickle.load(file)

with open('random_forest_model.pkl', 'rb') as file:
    random_forest_model = pickle.load(file)

####################### Ro-bo advisors #######################
#### I have pre-calculated the following data and export as csv files ####
# annual returns
annual_df = pd.read_csv('annual_returns.csv')
annual_return_dict = pd.Series(annual_df['Annual Return'].values, index=annual_df['Ticker']).to_dict()

# covariance matrix
cov_matrix = pd.read_csv('cov_matrix.csv', index_col=0)

# stock prices
stock_prices_df = pd.read_csv('stock_prices.csv')
stock_prices = pd.Series(stock_prices_df['Stock Price'].values, index=stock_prices_df['Ticker']).to_dict()

##############################################
@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    if request.method == 'POST':
        # Retrieve form data
        loan_duration = float(request.form['loan_duration'])
        loan_amount = float(request.form['loan_amount'])
        installment_percent = float(request.form['installment_percent'])
        age = float(request.form['age'])
        existing_credits_count = float(request.form['existing_credits_count'])
        model_type = request.form['model_type']

        # Combine input data into a numpy array
        input_data = np.array([[loan_duration, loan_amount, installment_percent, age, existing_credits_count]])

        # Select and apply the chosen model
        if model_type == 'decision_tree':
            prediction = decision_tree_model.predict(input_data)
        elif model_type == 'knn':
            prediction = knn_model.predict(input_data)
        elif model_type == 'random_forest':
            prediction = random_forest_model.predict(input_data)

        # Convert prediction to readable text
        prediction_text = 'Risky' if prediction[0] == 1 else 'Not Risky'

    return render_template('index.html', prediction_text=prediction_text)

@app.route('/robo', methods=['GET', 'POST'])
def robo_advisor():
    if request.method == 'POST':
        try:
            # Get user inputs
            initial = float(request.form['initial_investment'])
            duration = int(request.form['duration'])
            target = float(request.form['target_return'])

            # Calculate required annualized return from the inputs
            required_return = ((target / initial) ** (1 / duration)) - 1

            # Cost
            def integer_shares_cost(weights, stock_prices, initial):
                total_cost = 0
                for ticker, w in zip(stock_prices.keys(), weights):
                    dollars_allocated = w * initial
                    price = stock_prices[ticker]
                    quantity = int(dollars_allocated // price) 
                    total_cost += (quantity * price)
                return total_cost

            def portfolio_performance(weights, annual_returns, cov_matrix):
                portfolio_return = np.dot(weights, annual_returns)
                portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                return portfolio_return, portfolio_risk

            def generate_portfolios(annual_return_dict, cov_matrix, num_portfolios=10000):
                results = []
                tickers = list(annual_return_dict.keys())
                annual_returns = np.array(list(annual_return_dict.values()))

                for _ in range(num_portfolios):
                    weights = np.random.random(len(tickers))
                    weights /= np.sum(weights)

                    portfolio_return, portfolio_risk = portfolio_performance(
                        weights, annual_returns, cov_matrix
                    )
                    sharpe_ratio = (
                        portfolio_return / portfolio_risk
                        if portfolio_risk != 0
                        else 0
                    )

                    results.append((portfolio_return, portfolio_risk, sharpe_ratio, weights))

                results_df = pd.DataFrame(
                    results, columns=['Return', 'Risk', 'Sharpe Ratio', 'Weights']
                )
                return results_df

            # Generate random portfolios
            portfolios_df = generate_portfolios(annual_return_dict, cov_matrix)
            
            def filter_portfolios(portfolios_df, required_return, stock_prices, initial):
                valid_portfolios = []

                for _, row in portfolios_df.iterrows():
                    weights = row['Weights'] 

                    total_cost = integer_shares_cost(weights, stock_prices, initial)

                    # Ensure portfolio satisfies both return and cost constraints
                    if row['Return'] >= required_return and total_cost <= initial:
                        portfolio_data = dict(row)
                        portfolio_data['Total Cost'] = total_cost
                        valid_portfolios.append(portfolio_data)

                filtered_df = pd.DataFrame(valid_portfolios)
                if filtered_df.empty:
                    print("No portfolios satisfy the constraints.")
                    return pd.DataFrame()
                return filtered_df.sort_values(by='Sharpe Ratio', ascending=False).head(5)

            top_5_portfolios = filter_portfolios(
                portfolios_df, required_return, stock_prices, initial
            )

            if top_5_portfolios.empty:
                return render_template(
                    "robo.html", 
                    error_msg="No portfolio satisfies your constraints. Please increase duration or decrease target return."
                )

            def calculate_quantities(top_5_portfolios, stock_prices, initial):
                portfolio_details = []

                for i, row in top_5_portfolios.iterrows():
                    portfolio_weights = row['Weights']
                    
                    weighted_stocks = sorted(
                        zip(stock_prices.keys(), portfolio_weights),
                        key=lambda x: x[1],
                        reverse=True
                    )

                    leftover = initial
                    stock_quantities = {}

                    for stock, weight in weighted_stocks:
                        if weight > 0:
                            stock_price = stock_prices[stock]
                            allocation = leftover * weight
                            shares = int(allocation // stock_price)
                            cost = shares * stock_price
                            leftover -= cost
                            if shares > 0:
                                stock_quantities[stock] = shares

                    purchase_possible = True
                    while purchase_possible:
                        purchase_possible = False
                        for stock, weight in weighted_stocks:
                            price = stock_prices[stock]
                            if leftover >= price > 0:  
                                leftover -= price
                                stock_quantities[stock] = stock_quantities.get(stock, 0) + 1
                                purchase_possible = True

                    total_cost = initial - leftover

                    portfolio_details.append({
                        'Portfolio': len(portfolio_details) + 1,
                        'Stocks': stock_quantities,
                        'Total Cost': total_cost,
                        'Remaining Budget': leftover,
                        'Return': float(row['Return']),
                        'Risk': float(row['Risk']),
                        'Sharpe Ratio': float(row['Sharpe Ratio'])
                    })

                return portfolio_details

            portfolio_details = calculate_quantities(top_5_portfolios, stock_prices, initial)
            
            # ouput results
            return render_template(
                'robo.html',
                portfolios=portfolio_details,
                initial=initial,
                duration=duration,
                target=target
            )

        except Exception as e:
            flash(f"An error occurred: {e}", "danger")
            return redirect(url_for('robo_advisor'))
            
    return render_template('robo.html', portfolios=None)

@app.route('/analysis', methods=['GET'])
def analysis():
    return render_template('analysis.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
