import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output
from sklearn.metrics import confusion_matrix
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load datasets
df = pd.read_csv('loan_data_preprocessed.csv')
results_df = pd.read_csv('model_tuning_logistic_regression_results.csv')

# Load the trained Logistic Regression model
with open('tuned_logistic_regression.pkl', 'rb') as file:
    model = pickle.load(file)

# Define features and target globally
X = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
y = df['Loan_Status']
feature_columns = X.columns

# Fit StandardScaler specifically for numerical columns
numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Total_Income', 'EMI']
scaler = StandardScaler()
scaler.fit(X[numerical_cols])

# Function to calculate EMI
def calculate_emi(loan_amount, loan_term, monthly_rate=0.10/12):
    if pd.isna(loan_amount) or pd.isna(loan_term) or loan_amount == 0 or loan_term == 0:
        return 0
    n = loan_term
    r = monthly_rate
    emi = (loan_amount * r * (1 + r)**n) / ((1 + r)**n - 1)
    return emi

# Initialize Dash app with suppress_callback_exceptions
app = Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.4/css/all.min.css',
    'https://codepen.io/chriddyp/pen/bWLwgP.css'
])

# Dark theme styles
dark_theme = {
    'background': '#1C2526',  # Dark charcoal
    'card': '#2E3B3E',       # Dark slate
    'accent': '#4A919E',      # Muted teal
    'text': '#E0E0E0',       # Light gray
    'input_bg': '#000000',   # Black for inputs
    'input_text': '#FFFFFF'  # White for input text
}

# Layout
app.layout = html.Div(style={
    'backgroundColor': dark_theme['background'],
    'color': dark_theme['text'],
    'padding': '20px',
    'minHeight': '100vh',
    'fontFamily': 'Arial, sans-serif'
}, children=[
    html.H1([
        html.I(className='fas fa-tachometer-alt header-icon', style={'color': dark_theme['accent']}),
        "Loan Approval Prediction Dashboard"
    ], style={
        'textAlign': 'center',
        'color': dark_theme['accent'],
        'marginBottom': '20px'
    }),
    
    # Navigation Tabs
    dcc.Tabs(id='tabs', value='predict', children=[
        dcc.Tab(label="Predict", value='predict', style={
            'backgroundColor': dark_theme['card'],
            'color': dark_theme['text'],
            'padding': '10px 20px'
        }, selected_style={
            'backgroundColor': dark_theme['accent'],
            'color': '#FFFFFF',
            'padding': '10px 20px'
        }),
        dcc.Tab(label="Data Visualization", value='visualization', style={
            'backgroundColor': dark_theme['card'],
            'color': dark_theme['text'],
            'padding': '10px 20px'
        }, selected_style={
            'backgroundColor': dark_theme['accent'],
            'color': '#FFFFFF',
            'padding': '10px 20px'
        }),
        dcc.Tab(label=" Model Performance", value='performance', style={
            'backgroundColor': dark_theme['card'],
            'color': dark_theme['text'],
            'padding': '10px 20px'
        }, selected_style={
            'backgroundColor': dark_theme['accent'],
            'color': '#FFFFFF',
            'padding': '10px 20px'
        })
    ]),
    
    # Tab Content
    html.Div(id='tabs-content', style={
        'padding': '15px',
        'backgroundColor': dark_theme['card'],
        'borderRadius': '10px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.3)'
    })
])

# Callback to render tab content
@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'predict':
        return html.Div([
            html.H3([
                html.I(className='fas fa-money-check-alt header-icon', style={'color': dark_theme['accent']}),
                "Predict Loan Approval"
            ], style={'color': dark_theme['accent'], 'marginBottom': '15px'}),
            html.Div([
                # Left column
                html.Div([
                    html.Label([
                        html.I(className='fas fa-dollar-sign icon', style={'color': dark_theme['text']}),
                        "Applicant Income (M):"
                    ], style={'marginBottom': '5px'}),
                    dcc.Input(id='applicant-income', type='number', value=5000, style={
                        'width': '100%',
                        'backgroundColor': dark_theme['input_bg'],
                        'color': dark_theme['input_text'],
                        'border': f'1px solid {dark_theme["accent"]}',
                        'padding': '5px',
                        'marginBottom': '10px'
                    }),
                    html.Label([
                        html.I(className='fas fa-dollar-sign icon', style={'color': dark_theme['text']}),
                        "Coapplicant Income (M):"
                    ], style={'marginBottom': '5px'}),
                    dcc.Input(id='coapplicant-income', type='number', value=0, style={
                        'width': '100%',
                        'backgroundColor': dark_theme['input_bg'],
                        'color': dark_theme['input_text'],
                        'border': f'1px solid {dark_theme["accent"]}',
                        'padding': '5px',
                        'marginBottom': '10px'
                    }),
                    html.Label([
                        html.I(className='fas fa-money-bill icon', style={'color': dark_theme['text']}),
                        "Loan Amount (M):"
                    ], style={'marginBottom': '5px'}),
                    dcc.Input(id='loan-amount', type='number', value=150, style={
                        'width': '100%',
                        'backgroundColor': dark_theme['input_bg'],
                        'color': dark_theme['input_text'],
                        'border': f'1px solid {dark_theme["accent"]}',
                        'padding': '5px',
                        'marginBottom': '10px'
                    }),
                    html.Label([
                        html.I(className='fas fa-calendar-alt icon', style={'color': dark_theme['text']}),
                        "Loan Amount Term (Months):"
                    ], style={'marginBottom': '5px'}),
                    dcc.Input(id='loan-term', type='number', value=360, style={
                        'width': '100%',
                        'backgroundColor': dark_theme['input_bg'],
                        'color': dark_theme['input_text'],
                        'border': f'1px solid {dark_theme["accent"]}',
                        'padding': '5px',
                        'marginBottom': '10px'
                    }),
                    html.Label([
                        html.I(className='fas fa-users icon', style={'color': dark_theme['text']}),
                        "Dependents:"
                    ], style={'marginBottom': '5px'}),
                    dcc.Dropdown(id='dependents', options=[{'label': str(i), 'value': i} for i in [0, 1, 2, 3]], value=0, style={
                        'width': '100%',
                        'backgroundColor': dark_theme['input_bg'],
                        'color': dark_theme['input_text'],
                        'marginBottom': '10px'
                    })
                ], style={'width': '48%', 'marginRight': '2%'}),
                
                # Right column
                html.Div([
                    html.Label([
                        html.I(className='fas fa-user icon', style={'color': dark_theme['text']}),
                        "Gender:"
                    ], style={'marginBottom': '5px'}),
                    dcc.Dropdown(id='gender', options=[{'label': 'Male', 'value': 'Male'}, {'label': 'Female', 'value': 'Female'}], value='Male', style={
                        'width': '100%',
                        'backgroundColor': dark_theme['input_bg'],
                        'color': dark_theme['input_text'],
                        'marginBottom': '10px'
                    }),
                    html.Label([
                        html.I(className='fas fa-ring icon', style={'color': dark_theme['text']}),
                        "Married:"
                    ], style={'marginBottom': '5px'}),
                    dcc.Dropdown(id='married', options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}], value='No', style={
                        'width': '100%',
                        'backgroundColor': dark_theme['input_bg'],
                        'color': dark_theme['input_text'],
                        'marginBottom': '10px'
                    }),
                    html.Label([
                        html.I(className='fas fa-graduation-cap icon', style={'color': dark_theme['text']}),
                        "Education:"
                    ], style={'marginBottom': '5px'}),
                    dcc.Dropdown(id='education', options=[{'label': 'Graduate', 'value': 'Graduate'}, {'label': 'Not Graduate', 'value': 'Not Graduate'}], value='Graduate', style={
                        'width': '100%',
                        'backgroundColor': dark_theme['input_bg'],
                        'color': dark_theme['input_text'],
                        'marginBottom': '10px'
                    }),
                    html.Label([
                        html.I(className='fas fa-briefcase icon', style={'color': dark_theme['text']}),
                        "Self Employed:"
                    ], style={'marginBottom': '5px'}),
                    dcc.Dropdown(id='self-employed', options=[{'label': 'Yes', 'value': 'Yes'}, {'label': 'No', 'value': 'No'}], value='No', style={
                        'width': '100%',
                        'backgroundColor': dark_theme['input_bg'],
                        'color': dark_theme['input_text'],
                        'marginBottom': '10px'
                    }),
                    html.Label([
                        html.I(className='fas fa-credit-card icon', style={'color': dark_theme['text']}),
                        "Credit History:"
                    ], style={'marginBottom': '5px'}),
                    dcc.Dropdown(id='credit-history', options=[{'label': 'Good (1)', 'value': 1}, {'label': 'Bad (0)', 'value': 0}], value=1, style={
                        'width': '100%',
                        'backgroundColor': dark_theme['input_bg'],
                        'color': dark_theme['input_text'],
                        'marginBottom': '10px'
                    }),
                    html.Label([
                        html.I(className='fas fa-home icon', style={'color': dark_theme['text']}),
                        "Property Area:"
                    ], style={'marginBottom': '5px'}),
                    dcc.Dropdown(id='property-area', options=[{'label': area, 'value': area} for area in ['Urban', 'Semiurban', 'Rural']], value='Urban', style={
                        'width': '100%',
                        'backgroundColor': dark_theme['input_bg'],
                        'color': dark_theme['input_text'],
                        'marginBottom': '10px'
                    })
                ], style={'width': '48%'})
            ], style={'display': 'flex', 'justifyContent': 'space-between', 'marginBottom': '15px'}),
            html.Button([
                html.I(className='fas fa-play button-icon', style={'color': dark_theme['background']}),
                "Predict"
            ], id='predict-button', n_clicks=0, style={
                'backgroundColor': dark_theme['accent'],
                'color': dark_theme['background'],
                'padding': '10px 20px',
                'borderRadius': '5px',
                'border': 'none',
                'cursor': 'pointer',
                'width': '100%',
                'display': 'flex',
                'alignItems': 'center',
                'justifyContent': 'center'
            }),
            html.Div(id='prediction-output', style={'marginTop': '15px', 'fontSize': '18px'})
        ])
    elif tab == 'visualization':
        return html.Div([
            html.H3([
                html.I(className='fas fa-filter header-icon', style={'color': dark_theme['accent']}),
                "Filter Visualizations"
            ], style={'color': dark_theme['accent'], 'marginBottom': '15px'}),
            html.Label([
                html.I(className='fas fa-home icon', style={'color': dark_theme['text']}),
                "Filter by Property Area:"
            ], style={'marginBottom': '5px'}),
            dcc.Dropdown(
                id='property-area-dropdown',
                options=[{'label': 'All', 'value': 'All'}] + [{'label': str(area), 'value': area} for area in ['Urban', 'Semiurban', 'Rural']],
                value='All',
                style={
                    'width': '100%',
                    'backgroundColor': dark_theme['input_bg'],
                    'color': dark_theme['input_text'],
                    'marginBottom': '15px'
                }
            ),
            html.Label([
                html.I(className='fas fa-dollar-sign icon', style={'color': dark_theme['text']}),
                "Applicant Income Range:"
            ], style={'marginBottom': '5px'}),
            html.Div([
                dcc.RangeSlider(
                    id='income-slider',
                    min=df['ApplicantIncome'].min(),
                    max=df['ApplicantIncome'].max(),
                    value=[df['ApplicantIncome'].min(), df['ApplicantIncome'].max()],
                    marks={int(i): str(int(i)) for i in np.linspace(df['ApplicantIncome'].min(), df['ApplicantIncome'].max(), 5)},
                    step=0.1,
                    tooltip={'placement': 'bottom', 'always_visible': True}
                )
            ], style={'marginBottom': '15px'}),
            html.Div([
                html.Div([
                    dcc.Graph(id='loan-status-pie', style={'height': '400px'}),
                    dcc.Graph(id='income-loan-scatter', style={'height': '400px'})
                ], style={'width': '48%', 'marginRight': '2%'}),
                html.Div([
                    dcc.Graph(id='credit-history-bar', style={'height': '400px'}),
                    dcc.Graph(id='feature-importance-bar', style={'height': '400px'})
                ], style={'width': '48%'}),
                html.Div([
                    dcc.Graph(id='confusion-matrix-heatmap', style={'height': '400px'}),
                    dcc.Graph(id='property-approval-bar', style={'height': '400px'})
                ], style={'width': '100%', 'marginTop': '20px'})
            ], style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'justifyContent': 'space-between'
            }),
            html.Div(id='scatter-details', style={
                'marginTop': '20px',
                'padding': '15px',
                'backgroundColor': dark_theme['card'],
                'borderRadius': '10px',
                'boxShadow': '0 2px 4px rgba(0,0,0,0.3)'
            })
        ])
    elif tab == 'performance':
        return html.Div([
            html.H3([
                html.I(className='fas fa-chart-line header-icon', style={'color': dark_theme['accent']}),
                "Model Performance (Logistic Regression)"
            ], style={'color': dark_theme['accent'], 'marginBottom': '15px'}),
            html.P(f"Test Accuracy: {results_df['Test Accuracy'][0]:.4f}"),
            html.P(f"Precision: {results_df['Precision'][0]:.4f}"),
            html.P(f"Recall: {results_df['Recall'][0]:.4f}"),
            html.P(f"F1 Score: {results_df['F1 Score'][0]:.4f}"),
            html.P(f"ROC AUC: {results_df['ROC AUC'][0]:.4f}")
        ])

# Callback for prediction
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    [
        Input('applicant-income', 'value'),
        Input('coapplicant-income', 'value'),
        Input('loan-amount', 'value'),
        Input('loan-term', 'value'),
        Input('dependents', 'value'),
        Input('gender', 'value'),
        Input('married', 'value'),
        Input('education', 'value'),
        Input('self-employed', 'value'),
        Input('credit-history', 'value'),
        Input('property-area', 'value')
    ]
)
def predict_loan(n_clicks, applicant_income, coapplicant_income, loan_amount, loan_term, dependents, gender, married, education, self_employed, credit_history, property_area):
    if n_clicks == 0:
        return "Enter details and click Predict to see the result."
    
    # Validate inputs
    if any(x is None for x in [applicant_income, coapplicant_income, loan_amount, loan_term, dependents, gender, married, education, self_employed, credit_history, property_area]):
        return html.Div("Please fill in all fields.", style={'color': '#ff4d4d'})
    
    # Create input dataframe
    input_data = {
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Dependents': dependents,
        'Credit_History': credit_history,
        'Gender_Male': 1 if gender == 'Male' else 0,
        'Married_Yes': 1 if married == 'Yes' else 0,
        'Education_Not Graduate': 1 if education == 'Not Graduate' else 0,
        'Self_Employed_Yes': 1 if self_employed == 'Yes' else 0,
        'Property_Area_Semiurban': 1 if property_area == 'Semiurban' else 0,
        'Property_Area_Urban': 1 if property_area == 'Urban' else 0
    }
    
    # Calculate derived features
    input_data['Total_Income'] = applicant_income + coapplicant_income
    input_data['EMI'] = calculate_emi(loan_amount, loan_term)
    
    # Create DataFrame with all model features
    input_df = pd.DataFrame([input_data])
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)
    
    # Scale numerical features
    input_df[numerical_cols] = scaler.transform(input_df[numerical_cols])
    
    # Predict
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    # Format output
    result = "Approved" if prediction == 1 else "Not Approved"
    return html.Div([
        html.H4([
            html.I(className='fas fa-check-circle header-icon', style={'color': dark_theme['accent']}),
            "Prediction Result"
        ], style={'color': dark_theme['accent']}),
        html.P(f"Loan Status: {result}"),
        html.P(f"Approval Probability: {probability:.2%}")
    ])

# Callback for updating graphs based on filters
@app.callback(
    [
        Output('loan-status-pie', 'figure'),
        Output('income-loan-scatter', 'figure'),
        Output('credit-history-bar', 'figure'),
        Output('feature-importance-bar', 'figure'),
        Output('confusion-matrix-heatmap', 'figure'),
        Output('property-approval-bar', 'figure'),
        Output('scatter-details', 'children')
    ],
    [
        Input('property-area-dropdown', 'value'),
        Input('income-slider', 'value'),
        Input('income-loan-scatter', 'clickData')
    ]
)
def update_graphs(property_area, income_range, click_data):
    # Filter data based on inputs
    filtered_df = df.copy()
    if property_area != 'All':
        if property_area == 'Urban':
            filtered_df = filtered_df[filtered_df['Property_Area_Urban'] == 1]
        elif property_area == 'Semiurban':
            filtered_df = filtered_df[filtered_df['Property_Area_Semiurban'] == 1]
        else:  # Rural
            filtered_df = filtered_df[(filtered_df['Property_Area_Urban'] == 0) & (filtered_df['Property_Area_Semiurban'] == 0)]
    filtered_df = filtered_df[
        (filtered_df['ApplicantIncome'] >= income_range[0]) & 
        (filtered_df['ApplicantIncome'] <= income_range[1])
    ]
    
    # Handle empty filtered dataframe
    if filtered_df.empty:
        empty_fig = go.Figure().update_layout(
            paper_bgcolor=dark_theme['card'],
            plot_bgcolor=dark_theme['card'],
            font_color=dark_theme['text'],
            title_font_color=dark_theme['accent'],
            margin=dict(t=50, b=50, l=20, r=20),
            title="No data available for selected filters"
        )
        details = "No data available for selected filters."
        return empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, empty_fig, details
    
    # 1. Loan Status Pie Chart
    loan_status_counts = filtered_df['Loan_Status'].value_counts()
    pie_fig = px.pie(
        values=loan_status_counts.values,
        names=['Not Approved', 'Approved'] if loan_status_counts.index[0] == 0 else ['Approved', 'Not Approved'],
        title='<i class="fas fa-chart-pie icon"></i> Loan Status Distribution',
        color_discrete_sequence=['#ff4d4d', '#00cc66']
    )
    pie_fig.update_layout(
        paper_bgcolor=dark_theme['card'],
        plot_bgcolor=dark_theme['card'],
        font_color=dark_theme['text'],
        title_font_color=dark_theme['accent'],
        margin=dict(t=50, b=50, l=20, r=20)
    )
    
    # 2. Applicant Income vs. Loan Amount Scatter
    scatter_fig = px.scatter(
        filtered_df,
        x='ApplicantIncome',
        y='LoanAmount',
        color='Loan_Status',
        title='<i class="fas fa-chart-scatter icon"></i> Applicant Income vs. Loan Amount',
        color_continuous_scale=['#ff4d4d', '#00cc66'],
        labels={'Loan_Status': 'Loan Status', 'ApplicantIncome': 'Applicant Income', 'LoanAmount': 'Loan Amount'}
    )
    scatter_fig.update_layout(
        paper_bgcolor=dark_theme['card'],
        plot_bgcolor=dark_theme['card'],
        font_color=dark_theme['text'],
        title_font_color=dark_theme['accent'],
        margin=dict(t=50, b=50, l=20, r=20)
    )
    
    # 3. Credit History vs. Loan Status
    credit_data = filtered_df.groupby(['Credit_History', 'Loan_Status']).size().unstack().fillna(0)
    bar_fig = go.Figure(data=[
        go.Bar(name='Not Approved', x=credit_data.index, y=credit_data[0], marker_color='#ff4d4d'),
        go.Bar(name='Approved', x=credit_data.index, y=credit_data[1], marker_color='#00cc66')
    ])
    bar_fig.update_layout(
        barmode='group',
        title='<i class="fas fa-chart-bar icon"></i> Credit History vs. Loan Status',
        paper_bgcolor=dark_theme['card'],
        plot_bgcolor=dark_theme['card'],
        font_color=dark_theme['text'],
        title_font_color=dark_theme['accent'],
        margin=dict(t=50, b=50, l=20, r=20)
    )
    
    # 4. Feature Importance (Logistic Regression Coefficients)
    feature_importance = pd.Series(model.coef_[0], index=feature_columns).sort_values(key=abs, ascending=False)
    importance_fig = px.bar(
        x=feature_importance.values,
        y=feature_importance.index,
        title='<i class="fas fa-chart-bar icon"></i> Feature Importance (Logistic Regression Coefficients)',
        orientation='h',
        color_discrete_sequence=['#00cc66']
    )
    importance_fig.update_layout(
        paper_bgcolor=dark_theme['card'],
        plot_bgcolor=dark_theme['card'],
        font_color=dark_theme['text'],
        title_font_color=dark_theme['accent'],
        margin=dict(t=50, b=50, l=20, r=20)
    )
    
    # 5. Confusion Matrix
    filtered_X = filtered_df.drop(['Loan_Status', 'Loan_ID'], axis=1)
    filtered_y = filtered_df['Loan_Status']
    y_pred = model.predict(filtered_X)
    cm = confusion_matrix(filtered_y, y_pred)
    cm_fig = px.imshow(
        cm,
        text_auto=True,
        title='<i class="fas fa-table icon"></i> Confusion Matrix',
        color_continuous_scale='Blues',
        labels=dict(x='Predicted', y='Actual')
    )
    cm_fig.update_layout(
        paper_bgcolor=dark_theme['card'],
        plot_bgcolor=dark_theme['card'],
        font_color=dark_theme['text'],
        title_font_color=dark_theme['accent'],
        margin=dict(t=50, b=50, l=20, r=20)
    )
    
    # 6. Loan Approval Rate by Property Area
    def get_property_area(row):
        if row['Property_Area_Urban'] == 1:
            return 'Urban'
        elif row['Property_Area_Semiurban'] == 1:
            return 'Semiurban'
        else:
            return 'Rural'
    
    filtered_df['Property_Area'] = filtered_df.apply(get_property_area, axis=1)
    approval_rate = filtered_df.groupby('Property_Area')['Loan_Status'].mean().reindex(['Rural', 'Semiurban', 'Urban'], fill_value=0)
    areas = approval_rate.index.tolist()
    rates = approval_rate.values.tolist()
    
    approval_fig = px.bar(
        x=areas,
        y=rates,
        title='<i class="fas fa-chart-bar icon"></i> Loan Approval Rate by Property Area',
        color_discrete_sequence=['#00cc66']
    )
    approval_fig.update_layout(
        paper_bgcolor=dark_theme['card'],
        plot_bgcolor=dark_theme['card'],
        font_color=dark_theme['text'],
        title_font_color=dark_theme['accent'],
        margin=dict(t=50, b=50, l=20, r=20)
    )
    
    # 7. Scatter plot click details
    details = "Click on a point in the scatter plot to see details."
    if click_data:
        point = click_data['points'][0]
        details = html.Div([
            html.H4([
                html.I(className='fas fa-info-circle header-icon', style={'color': dark_theme['accent']}),
                "Selected Applicant Details"
            ], style={'color': dark_theme['accent']}),
            html.P(f"Applicant Income: {point['x']:.2f}"),
            html.P(f"Loan Amount: {point['y']:.2f}"),
            html.P(f"Loan Status: {'Approved' if point['marker.color'] == 1 else 'Not Approved'}")
        ])
    
    return pie_fig, scatter_fig, bar_fig, importance_fig, cm_fig, approval_fig, details

# Run the app
if __name__ == '__main__':
    app.run(debug=True)