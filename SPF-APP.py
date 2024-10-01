import pandas as pd 
import numpy as np
import streamlit as st
from streamlit_option_menu import option_menu
import plotly.express as px
import pickle
import joblib
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
import base64

df1=pd.read_csv(r"D:\Guvi_Data_Science\MDT33\Capstone_Project\Singapore-Resale-Flat\1990-1999\combined.csv",low_memory=False)
df=pd.read_csv(r'D:\Guvi_Data_Science\MDT33\Capstone_Project\Singapore-Resale-Flat\1990-1999\Processed_data.csv',low_memory=False)
df=df.drop(['Unnamed: 0'],axis=1) 

town_mapping = {'ANG MO KIO': 1, 'BEDOK': 2, 'BISHAN': 3, 'BUKIT BATOK': 4, 'BUKIT MERAH': 5, 'BUKIT TIMAH': 6,
                    'CENTRAL AREA': 7, 'CHOA CHU KANG': 8, 'CLEMENTI': 9, 'GEYLANG': 10, 'HOUGANG': 11,
                    'JURONG EAST': 12, 'JURONG WEST': 13, 'KALLANG/WHAMPOA': 14, 'MARINE PARADE': 15, 'QUEENSTOWN': 16,
                    'SENGKANG': 17, 'SERANGOON': 18, 'TAMPINES': 19, 'TOA PAYOH': 20, 'WOODLANDS': 21, 'YISHUN': 22,
                    'LIM CHU KANG': 23, 'SEMBAWANG': 24, 'BUKIT PANJANG': 25, 'PASIR RIS': 26, 'PUNGGOL': 27}

# Define a mapping of flat_type to numbers
category_mapping = {
        '1 ROOM': 1,
        '2 ROOM': 2,
        '3 ROOM': 3,
        '4 ROOM': 4,
        '5 ROOM': 5,
        'EXECUTIVE': 6,
        'MULTI GENERATION': 7
        }

def map_street_to_number(street_name):
            street_mapping = {street: idx + 1 for idx, street in enumerate(streets)}
            return street_mapping.get(street_name)        

# Flat Model
flat_model_mapping = {'IMPROVED': 1, 'NEW GENERATION': 2, 'MODEL A': 3, 'STANDARD': 4, 'SIMPLIFIED': 5,
                        'MODEL A-MAISONETTE': 6, 'APARTMENT': 7, 'MAISONETTE': 8, 'TERRACE': 9, '2-ROOM': 10,
                        'IMPROVED-MAISONETTE': 11,
                        'MULTI GENERATION': 12, 'PREMIUM APARTMENT': 13, 'Improved': 14, 'New Generation': 15,
                        'Model A':
                            16, 'Standard': 17, 'Apartment': 18, 'Simplified': 19, 'Model A-Maisonette': 20,
                        'Maisonette':
                            21, 'Multi Generation': 22, 'Adjoined flat': 23, 'Premium Apartment': 24, 'Terrace': 25,
                        'Improved-Maisonette': 26, 'Premium Maisonette': 27, '2-room': 28, 'Model A2': 29, 'DBSS': 30,
                        'Type S1': 31, 'Type S2': 32, 'Premium Apartment Loft': 33, '3Gen': 34}


def predict_price(town, flat_type_value, block_decimal,selected_street_number, floor_area, flat_model_value, lease_commence_date, remaining_lease, resale_year, resale_month, storey_lower, storey_upper):
    # Load the regression model
        model_path = 'D:/Guvi_Data_Science/MDT33/Capstone_Project/Singapore-Resale-Flat/model.joblib'
        with open(model_path, 'rb') as file:
            RFR_model2 = joblib.load(file)

    # Define the user data as a 2D numpy array
        userdata = np.array([[town, flat_type_value, block_decimal,selected_street_number, floor_area, flat_model_value, lease_commence_date, remaining_lease,resale_year,resale_month, storey_lower, storey_upper]])
        pred = RFR_model2.predict(userdata)
        predict_price = pred[0]
        return predict_price
    

st.set_page_config(page_title= "Singapore-Flat-Price-Prediction",
                   layout= "wide",
                   initial_sidebar_state= "expanded")
def setting_bg():
    # Open the image file and convert it to Base64
    with open("C:/Users/Admin/Downloads/singa.jpg", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()

    # Create a CSS block with the Base64 image as the background
    st.markdown(f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
        }}
        </style>
        """, unsafe_allow_html=True)

# Call the function to set the background
setting_bg()

st.markdown("<h1 style='text-align: center;font-size: 50px; color: lightgreen;'>Analysis of Singapore Flat Resale Prices</h1>", unsafe_allow_html=True)

st.markdown("""
    <style>
    /* Change tab text color and background color */
    div[data-testid="stTabs"] button {
        color: black;
        background-color: lightgreen;
        font-size: 15px; /* Adjust font size */
        border-radius: 25px;
        padding: 30px;
    }

    /* Change active tab background and text color */
    div[data-testid="stTabs"] button[aria-selected="true"] {
        background-color: darkgreen;
        color: black;
        font-weight: bold;
        border-bottom: 2px solid red; /* Optional: Add border to active tab */
    }

    /* Hover effect for the tabs */
    div[data-testid="stTabs"] button:hover {
        background-color: lightblue;
        color: black;
    }
    </style>
""", unsafe_allow_html=True)

tab1,tab2,tab3=st.tabs(['DATA INSIGHTS','EXPLORATARY DATA ANALYSIS (EDA)','MODEL PREDICTION'])

with tab1:
    col1,col2=st.columns(2,gap="small")  
    st.header(":orange[DataFrame and Matrix Insights]")
    st.dataframe(df1.head(5))
    st.header(":orange[Model Performance]")
    data = {
            "Algorithm": ['KNeighborsRegressor', 'DecisionTreeRegressor', 'RandomForestRegressor', 'XGBoostRegressor'],
            "Mean Absolute Error (MAE)": [28496.45, 17650.76, 14015.83, 17217.50],
            "Root Mean Square Error (RMSE)": [1746823341.26, 697200898.068,414063883.56,578963326.88],
            "Mean Squared Error (MSE)": [41795.015, 26404.56, 20348.55,20348.55],
            "R-squared (R2) Score": [93.53,97.41,98.4,97.8]
                        
                        }
    dff = pd.DataFrame(data)
    st.dataframe(dff)
    
    colors = ['#FF9999', '#66B3FF', '#99FF99', '#FFCC99'] 
    # Plot for R-squared (R2) Score
    fig = px.bar(dff, x='R-squared (R2) Score', y='Algorithm', orientation='h', 
            title='R-squared (R2) Score Comparison',
            text='R-squared (R2) Score', labels={'R-squared (R2) Score': 'R-squared (R2) Score', 'Algorithm': 'Algorithm'},
            color='Algorithm',
            color_discrete_sequence=colors)

    # Display the plot in Streamlit
    st.plotly_chart(fig)
    st.markdown(f"## The Selected Algorithm is :green[*RandomForestRegressor*] and its R-squared Score  is   :green[*98%*]")

with tab2:

    col1,col2 =st.columns(2)

    color_sequence = ['#FF5733', '#33FF57', '#3357FF', '#F0E68C'] 

    def plot_horizontal_bar_plotly(df, group_col, value_col, color='skyblue'):
    # Group by the specified column and calculate the mean
        df_grouped = df.groupby(group_col, as_index=False)[value_col].mean()

        # Sort by the mean value in descending order
        df_grouped = df_grouped.sort_values(by=value_col, ascending=False)

        # Create the horizontal bar plot with Plotly
        fig = px.bar(df_grouped, 
                    x=value_col, 
                    y=group_col, 
                    orientation='h', 
                    title=f'Average {value_col} by {group_col}',
                    labels={value_col: f'Average {value_col}', group_col: group_col},
                    text=value_col,  # Add text labels on the bars
                    color=group_col,
                    color_discrete_sequence=px.colors.qualitative.Vivid)  # Use the color parameter
        
        # Update layout for better readability
        fig.update_layout(
            xaxis_title=f'Average {value_col}',
            yaxis_title=group_col,
            title_font_size=16,
            xaxis_title_font_size=14,
            yaxis_title_font_size=14,
            height=400  # Adjust the height of the figure if needed
        )
        
        # Display the plot
        st.plotly_chart(fig)

    #Plotly pie chart
    def plot_pie_chart(df, labels_col, values_col, title):
        fig = px.pie(df, 
                    names=labels_col, 
                    values=values_col, 
                    title=title,
                    color_discrete_sequence=px.colors.qualitative.Plotly,  # Use the "Paired" color palette
                    hole=0.3,  # Optional: Add a hole for a donut chart
                    )
        
        fig.update_traces(textinfo='percent+label')  # Show both percent and label in slices
        
        # Update layout for title font size and chart dimensions
        fig.update_layout(
            title_font_size=16,
            height=400,
            width=600
        )
        
        st.plotly_chart(fig)

    def plot_vertical_bar_plotly(df, group_col, value_col):
        # Group by the specified column and calculate the mean
        df_grouped = df.groupby(group_col, as_index=False)[value_col].mean()

        # Sort by the mean value in descending order
        df_grouped = df_grouped.sort_values(by=value_col, ascending=False)

        # Create the vertical bar plot with Plotly
        fig = px.bar(df_grouped, 
                 x=group_col, 
                 y=value_col, 
                 title=f'Average {value_col} by {group_col}',
                 labels={value_col: f'Average {value_col}', group_col: group_col},
                 text=value_col,  # Add text labels on the bars
                 color=group_col,
                 color_discrete_sequence=px.colors.qualitative.Vivid)  # Use the color parameter

        # Update layout for better readability
        fig.update_layout(
        xaxis_title=group_col,
        yaxis_title=f'Average {value_col}',
        title_font_size=16,
        xaxis_title_font_size=14,
        yaxis_title_font_size=14,
        height=400  # Adjust the height of the figure if needed
    )
    
        # Display the plot
        st.plotly_chart(fig)
    with col1:

        plot_horizontal_bar_plotly(df1, 'town', 'resale_price')
        plot_pie_chart(df1, 'flat_type', 'resale_price', 'flat_type vs Re-sale Price')
               

    with col2:
        plot_horizontal_bar_plotly(df1, 'flat_model', 'resale_price')
        plot_vertical_bar_plotly(df, 'resale_year', 'resale_price')

with tab3:
    
    col1,col2,col3,col4,col5 = st.columns(5)

    with col1:

        town_key = st.selectbox('**Select a town**', list(town_mapping.keys()))
        town = town_mapping[town_key]
        
        flat_type = st.selectbox('**Select Flat Type**', list(category_mapping.keys()))
        flat_type_value = category_mapping[flat_type]

        block_decimal = st.text_input('**Enter the block number**', value=438)

        streets = df1['street_name'].unique()
        selected_street = st.selectbox('Select Street Name:', streets)

        selected_street_number = map_street_to_number(selected_street)
                
        # Define a mapping for letters to decimal values
        #letter_mapping = {chr(ord('A') + i): f'.{i + 1}' for i in range(26)}
        #block_decimal = float(''.join(letter_mapping.get(c, c) for c in block))
     

    with col3:
        
        floor_area = st.number_input("**Enter the area**", value=95.0)
        
        flat_model = st.selectbox("**Select Flat Model**", list(flat_model_mapping.keys()))
        flat_model_value = flat_model_mapping[flat_model]  

        lease_commence_date = st.number_input("**Enter the lease commence year**", value=1990)

        remaining_lease = st.text_input("**Enter the remaining lease duration (years-months e.g., '63-7')**")

    with col5:
                
        resale_year = st.number_input('**Enter the resale year**', min_value=lease_commence_date,max_value=2023)

        resale_month = st.number_input("**Enter the resale month**",max_value=12)

        storey_lower = st.number_input("**Enter the lower bound of the storey range**",min_value=0,max_value=10)
        
        storey_upper = st.number_input("**Enter the upper bound of the storey range**", min_value=storey_lower)

     
    with col3:
        
        button = st.button(":yellow[**PREDICT THE PRICE**]", use_container_width=True)
    
        if button:
            predicted_price = predict_price(town, flat_type_value, block_decimal, selected_street_number,floor_area,flat_model_value, lease_commence_date, remaining_lease,resale_year,resale_month, storey_lower, storey_upper)
            st.success(f"Predicted Resale Price: ${predicted_price:.2f}")