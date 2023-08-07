import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
import numpy as np
import plotly.express as px
import re
from hugchat import hugchat
from hugchat.login import Login
from streamlit_option_menu import option_menu
import seaborn as sns
import matplotlib.pyplot as plt

# Function to standardize names
def standardize_name(name):
    name = str(name).lower()  # Convert to lowercase
    name = re.sub(r'\W+', '', name)  # Remove special characters
    return name

# Set page configuration
st.set_page_config(
    page_title="Lassa Fever Dashboard",
    page_icon="🔬",
    layout="wide"
)
# Read the data
file_path = 'SeperateYears.csv'

# Load the data
df = pd.read_csv('SeperateYears.csv')

# Group by admin1Name, LGA, and perform the aggregation
aggregated_df = df.groupby(['admin1Name', 'LGA']).agg({
    'UrbanProp_ESA': 'mean',
    'ForestProp_ESA': 'mean',
    'AgriProp_ESA': 'mean',
    'Cases': 'sum',
    'PovertyProp_Mean': 'mean'
}).reset_index()

# Convert Year column to datetime
df['Year'] = pd.to_datetime(df['Year'], format="%Y")

# Create options menu
with st.sidebar:
    choice = option_menu("Navigation", ["📊 Home", "✍️ Form", "🔍 Prediction"],
                         icons=['house', 'pencil-square', 'search'], menu_icon="list", default_index=0)

# Functions for each of the tabs in the home page
def cases_distribution():
    st.header('Begin exploring the data using the menu on the left')
    
    # Rename the columns
    df_renamed = df.rename(columns={'Cases_SuspectedUnconfirmed': 'Suspected Cases', 'Reports_All': 'Reports'})

    # Compute sums by year for the specific columns
    columns_to_sum = ['Cases', 'Suspected Cases', 'Reports']
    df_summed = df_renamed.groupby(df_renamed['Year'].dt.year)[columns_to_sum].sum().reset_index()
    
    # Add a title for the pie and bar charts
    st.subheader("LGA Data for the Selected State")

    # Calculate total cases for each state and LGA
    lga_cases = df_renamed.groupby(['admin1Name', 'LGA'])[['Cases', 'Suspected Cases', 'Reports']].sum().reset_index()

    # Dropdown to select a state, sorted alphabetically
    selected_state = st.selectbox('Select a state', sorted(df_renamed['admin1Name'].unique()), key='state_selection_cases')

    # Dropdown to select the data to display
    data_options = ['Cases', 'Suspected Cases', 'Reports']
    selected_data = st.selectbox('Select data to display', data_options)

    # Checkbox to select whether to include zero cases, with the default behavior reversed
    include_zero_cases = st.checkbox('Exclude zero cases', value=True)

    # Filter data based on the checkbox selection
    filtered_df = df_renamed[df_renamed[selected_data] > 0] if include_zero_cases else df_renamed
    filtered_lga_cases = lga_cases[lga_cases[selected_data] > 0] if include_zero_cases else lga_cases

    # Create a pie chart for the selected state
    state_df = filtered_df[filtered_df['admin1Name'] == selected_state]
    fig = px.pie(state_df, values=selected_data, names='LGA', title=f'{selected_data} by LGA in {selected_state}',
                 hover_data=[selected_data], labels={selected_data: 'Number of Cases'})

    # Set the hover template
    hover_template = '<b>%{label}</b><br>' + f'{selected_data}: %{{value}}'
    fig.update_traces(hovertemplate=hover_template)

    st.plotly_chart(fig)

    # Create a bar chart for the LGAs in the selected state
    lga_df = filtered_lga_cases[filtered_lga_cases['admin1Name'] == selected_state]
    fig = px.bar(lga_df, x='LGA', y=selected_data, title=f'{selected_data} by LGA in {selected_state}',
                 hover_data=[selected_data], labels={selected_data: 'Number of Cases'},
                 color=selected_data, color_continuous_scale='Blues')
    st.plotly_chart(fig)

    # Scatter plot and metrics
    st.subheader("Distribution of Travel Times")

    # Button to switch between different sets of variables
    travel_time_vars = st.radio('Switch Travel Time Variables', ['Mean Travel Time', 'Travel Time'])

    # Define the labels for each variable
    labels = {
        'LabTravelTime': 'Lab Travel Time (Mean)' if travel_time_vars == 'Mean Travel Time' else 'Lab Travel Time',
        'CityTravTime_Mean': 'City Travel Time (Mean)' if travel_time_vars == 'Mean Travel Time' else 'City Travel Time',
        'HospitalDist_mean_km': 'Hospital Distance (Mean)',
        'HealthFacilityDist_mean_km': 'Health Facility Distance (Mean)',
        'AllHealthFacility_TravelTime': 'All Health Facility Travel Time',
        'Hospital_TravelTime': 'Hospital Travel Time',
        'LabDist': 'Lab Distance'
    }

    variables = ['LabTravelTime', 'CityTravTime_Mean', 'HospitalDist_mean_km', 'HealthFacilityDist_mean_km'] \
        if travel_time_vars == 'Mean Travel Time' else ['AllHealthFacility_TravelTime', 'Hospital_TravelTime', 'LabDist']

    # Dropdown to select a variable
    var = st.selectbox('Select Variable to Visualize', options=variables, format_func=lambda x: labels[x])

    # Update dataframe with new labels
    df_renamed = df.rename(columns={var: labels[var], 'admin1Name': 'State'})

    # Button to switch between displaying by state or by LGA within each state
    view_by_state = st.button('View by State', key='view_by_state_lab_distance')

    if view_by_state:
        fig = px.box(df_renamed, x='State', y=labels[var], color='State', title=f"Distribution of {labels[var]} by State", height=600)
    else:
        state = st.selectbox('Select a State', sorted(df_renamed['State'].unique()))
        df_state = df_renamed[df_renamed['State'] == state]
        fig = px.box(df_state, x='LGA', y=labels[var], color='LGA', title=f"Distribution of {labels[var]} by LGA in {state}", height=600)
    
    st.plotly_chart(fig, use_container_width=True)

    # Metrics to compare different years
    years = sorted(df_renamed['Year'].dt.year.unique())
    year1, year2 = st.multiselect('Select Two Years to Compare', options=years, default=[years[0], years[-1]])

    if year1 and year2:
        avg_year1 = round(df_renamed[df_renamed['Year'].dt.year == year1][labels[var]].mean(),2)
        avg_year2 = round(df_renamed[df_renamed['Year'].dt.year == year2][labels[var]].mean(),2)
        st.metric(label=f"Average {labels[var]} in {year1}", value=avg_year1)
        st.metric(label=f"Average {labels[var]} in {year2}", value=avg_year2, delta=avg_year2-avg_year1)

    # Plot line chart with custom colors
    fig = px.line(df_summed, x='Year', y=columns_to_sum, title="Cases over time",
                  color_discrete_sequence=px.colors.qualitative.Bold)  # Using a bold color sequence
    fig.update_layout(xaxis_title="Year", yaxis_title="Cases")
    st.plotly_chart(fig, use_container_width=True)

def infection_rate_variables():
    st.header("Infection Rate Variables")
    demographic_distribution()
    climate_cases_relation(df)
    correlation_heatmaps()  # Call the function to render the heatmaps

def demographic_distribution():
    st.subheader("Distribution of Demographic Proportions and Poverty Rates")

    # Colors for different demographics
    colors = ['blue', 'green', 'orange']

    # Button to switch between displaying by state or by LGA within each state
    view_by_state = st.button('View by State', key='view_by_state_demographic')

    min_marker_size = 5
    max_marker_size = 50
    min_line_width = 1
    max_line_width = 5

    if view_by_state:
        # Plotting by state
        fig = go.Figure()

        for prop, label, color in zip(['Urb_prop_ESA2015', 'Forest_prop_ESA2015', 'Agri_prop_ESA2015'], ['Urban', 'Forest', 'Agriculture'], colors):
            # Calculate mean values for each property by state
            state_means = df.groupby('admin1Name')[prop].mean()
            state_cases = df.groupby('admin1Name')['Cases'].sum()
            state_poverty = df.groupby('admin1Name')['PovertyProp_Mean'].mean()

            marker_sizes = state_means * 100 / 100 * (max_marker_size - min_marker_size) + min_marker_size
            line_widths = (state_cases - state_cases.min()) / (state_cases.max() - state_cases.min()) * (max_line_width - min_line_width) + min_line_width

            fig.add_trace(go.Scatter(x=state_means.index,
                                     y=state_means * 100,
                                     mode='markers',
                                     name=label,
                                     hovertemplate='State: %{x}<br>Cases: %{text}<br>Poverty Rate: %{marker.opacity:.2%}<br>Proportion: %{y:.2f}%',
                                     text=state_cases,
                                     marker=dict(size=marker_sizes, color=color, opacity=state_poverty, line=dict(width=line_widths, color='black'))))

        fig.update_layout(title="Demographic Proportions vs States",
                          xaxis_title='State',
                          yaxis_title='Proportion (%)',
                          yaxis=dict(range=[0, 100]),
                          legend_title="Demographic")

    else:
        # Dropdown to select a state, sorted alphabetically
        selected_state = st.selectbox('Select a state', sorted(df['admin1Name'].unique()), key='state_selection_demographic')

        # Filter by the selected state
        df_state = df[df['admin1Name'] == selected_state]

        # Create a scatter plot for Urban, Forest, and Agriculture proportions
        fig = go.Figure()

        for prop, label, color in zip(['Urb_prop_ESA2015', 'Forest_prop_ESA2015', 'Agri_prop_ESA2015'], ['Urban', 'Forest', 'Agriculture'], colors):
            # Calculate opacity based on poverty rate
            opacity = df_state['PovertyProp_Mean']
            marker_sizes = df_state[prop] * 100 / 100 * (max_marker_size - min_marker_size) + min_marker_size
            line_widths = (df_state['Cases'] - df_state['Cases'].min()) / (df_state['Cases'].max() - df_state['Cases'].min()) * (max_line_width - min_line_width) + min_line_width

            fig.add_trace(go.Scatter(x=df_state['LGA'],
                                     y=df_state[prop] * 100,
                                     mode='markers',
                                     name=label,
                                     hovertemplate='LGA: %{x}<br>Cases: %{text}<br>Poverty Rate: %{marker.opacity:.2%}<br>Proportion: %{y:.2f}%',
                                     text=df_state['Cases'],
                                     marker=dict(size=marker_sizes, color=color, opacity=opacity, line=dict(width=line_widths, color='black'))))

        fig.update_layout(title=f"Demographic Proportions vs LGAs in {selected_state}",
                          xaxis_title='LGA',
                          yaxis_title='Proportion (%)',
                          yaxis=dict(range=[0, 100]),
                          legend_title="Demographic")

    st.plotly_chart(fig, use_container_width=True)

def climate_cases_relation(df):
    st.header("Climate and Cases Relation")

    # Mapping of variables to descriptive labels
    precipitation_labels = {
        'PrecipMeanAnnual_2011_2019_CHIRPS': 'CHIRPS Mean Annual Precipitation',
        'PrecipMeanWettest_2011_2019_CHIRPS': 'CHIRPS Mean Wettest Precipitation',
        'PrecipMeanDriest_2011_2019_CHIRPS': 'CHIRPS Mean Driest Precipitation',
        'PrecipMonthlyCoefv_2011_2019_CHIRPS': 'CHIRPS Monthly Coefficient of Precipitation',
        'CHELSA_PrecipTotalAnnual': 'CHELSA Total Annual Precipitation',
        'CHELSA_PrecipSeasonality': 'CHELSA Seasonality of Precipitation',
        'CHELSA_PrecipWettestQ': 'CHELSA Wettest Quarter Precipitation'
    }
    temperature_labels = {
        'TempMeanAnnual_201119_NOAA': 'NOAA Mean Annual Temperature',
        'TempMonthlyCoefv_201118_NOAA': 'NOAA Monthly Coefficient of Temperature',
        'CHELSA_TempAnnualMean': 'CHELSA Annual Mean Temperature',
        'CHELSA_TempSeasonality': 'CHELSA Seasonality of Temperature',
        'CHELSA_TempMeanWarmestQ': 'CHELSA Mean Warmest Quarter Temperature'
    }

    # Dropdown to select level (State or LGA)
    level = st.selectbox('Select Level', ['State', 'LGA'], key='level')

    # Dropdown to select precipitation variable
    selected_precipitation_label = st.selectbox('Select Precipitation Variable', list(precipitation_labels.values()))
    selected_precipitation = [key for key, value in precipitation_labels.items() if value == selected_precipitation_label][0]

    # Dropdown to select temperature variable
    selected_temperature_label = st.selectbox('Select Temperature Variable', list(temperature_labels.values()))
    selected_temperature = [key for key, value in temperature_labels.items() if value == selected_temperature_label][0]

    # Filter data based on view selection (State or LGA)
    if level == 'State':
        df_view = df.groupby('admin1Name').agg({
            selected_precipitation: 'mean',
            selected_temperature: 'mean',
            'Cases': 'sum'
        }).reset_index()
    else:
        df_view = df

    # Round the values
    df_view[selected_precipitation] = df_view[selected_precipitation].round().astype(int)
    df_view[selected_temperature] = df_view[selected_temperature].round().astype(int)

    # Add units to the labels
    selected_precipitation_label += ' (mm)'
    selected_temperature_label += ' (°C)'

    # Plot the scatter plot
    fig = px.scatter(df_view,
                     x=selected_precipitation,
                     y=selected_temperature,
                     size='Cases',
                     hover_data=['admin1Name' if level == 'State' else 'LGA', 'Cases'],
                     title='Climate vs Cases Relation',
                     labels={'admin1Name': 'State', 'LGA': 'Local Government Area'},
                     size_max=60)

    fig.update_layout(xaxis_title=selected_precipitation_label,
                      yaxis_title=selected_temperature_label,
                      legend_title="Cases")

    st.plotly_chart(fig, use_container_width=True)

def correlation_heatmaps():
    # ------ Temperature and Precipitation Correlation Heatmap ------
 st.header("Temperature and Precipitation Correlation with Cases")

 # Select relevant columns
 columns = [
     'PrecipMeanAnnual_2011_2019_CHIRPS', 'PrecipMeanWettest_2011_2019_CHIRPS',
     'PrecipMeanDriest_2011_2019_CHIRPS', 'PrecipMonthlyCoefv_2011_2019_CHIRPS',
     'TempMeanAnnual_201119_NOAA', 'TempMonthlyCoefv_201118_NOAA',
     'CHELSA_TempAnnualMean', 'CHELSA_TempSeasonality', 'CHELSA_TempMeanWarmestQ',
     'CHELSA_PrecipTotalAnnual', 'CHELSA_PrecipSeasonality', 'CHELSA_PrecipWettestQ',
     'Cases', 'Cases_SuspectedUnconfirmed', 'Reports_All']

 # Compute the correlation matrix
 corr_matrix = df[columns].corr()

 # Apply normalization
 corr_matrix = (corr_matrix + 1) / 2 * 100

 # Define labels
 labels = [
     'CHIRPS Mean Annual Precipitation', 'CHIRPS Mean Wettest Precipitation',
     'CHIRPS Mean Driest Precipitation', 'CHIRPS Monthly Coefficient of Precipitation',
     'NOAA Mean Annual Temperature', 'NOAA Monthly Coefficient of Temperature',
     'CHELSA Annual Mean Temperature', 'CHELSA Temperature Seasonality', 'CHELSA Mean Warmest Quarter Temperature',
     'CHELSA Total Annual Precipitation', 'CHELSA Precipitation Seasonality', 'CHELSA Wettest Quarter Precipitation',
     'Total Cases', 'Suspected Cases', 'Total Reports']
    

 # Create a figure and axes object
 fig, ax = plt.subplots(figsize=(14, 10))

 # Create the heatmap using the axes object
 sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=100, cbar_kws={'label': 'Correlation (%)'}, fmt='.0f', linewidths=.5, xticklabels=labels, yticklabels=labels, ax=ax)

 # Display the plot in Streamlit
 st.pyplot(fig)

 # ------ Cases Correlation with Geographical and Demographical Information Heatmap ------

 st.header("Cases Correlation with Geographical and Demographical Information")

 # Select relevant columns
 columns = [
     'Cases', 'Cases_SuspectedUnconfirmed', 'Reports_All', 'LabDist', 'LabTravelTime',
     'TotalPopulation_ByYear', 'AgriProp_ESA', 'UrbanProp_ESA', 'ForestProp_ESA', 'LGAarea_km2',
     'CityTravTime_Mean', 'HospitalDist_mean_km', 'HealthFacilityDist_mean_km', 'ImprovedHousingPrev_PopWeighted',
     'Cropland_prop_ESA2015', 'Agri_prop_ESA2015', 'Urb_prop_ESA2015', 'Forest_prop_ESA2015',
     'PovertyProp_Mean', 'PovertyProp_PopWeighted', 'Hospital_TravelTime', 'AllHealthFacility_TravelTime',
     'PopDens2015', 'TotalPop2015', 'TotalUrbanPop2015', 'TotalRuralPop2015', 'TotalPop2015_PropUrban']

 # Compute the correlation matrix
 corr_matrix = df[columns].corr()

 # Apply normalization
 corr_matrix = (corr_matrix + 1) / 2 * 100

 # Define labels
 labels = [
     'Total Cases', 'Suspected Cases', 'Total Reports', 'Distance to Lab', 'Travel Time to Lab',
     'Total Population', 'Agricultural Proportion (ESA)', 'Urban Proportion (ESA)', 'Forest Proportion (ESA)', 'LGA Area (km²)',
     'Mean City Travel Time', 'Mean Hospital Distance (km)', 'Mean Health Facility Distance (km)', 'Improved Housing Prevalence (Pop Weighted)',
     'Cropland Proportion (ESA 2015)', 'Agricultural Proportion (ESA 2015)', 'Urban Proportion (ESA 2015)', 'Forest Proportion (ESA 2015)',
     'Mean Poverty Proportion', 'Poverty Proportion (Pop Weighted)', 'Hospital Travel Time', 'All Health Facility Travel Time',
     'Population Density (2015)', 'Total Population (2015)', 'Total Urban Population (2015)', 'Total Rural Population (2015)', 'Total Population Proportion Urban (2015)']

 # Create a figure and axes object
 fig, ax = plt.subplots(figsize=(20, 18))

 # Create the heatmap using the axes object
 sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=0, vmax=100, cbar_kws={'label': 'Correlation (%)'}, fmt='.0f', linewidths=.5, xticklabels=labels, yticklabels=labels, ax=ax)

 # Display the plot in Streamlit
 st.pyplot(fig)


def gis_map(df):
    st.header("GIS Map")

    # GIS Map Code for State
    # Load the GeoJSON file
    with open('NigeriaState.geojson') as f:
        nigeria_geojson = json.load(f)

    # Convert 'Cases' column to numeric
    df['Cases'] = pd.to_numeric(df['Cases'], errors='coerce')

    # Apply a logarithmic transformation to the 'Cases' column to reduce the impact of outliers
    df['Cases'] = df['Cases'].apply(lambda x: np.log(x + 1))

    # Replace 'Federal Capital Territory' with 'Fct, Abuja'
    df['admin1Name'] = df['admin1Name'].replace('Federal Capital Territory', 'Fct, Abuja')

    # Filter the DataFrame for 'admin1Name' and 'Year'
    df_state_year = df.groupby(['admin1Name', 'Year'])['Cases'].sum().reset_index()
    df_state_year.columns = ['state', 'Year', 'Cases']

    # Check for any states in the DataFrame that are not in the GeoJSON
    geojson_state_names = [feature['properties']['state'] for feature in nigeria_geojson['features']]
    missing_states = [state for state in df_state_year['state'].unique() if state not in geojson_state_names]
    if missing_states:
        st.write(f"The following states are in the DataFrame but not in the GeoJSON: {missing_states}")

    # Check for any states in the GeoJSON that are not in the DataFrame
    extra_states = [state for state in geojson_state_names if state not in df_state_year['state'].unique()]
    if extra_states:
        st.write(f"The following states are in the GeoJSON but not in the DataFrame: {extra_states}")

    # Define the Choropleth layer
    choropleth_layer_states = go.Choroplethmapbox(
        geojson=nigeria_geojson,
        locations=df_state_year['state'],  # DataFrame column with identifiers matching those in the GeoJSON file
        z=df_state_year['Cases'],  # DataFrame column with values to be color-coded
        featureidkey='properties.state',
        colorscale=[(0, 'lightblue'), (0.5, 'blue'), (1, 'darkblue')],  # Custom color scale
        zmin=0,  # Minimum color value
        zmax=np.log(df_state_year['Cases'].max() + 1),  # Maximum color value
        marker_opacity=0.5,  # Opacity of the areas
        marker_line_width=0,  # Border line width of the areas
        colorbar=dict(
            title="Log Cases"  
        ),
        customdata=np.stack((df_state_year['state'], np.exp(df_state_year['Cases']) - 1), axis=-1),  # Updated custom data
        hovertemplate="<b>%{customdata[0]}</b><br>Cases: %{customdata[1]:.0f}<extra></extra>",  # Updated hover information
    )

    # Define the base map
    fig = go.Figure(data=choropleth_layer_states)

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=5,
        mapbox_center={"lat": 9.0820, "lon": 8.6753},
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    # Show the title in Streamlit
    st.title('State Map for Lassa Fever Cases')

    # Show the figure in Streamlit
    st.plotly_chart(fig)

    # GIS Map Code for LGA
    # Load the GeoJSON file
    with open('LGA.json') as f:
        geojson = json.load(f)

    # Standardize the LGA names in the DataFrame
    df['LGA_standard'] = df['LGA'].apply(standardize_name)

    # Standardize the LGA names in the GeoJSON
    geojson_lga_names = [feature['properties']['admin2Name'] for feature in geojson['features']]
    geojson_lga_names_standard = [standardize_name(name) for name in geojson_lga_names]

    # Create a dictionary to map the standardized names in the GeoJSON to the original names
    reverse_mapping_dict = {standard_name: original_name for standard_name, original_name in zip(geojson_lga_names_standard, geojson_lga_names)}

    # Replace the standardized LGA names in the DataFrame with the original names from the GeoJSON
    df['LGA'] = df['LGA_standard'].replace(reverse_mapping_dict)

    # Remove the 'LGA_standard' column from the DataFrame
    df = df.drop(columns=['LGA_standard'])

    # Check for any LGAs in the DataFrame that are not in the GeoJSON
    missing_lgas = [lga for lga in df['LGA'].unique() if lga not in geojson_lga_names]
    if missing_lgas:
        st.write(f"The following LGAs are in the DataFrame but not in the GeoJSON: {missing_lgas}")

    # Check for any LGAs in the GeoJSON that are not in the DataFrame
    extra_lgas = [lga for lga in geojson_lga_names if lga not in df['LGA'].unique()]
    if extra_lgas:
        st.write(f"The following LGAs are in the GeoJSON but not in the DataFrame: {extra_lgas}")

    # Convert 'Cases' column to numeric
    df['Cases'] = pd.to_numeric(df['Cases'], errors='coerce')

    # Apply a logarithmic transformation to the 'Cases' column to reduce the impact of outliers
    df['Cases'] = df['Cases'].apply(lambda x: np.log(x + 1))

    # Define the Choropleth layer
    choropleth_layer = go.Choroplethmapbox(
        geojson=geojson,
        locations=df['LGA'],  # DataFrame column with identifiers matching those in the GeoJSON file
        z=df['Cases'],  # DataFrame column with values to be color-coded
        featureidkey='properties.admin2Name',  # Correct featureidkey
        colorscale=[(0, 'lightblue'), (0.5, 'blue'), (1, 'darkblue')],  # Custom color scale
        zmin=0,  # Minimum color value
        zmax=np.log(df['Cases'].max() + 1),  # Maximum color value
        marker_opacity=0.5,  # Opacity of the areas
        marker_line_width=0,  # Border line width of the areas
        colorbar=dict(
            title="Log Cases"  # Legend title
            ),
            customdata=np.stack((df['LGA'], np.exp(df['Cases']) - 1), axis=-1),  # Add custom data
            hovertemplate="<b>%{customdata[0]}</b><br>Cases: %{customdata[1]:.0f}<extra></extra>",  # Define hover information
    )

    # Define the base map
    fig = go.Figure(data=choropleth_layer)

    fig.update_layout(
        mapbox_style="carto-positron",
        mapbox_zoom=5,
        mapbox_center={"lat": 9.0820, "lon": 8.6753},
        margin={"r": 0, "t": 0, "l": 0, "b": 0}
    )

    # Show the title in Streamlit
    st.title('LGA Map for Lassa Fever Cases')

    # Show the figure in Streamlit
    st.plotly_chart(fig)
# Home page function
def home(df):
    overview_tabs = st.tabs(["Cases distribution", "Infection rate variables", "GIS Map"])
    with overview_tabs[0]:
        cases_distribution()
    with overview_tabs[1]:
        infection_rate_variables()
    with overview_tabs[2]:
        gis_map(df)

# Other page functions
def data_prediction():
    st.header("Data Prediction Page")

def individual_form():
    st.header("Symptom Questionnaire for Individuals")

    # Questions for individuals
    age = st.number_input("What is your age?", min_value=0, max_value=120, step=1)
    sex = st.selectbox("What is your sex?", ["Male", "Female", "Other"])
    state = st.selectbox("What state do you live in?", sorted(df['admin1Name'].unique()))
    lgas_in_state = df[df['admin1Name'] == state]['LGA'].unique()
    lga = st.selectbox("What LGA do you live in?", sorted(lgas_in_state))
    exposed_to_lf = st.radio("Have you been exposed to anyone with Lassa Fever in the past week?", ["Yes", "No"])
    main_symptoms = st.multiselect("What are your main symptoms?", 
                                   sorted(["Fever", "Headache", "Sore throat", "Muscle pain", 
                                    "Nausea, vomiting or diarrhea", "Abdominal pain", 
                                    "Facial swelling", "External bleeding from mouth or nose"]))

    # Store the data if required (e.g., in a database)
    # You can implement the storage logic here
    if st.button("Submit"):
        # Logic to store data in an Excel spreadsheet or database goes here
        # For now, we'll just print the data as an example
        st.success("Data submitted successfully!")
        st.write("Age:", age)
        st.write("Sex:", sex)
        st.write("State:", state)
        st.write("LGA:", lga)
        st.write("Exposed to Lassa Fever:", exposed_to_lf)
        st.write("Main Symptoms:", main_symptoms)

def organization_form():
    st.header("Lassa Fever Data Upload for Organizations")

    # Instructions for organizations
    st.markdown("""
        Please download the [template file](https://studentcitruscollege-my.sharepoint.com/:x:/g/personal/davalvarez928_student_citruscollege_edu/ES6bdWHQqVhPjsOlJ1mIkVEBuXCkY3Z-glkURScP7QLUlA?e=KuEXhz) and fill out the following information for each patient you have charted:

        - Age
        - Sex
        - State
        - LGA (Local Government Area)
        - Cases confirmed
        - Cases suspected

        Once you have filled out the template, please upload the file using the file uploader below.
    """)

    # File uploader for organizations
    organization_file = st.file_uploader('Upload your filled-out template')

    # Check if a file has been uploaded
    if organization_file is not None:
        # Load the data
        org_df = pd.read_csv(organization_file)

        # Display a success message and show the uploaded data
        st.success('File uploaded successfully!')
        st.dataframe(org_df)

def form():
    st.header("Form Page")

    # Radio button to choose between individual and organization
    form_type = st.radio("Are you an individual or an organization?", ["Individual", "Organization"])

    # Display symptom questionnaire for individuals
    if form_type == "Individual":
        individual_form()
    # Display data upload form for organizations
    elif form_type == "Organization":
        organization_form()

    # HugChat implementation
    st.subheader("AI-powered Chatbot")
    st.write("You can ask the AI-powered chatbot any additional questions you may have about Lassa Fever sickness.")

    hf_email = st.secrets["general"]["hf_email"]
    hf_password = st.secrets["general"]["hf_password"]

    # Create a function for generating LLM response
    def generate_response(prompt_input, email, passwd):
        sign = Login(email, passwd)
        cookies = sign.login()
        chatbot = hugchat.ChatBot(cookies=cookies.get_dict())
        return chatbot.chat(prompt_input)

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{"role": "assistant", "content": "How may I help you?"}]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # User-provided prompt
    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = generate_response(prompt, hf_email, hf_password) 
                st.write(response) 
        message = {"role": "assistant", "content": response}
        st.session_state.messages.append(message)

# Call the function associated with the choice
if choice == "📊 Home":
    home(df)
elif choice == "✍️ Form":
    form()
elif choice == "🔍 Prediction":
    data_prediction()