import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

from sklearn.cluster import KMeans



# Read CSV

file_name = r"C:\Users\PAVAN\Downloads\API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_5359189.xls"

df = pd.read_csv(file_name, skiprows=4)



# Selecting the columns to be used

columns_to_use = [str(year) for year in range(1971, 2015)]

df_years = df[['Country Name', 'Country Code'] + columns_to_use]



# Fill missing values with the mean

df_years = df_years.fillna(df_years.mean())





# Importing the functions from topic 8

def scaler(df):

    """ Expects a dataframe and normalises all 

        columnsto the 0-1 range. It also returns 

        dataframes with minimum and maximum for

        transforming the cluster centres"""



    # Uses the pandas methods

    df_min = df.min()

    df_max = df.max()



    df = (df-df_min) / (df_max - df_min)



    return df, df_min, df_max





def backscale(arr, df_min, df_max):

    """ Expects an array of normalised cluster centres and scales

        it back. Returns numpy array.  """



    # convert to dataframe to enable pandas operations

    minima = df_min.to_numpy()

    maxima = df_max.to_numpy()



    # loop over the "columns" of the numpy array

    for i in range(len(minima)):

        arr[:, i] = arr[:, i] * (maxima[i] - minima[i]) + minima[i]



    return arr





# Normalize the data using the custom scaler function

df_normalized, df_min, df_max = scaler(df_years[columns_to_use])



# Find the optimal number of clusters using the elbow method

inertia = []

num_clusters = range(1, 11)



for k in num_clusters:

    kmeans = KMeans(n_clusters=k, random_state=42)

    kmeans.fit(df_normalized)

    inertia.append(kmeans.inertia_)



# Plot the explained variation as a function of the number of clusters

plt.figure(figsize=(12, 8))

plt.plot(num_clusters, inertia, marker='o')

plt.xlabel('Number of Clusters')

plt.ylabel('Inertia')

plt.title('Elbow Method for Optimal Number of Clusters')

plt.show()



# Set the optimal number of clusters based on the elbow plot

optimal_clusters = 2



# Perform K-means clustering with the optimal number of clusters

kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)

df_years['Cluster'] = kmeans.fit_predict(df_normalized)



# Calculate cluster centers and scale them back to the original scale

cluster_centers = backscale(kmeans.cluster_centers_, df_min, df_max)



# Plot the clusters

plt.figure(figsize=(12, 8))



for i in range(optimal_clusters):

    cluster_data = df_years[df_years['Cluster'] == i][columns_to_use].mean()

    plt.plot(columns_to_use, cluster_data, label=f'Cluster {i+1}', linewidth=2)



plt.plot(columns_to_use, cluster_centers.T, 'k+', markersize=10, label='Cluster Centers')



plt.xlabel('Year')

plt.ylabel('Electric power consumption (kWh per capita))')

plt.title('Electric power consumption Clustering')



# Adjust x-axis labels to display in intervals of 5 years

years_interval = [str(year) for year in range(1971, 2015, 5)]

plt.xticks(years_interval, years_interval)



plt.legend()

plt.show()





import random



# Create a new DataFrame to store 5 country names from each cluster

cluster_countries = pd.DataFrame()



# List of priority countries

priority_countries = ['China', 'United States', 'Russian Federation', 'Germany', 'Ukraine']



for i in range(optimal_clusters):

    cluster_data = df_years[df_years['Cluster'] == i][['Country Name', 'Cluster']]

    

    # Select priority countries if they belong to the current cluster

    priority_cluster_data = cluster_data[cluster_data['Country Name'].isin(priority_countries)].head(5)



    # Select random countries from the remaining countries in the cluster

    remaining_cluster_data = cluster_data[~cluster_data['Country Name'].isin(priority_countries)].sample(5 - len(priority_cluster_data))



    # Combine priority countries and random countries

    combined_cluster_data = pd.concat([priority_cluster_data, remaining_cluster_data])



    # Add the combined_cluster_data to the cluster_countries DataFrame

    cluster_countries = pd.concat([cluster_countries, combined_cluster_data])



# Reset the index and display the DataFrame

cluster_countries.reset_index(drop=True, inplace=True)

display(cluster_countries)







# List of 5 countries to plot

selected_countries = ['China', 'Russian Federation', 'Germany', 'Ukraine', 'Low income']



# Filter the data to include only the selected countries

selected_data = df_years[df_years['Country Name'].isin(selected_countries)]



# Plot the time series for the selected countries

plt.figure(figsize=(12, 8))



for _, country_data in selected_data.iterrows():

    plt.plot(columns_to_use, country_data[columns_to_use], label=country_data['Country Name'], linewidth=2)



plt.xlabel('Year')

plt.ylabel('Electric power consumption (kWh per capita)')

plt.title('Electric power consumption for Cluster 1')



# Adjust x-axis labels to display in intervals of 5 years

years_interval = [str(year) for year in range(1971, 2015, 5)]

plt.xticks(years_interval, years_interval)



plt.legend()

plt.show()





import matplotlib.pyplot as plt



# List of 5 countries you want to plot

selected_countries = ['North America', 'United States', 'Qatar', 'Australia', 'Iceland']



# Filter the data to include only the selected countries

selected_data = df_years[df_years['Country Name'].isin(selected_countries)]



# Plot the time series for the selected countries

plt.figure(figsize=(12, 8))



for _, country_data in selected_data.iterrows():

    plt.plot(columns_to_use, country_data[columns_to_use], label=country_data['Country Name'], linewidth=2)



plt.xlabel('Year')

plt.ylabel('Electric power consumption (kWh per capita)')

plt.title('Electric power consumption for Cluster 1')



# Adjust x-axis labels to display in intervals of 5 years

years_interval = [str(year) for year in range(1971, 2015, 5)]

plt.xticks(years_interval, years_interval)



plt.legend()

plt.show()









from scipy.optimize import curve_fit





def err_ranges(x, func, param, sigma):

    """

    Calculates the upper and lower limits for the function, parameters and

    sigmas for single value or array x. Functions values are calculated for 

    all combinations of +/- sigma and the minimum and maximum is determined.

    Can be used for all number of parameters and sigmas >=1.

    

    This routine can be used in assignment programs.

    """



    import itertools as iter

    

    # initiate arrays for lower and upper limits

    lower = func(x, *param)

    upper = lower

    

    uplow = []   # list to hold upper and lower limits for parameters

    for p, s in zip(param, sigma):

        pmin = p - s

        pmax = p + s

        uplow.append((pmin, pmax))

        

    pmix = list(iter.product(*uplow))

    

    for p in pmix:

        y = func(x, *p)

        lower = np.minimum(lower, y)

        upper = np.maximum(upper, y)

        

    return lower, upper   









# Read CSV

file_name = 'API_EG.USE.ELEC.KH.PC_DS2_en_csv_v2_5359189.csv'

df = pd.read_csv(file_name, skiprows=4)



# Selecting the columns to be used

columns_to_use = [str(year) for year in range(1970, 2015)]

df_years = df[['Country Name', 'Country Code'] + columns_to_use]



# Fill missing values with the mean

df_years = df_years.fillna(df_years.mean())



# Define a polynomial function to fit the data

def polynomial(x, a, b, c, d):

    return a * x**3 + b * x**2 + c * x + d



# Select data for a specific country

country = 'United States'  # Replace with your desired country

df_country = df_years[df_years['Country Name'] == country][columns_to_use].values.flatten()



# X values (years)

x_data = np.arange(1970, 2015)

y_data = df_country



# Fit the polynomial function to the data

params, params_covariance = curve_fit(polynomial, x_data - 1970, y_data)

sigma = np.sqrt(np.diag(params_covariance))



# Predict values in ten years

x_future = np.arange(1970, 2050)

y_pred = polynomial(x_future - 1970, *params)



# Calculate the confidence range using the err_ranges function

y_lower, y_upper = err_ranges(x_future - 1970, polynomial, params, sigma)



# Plot the data, the best fitting function, and the confidence range

plt.figure(figsize=(12, 8))

plt.scatter(x_data, y_data, label='Data', alpha=0.5)

plt.plot(x_future, y_pred, label='Best Fit', color='r', linewidth=2)

plt.fill_between(x_future, y_lower, y_upper, color='r', alpha=0.2, label='Confidence Range')



plt.xlabel('Year')

plt.ylabel('Electric power consumption (kWh per capita)')

plt.title(f'{country} Electric Power Consumption - Polynomial Fit with Confidence Range')

plt.legend()



plt.show()

