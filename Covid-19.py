#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import  pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from shapely.geometry  import point,polygon
import plotly.express as px
import seaborn as sns
import geopandas as gpd
import warnings
import plotly as p
from fbprophet import Prophet
import folium


# In[ ]:


get_ipython().system('pip install geopandas')


# In[ ]:


confirmed = pd.read_csv("/content/drive/My Drive/Colab Notebooks/time_series_covid19_confirmed_global copy.csv")
death = pd.read_csv("/content/drive/My Drive/Colab Notebooks/time_series_covid19_deaths_global copy.csv")
recovered = pd.read_csv("/content/drive/My Drive/Colab Notebooks/time_series_covid19_recovered_global.csv")
print(confirmed.shape)
print(death.shape)
print(recovered.shape)

confirmed['Province/State'].fillna(confirmed['Country/Region'], inplace=True)
death['Province/State'].fillna(death['Country/Region'], inplace=True)
recovered['Province/State'].fillna(recovered['Country/Region'], inplace=True)

confirmed = pd.melt(confirmed,id_vars=['Province/State','Country/Region','Lat','Long'],var_name=['date'])
death = pd.melt(death,id_vars=['Province/State','Country/Region','Lat','Long'],var_name=['date'])
recovered = pd.melt(recovered,id_vars=['Province/State','Country/Region','Lat','Long'],var_name=['date'])

confirmed['date'] = pd.to_datetime(confirmed['date'])
death['date'] = pd.to_datetime(death['date'])
recovered['date'] = pd.to_datetime(recovered['date'])

print(confirmed.head())
print(death.tail())
print(confirmed.tail())
print(death.shape)
print(recovered.shape)
confirmed.columns = confirmed.columns.str.replace('value','confirmed')
death.columns = death.columns.str.replace('value','deaths')
recovered.columns = recovered.columns.str.replace('value','recovered')
print(confirmed.tail())
print(death.tail())
print(recovered.tail())
rrecovered = recovered.fillna(0)

covid19 = confirmed.merge(death[['Province/State','Country/Region','date','deaths']], how="left", left_on = ['Province/State','Country/Region','date'],    right_on = ['Province/State', 'Country/Region','date'])
covid19 = covid19.merge(recovered[['Province/State','Country/Region','date','recovered']], how = 'left',left_on = ['Province/State','Country/Region','date'],  right_on = ['Province/State', 'Country/Region','date'])

print(covid19.isnull().sum())
covid19["recovered"]=covid19["recovered"].fillna(0)
print(covid19.isnull().sum())
print(covid19.head())
print(covid19.tail())
covid19.shape


# In[ ]:


confirmed = covid19.groupby('date').sum()['confirmed'].reset_index()
confirmed.shape
deaths = covid19.groupby('date').sum()['deaths'].reset_index()
recovered = covid19.groupby('date').sum()['recovered'].reset_index()
confirmed.columns=['ds','y']
print(confirmed.tail(5))
confirmed['ds'] = pd.to_datetime(confirmed['ds'])
m = Prophet(interval_width=0.95)
m.fit(confirmed)
future = m.make_future_dataframe(periods=15)
future.tail()


# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


from google.colab import files
confirmed_forecast_plot = m.plot(forecast)
plt.savefig("confirmed_forcast.png")
files.download("confirmed_forcast.png")


# In[ ]:


confirmed_forecast_plot =m.plot_components(forecast)
plt.savefig("Conf_weekly.png")
files.download("Conf_weekly.png")


# In[ ]:


recovered.columns = ['ds','y']
recovered['ds'] = pd.to_datetime(recovered['ds'])


# In[ ]:


m = Prophet(interval_width=0.95)
m.fit(recovered)
future = m.make_future_dataframe(periods=15)
future.tail()


# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


recovered_forecast_plot = m.plot(forecast.tail(20))
plt.savefig("recovered_forcast.png")
files.download("recovered_forcast.png")


# In[ ]:


deaths.columns = ['ds','y']
deaths['ds'] = pd.to_datetime(deaths['ds'])


# In[ ]:


m = Prophet(interval_width=0.95)
m.fit(deaths)
future = m.make_future_dataframe(periods=15)
future.tail()


# In[ ]:


forecast = m.predict(future)
forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()


# In[ ]:


confirmed_forecast_plot =m.plot_components(forecast)
plt.savefig("death_weekly.png")
files.download("death_weekly.png")


# In[ ]:


recovered_forecast_plot = m.plot(forecast.tail(20))
plt.savefig("death_forcast.png")
files.download("death_forcast.png")


# In[ ]:


covid19.groupby('Province/State')['confirmed'].max()


# In[ ]:


l = covid19.groupby('country')['recovered'].max()


# In[ ]:


k = covid19.groupby('Province/State')['deaths'].max()
print(k.idxmax())
k.max()


# In[ ]:


plt.boxplot(k)
plt.title('deaths according to countries')
plt.savefig("deaths_bp.png")
files.download("deaths_bp.png")


# In[ ]:


map = folium.Map(location=[20, 70], zoom_start=4,tiles='Stamenterrain')

for lat, lon, value, name in zip(covid19['Lat'], covid19['Long'], covid19['confirmed'], covid19['Province/State']):
    folium.CircleMarker([lat, lon], radius=value*0.8, popup = ('<strong>State</strong>: ' + str(name).capitalize() + '<br>''<strong>Total Cases</strong>: ' + str(value) + '<br>'),color='red',fill_color='red',fill_opacity=0.3 ).add_to(map)
map


# In[ ]:


plt.boxplot(k)
plt.title('confirmed according to countries')
plt.savefig("confirmed_bp.png")
files.download("confirmed_bp.png")


# In[ ]:


plt.boxplot(k)
plt.title('recovered according to countries')
plt.savefig("recovered_bp.png")
files.download("recovered_bp.png")


# In[ ]:


last_date =covid19['date'].max()
df_countries = covid19[covid19['date']==last_date]
df_countries =covid19.groupby('Province/State',as_index=False)['confirmed','deaths','recovered'].sum()
df_countries = df_countries.nlargest(10,'confirmed')
df_trend = covid19.groupby(['date','Province/State'],as_index=False)['confirmed','deaths','recovered'].sum()
df_trend = df_trend.merge(df_countries, on='Province/State')
df_trend.rename(columns={'Province/State':'Country', 'confirmed_x':'Cases', 'deaths_x':'Deaths','recovered_x':'recovered'}, inplace=True)
df_trend['log(Cases)'] = np.log(df_trend['Cases']+1)# Added 1 to remove error due to log(0).
df_trend['log(Deaths)'] = np.log(df_trend['Deaths']+1)
df_trend['log(recovered)'] = np.log(df_trend['recovered']+1)
df_trend.head()


# In[ ]:


px.line(df_trend, x='date', y='Cases', color='Country', title='COVID19 Total Cases growth for top 10 worst affected countries')


# In[ ]:


px.line(df_trend, x='date', y='Deaths', color='Country', title='COVID19 Total deaths growth for top 10 worst affected countries')


# In[ ]:


px.line(df_trend, x='date', y='recovered', color='Country', title='COVID19 Total recovered growth for top 10 worst affected countries')


# In[ ]:


covid19.dtypes

