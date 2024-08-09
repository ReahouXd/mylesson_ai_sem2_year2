import pandas as pd
import plotly.express as px

data = pd.read_csv('mtcars.csv')
fig = px.scatter(data, x='hp', y='mpg',
                 title='Horsepower vs. Miles per Gallon',
                 labels={'hp': 'Horsepower', 'mpg': 'Miles per Gallon'},
                 hover_data=['carb'])
fig.write_html('scatter_plot.html')
fig.show()