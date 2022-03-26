df_train['prix_q'] = df_train['prix_q'].astype(str)
fig = px.scatter_mapbox(df_train.sort_values(by='prix_q'),
                        lat='lat',
                        lon='lon',
                        hover_name='vin',
                        hover_data=['appellation', 'cuvee', 'producteur'],
                        zoom=5, 
                        color='prix_q', 
                        color_discrete_map ={'0': 'green', '1': 'yellow', '2': 'orange', '3': 'red'})
fig.update_layout(title='Vins de France - Prix (quartile)', mapbox_style='open-street-map')
fig.write_html('Carte de France - quantile.html', auto_open=True)
df_train['prix_q'] = df_train['prix_q'].astype(int)