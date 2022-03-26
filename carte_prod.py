fig = px.scatter_mapbox(df_train.sort_values(by='region'),
                        lat='lat', 
                        lon='lon', 
                        hover_name="producteur",
                        hover_data=['region'],
                        color='region',
                        color_discrete_sequence=px.colors.qualitative.D3,
                        zoom=5)
fig.update_layout(title="Carte de France des producteurs de vins", mapbox_style="open-street-map")
fig.write_html('Carte de France - Producteurs.html', auto_open=True)