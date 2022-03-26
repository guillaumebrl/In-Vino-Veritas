fig = px.scatter_mapbox(df_train.sort_values(by='appellation'),
                        lat='lat', 
                        lon='lon', 
                        hover_name="vin",
                        hover_data=['appellation', 'producteur', 'region'],
                        color='appellation',
                        color_discrete_sequence=px.colors.qualitative.D3,
                        zoom=5)
fig.update_layout(title="Carte de France des vins", mapbox_style="open-street-map")
fig.write_html('Carte de France - Appellations.html', auto_open=True)