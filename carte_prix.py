fig = px.scatter_mapbox(df_train, lat='lat', lon='lon', hover_name='vin', hover_data=['appellation', 'cuvee', 'producteur'], zoom=5, 
                        color=np.log(df_train['prix_m']), color_continuous_scale=px.colors.sequential.Burg, range_color=(1,9))
fig.update_layout(title='Vins de France - Prix (log)', mapbox_style='open-street-map')
fig.write_html('Carte de France - Prix.html', auto_open=True)