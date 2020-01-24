import cartopy.crs as ccrs 
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns


data_url = 'http://bit.ly/2cLzoxH'
gapminder = pd.read_csv(data_url)
print(gapminder.head(3))

df1 = gapminder[['continent', 'year','lifeExp']]
print(df1.head())

heatmap1_data = pd.pivot_table(df1, values='lifeExp', index=['continent'], columns='year')

#sns.heatmap(heatmap1_data, cmap="YlGnBu")

#plt.show()




mypt = (6, 56)

ax0 = plt.subplot(221, projection=ccrs.PlateCarree())
ax1 = plt.subplot(222, projection=ccrs.Mercator())
ax1bis = plt.subplot(223, projection=ccrs.Mercator())
ax2 = plt.subplot(224, projection=ccrs.Mercator())

def plotpt(ax, extent=(-15,15,46,62), **kwargs):
    ax.plot(mypt[0], mypt[1], 'r*', ms=20, **kwargs)
    ax.set_extent(extent)
    sns.heatmap(heatmap1_data, cmap="YlGnBu")
    #ax.coastlines(resolution='50m')
    ax.gridlines(draw_labels=True)
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

plotpt(ax0) # correct because projection and data share the same system
plotpt(ax1, transform=ccrs.Mercator()) # WRONG
plotpt(ax1bis, transform=ccrs.PlateCarree()) # Correct, projection and transform are different!
plotpt(ax2, extent=(-89,89,-89,89), transform=ccrs.Mercator()) # WRONG

plt.show()