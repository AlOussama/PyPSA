import pypsa
import re
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pandas as pd
from pypsa.networkclustering import get_clustering_from_busmap, busmap_by_kmeans

# Quelle: https://pypsa.readthedocs.io/en/latest/examples/spatial-clustering.html

n = pypsa.examples.scigrid_de()
n.lines["type"] = np.nan # delete the 'type' specifications to make this example easier

weighting = pd.Series(1, n.buses.index)
busmap2 = busmap_by_kmeans(n, bus_weightings=weighting, n_clusters=14)

C2 = get_clustering_from_busmap(n, busmap2)
nc2 = C2.network

print(C2)

pypsa.Network.export_to_csv_folder(nc2,"examples/scigrid-de/scigrid-cluster/CSV")
pypsa.Network.export_to_netcdf(nc2,"examples/scigrid-de/scigrid-cluster/nCDF/grid.nc")
pypsa.Network.export_to_hdf5(nc2,"examples/scigrid-de/scigrid-cluster/hdf5/grid.hdf5")

fig, (ax, ax1) = plt.subplots(1, 2, subplot_kw={"projection": ccrs.EqualEarth()}, figsize=(12,12))
plot_kwrgs = dict(bus_sizes=1e-3, line_widths=0.5)
n.plot(ax=ax, title="original", **plot_kwrgs)
nc2.plot(ax=ax1, title="clustered by kmeans", **plot_kwrgs)
fig.tight_layout()
plt.savefig("examples/scigrid-de/scigrid-cluster/Plot/cluster.png")
plt.show()