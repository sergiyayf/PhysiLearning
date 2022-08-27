# python-loader
Python data loader for PhysiCell digital snapshots.


Code based on Raimond's code from Rf. http://www.mathcancer.org/blog/python-loader/ . Directory `load` contains original Raimonds loading functions for reading XML and Mat files and extracting neccessary data from the output files. Some of his files, also in the `plotting` directory have some good code for visualization of the results, but I haven't looked thouroughly at it yet. 

## Scatter plot cell densities

To scatter plot cell densities: load the output data using `pyMCDS` with input (filename, directory). Get the cell dataframe using method: `get_cell_df()`. 

Now you can plot cell densities simply by inputing your cells dataframe to the plotting function in the `plot` package. Use for example `plot.density.get_single_density_plot(cells, colormap)` to plot the density, or just have a look at the `get_density_plot_example`. 
## test_visualization 

this code contains initial messy implementation of lineage tracking, which would be brought to nice order soon;

## Cells on the colony boundary

Cells on the edge of the colony could be found using `get_perifery` in the auxiliary directory. It contains for now 2 usefull functions, 1 for getting front cells positions and types, and one for visualization of those cells, to ensure right front determination and see if there not too many gaps. For further details look at the source code. For an example function usage see `get_cell_perifery_example.py` 

## Clone history plots 


