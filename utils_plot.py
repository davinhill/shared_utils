import numpy as np

def shiftedColorMap(cmap, start=0, midpoint=0.5, stop=1.0, color_output = 'plotly'):
    '''
    scale matplotlib colormap to set min, mid, and max values.

    output can be a list of rgba colors (for plotly), or a new cmap.
    '''
    from matplotlib import colors

    # shifted index to match the data
    step = (stop - start) / 1000
    max_step = (midpoint - start / 2)
    step = min(step, max_step)

    firsthalf = np.arange(start, midpoint, step)
    secondhalf = np.arange(midpoint, stop+step,step) - midpoint

    shift_index = np.hstack([
        (firsthalf / firsthalf.max())*0.5,
        (secondhalf / secondhalf.max())*0.5+0.5,
        ])

    if color_output == 'plotly':
        color_list = cmap(shift_index).tolist()
        # color_list = ['rgba(' + ','.join(color) + ')' for color in color_list]
        plotly_colorscale = []
        for i,color in enumerate(color_list):
            plotly_colorscale.append(
                # [shift_index[i], 'rgb(' + ','.join(color[:-1]) + ')']
                [i / (len(color_list)-1), colors.rgb2hex(color)]
            )
        return plotly_colorscale
    elif color_output == 'list':
        newcmap = colors.ListedColormap(cmap(shift_index).tolist())
        return newcmap
    elif color_output == 'cmap':
        newcmap = colors.ListedColormap(cmap(shift_index))
        return newcmap