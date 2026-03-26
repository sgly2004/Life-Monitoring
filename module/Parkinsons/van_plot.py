import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn           as sea

from cycler import cycler
from scipy.stats import norm

color_list = ['#C83232', '#199619', '#0000FF', 
              '#009696', '#BB00BB', '#FFFF00',
              '#000000', '#FFFFFF', '#808080']

def plot_defaults(plot, x_size=10, y_size=10):
    '''
    My canonical graphing style.
    Read van_plot.py for usage information.
    '''
    # plot.xkcd()
    plot.figure(figsize=(x_size,y_size), facecolor='white') 
    plot.rcParams.update({'font.size': 16})
    plot.rcParams['axes.facecolor'] = '#AADDAA'
    plot.rcParams['lines.linewidth'] = 3
    plot.rcParams['lines.color'] = 'red'
    plot.rcParams['axes.prop_cycle'] = cycler('color', color_list)
    plot.grid(True)
    
def plot_labels(plot, x_label, y_label, title):
    plot.xlabel(x_label)
    plot.ylabel(y_label)
    plot.title(title)
    
def plot_limits(plot, mx, px, my, py):
    plot.xlim((mx, px))
    plot.ylim((my, py))
        
def slope_intercept(p0, p1):
    m = (p1[1]-p0[1])/(p1[0]-p0[0])
    b =  p1[1] - m* p1[0]
    return m, b

def inf_line(plot, p0, p1, linewidth=1, alpha=0.5, zorder=1):
    '''
    Plot line from slope and intercept:
    adapted from: http://bit.ly/3bdq9Y7
    '''
    axes = plot.gca()
    m, b = slope_intercept(p0, p1)
    x = np.array(axes.get_xlim())
    y = b + m * x
    plot.plot(x, y,linewidth=linewidth, alpha=alpha, zorder=zorder)

def mb_line(plot, m, b, linewidth=1, alpha=0.5, zorder=1):
    axes = plot.gca()
    x = np.array(axes.get_xlim())
    y = b + m * x
    plot.plot(x, y, linewidth=linewidth, alpha=alpha, zorder=zorder)

def plot_test(plot, x_size=10, y_size=10):  
    plot_defaults(plot, x_size, y_size)
    plot_labels(plot, 'x', 'f(x)', 'f(x) vs. x for some common functions')
    plot_limits(plot, -math.pi, math.pi, -math.pi, math.pi)

    x = np.arange(-math.pi, math.pi, 0.01)
    # print(type(x).__name__, len(x))

    y = 1/x
    y[y >  math.pi] =  np.inf # remove singularity
    y[y < -math.pi] = -np.inf
    plot.plot(x, y)

    y = x*x
    plot.plot(x, y)

    y = norm.pdf(x)
    plot.plot(x, y)

    y = np.sin(x)
    plot.plot(x, y)

    y = np.cos(x)
    plot.plot(x, y)

    y = np.tan(x)
    y[y >  math.pi] =  np.inf # remove singularity
    y[y < -math.pi] = -np.inf
    plot.plot(x, y)

    y = x
    plot.plot(x, y)

    x = np.arange(-math.pi, 0.9*math.pi, 0.01) # miss the signature
    y = -x
    plot.plot(x, y)

    plot.axhline(0, color='#808080')
    plot.axvline(0, color='#808080')
    plot.text(2.1,-3,'van2022')
    plot.legend([  '1/x' ,     'x²', 'pdf(x)', 
                'sin(x)', 'cos(x)', 'tan(x)',
                    'x' ,    '-x' ], title='   f(x)', loc=1)

    plot.show();

def plot_hist(plot, y, a, b, columns, alpha=0.5, x_title=''):
    
    plot_defaults(plot, x_size=6, y_size=4)
    plot_labels(plot,
                    x_title,
                    'Number of Instances',
                    'Number of Instances vs ' + x_title)

    plot.text(0.9, -10.0,'van2021')

    [plot.hist(y[i], bins=30, alpha=alpha, label='foo') for i in range(0,b-a)]

    plot.legend(columns, title='Feature', loc=1)
#   plot.axhline(0, color='#000000')
#   plot.axvline(0, color='#000000')
    plot.grid(False)
    plot.show();

def plot_ROC(plot, name, fpr, tpr, threshold): # plot receiver operating characteristic
    plot_defaults(plot, x_size=3.75, y_size=3.75)
    plot_labels(plot,
                    'False Positive Rate',
                    'True  Positive Rate', 
                    'Receiver Operating Characteristic')
    plot.plot(fpr ,tpr) # ROC curve
    ident = [0.0, 1.0]
    plot.plot(ident,ident)
    auc = np.trapz(tpr,fpr) # compute the area using trapezoidal rule
    plot.legend([round(auc,2)], title='AUC', loc=4)
    plot.text(0.9, -0.30,'van2021', fontsize=10)
    plot.text(0.01,  0.90 , name, fontsize=12)
    plot.show()

def plot_CM(plot, name, CM): # plot confusion matrix

    group_names  = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ['{0:0.0f}'.format(value) \
                    for value in CM.flatten()]
    group_percents = ['{0:.2%}'.format(value) \
                      for value in CM.flatten()/np.sum(CM)]
    labels = [f'{v1}\n{v2}\n{v3}' \
              for v1, v2, v3 in zip(group_names,group_counts,group_percents)]
    labels = np.asarray(labels).reshape(2,2)

    plot_defaults(plot, x_size=4.5, y_size=4.5)
    sea.heatmap(CM, annot=labels, fmt='', cmap='Blues')
    plot.show()


'''
To save the image to a file and preview uncomment the next three lines
and comment out the plot.show() line above.

plt.savefig('/tmp/tmpPlot.png', format='png')
!open /tmp/tmpPlot.png
Select Cell, Hit ESC, then Shift-O to prevent cell from scrolling.
'''
