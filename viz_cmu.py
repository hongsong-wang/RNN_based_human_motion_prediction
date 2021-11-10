import numpy as np

def _some_variables():
  """
  We define some variables that are useful to run the kinematic tree

  Args
    None
  Returns
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  """

  parent = np.array([0, 1, 2, 3, 4, 5, 6,1, 8, 9,10, 11,12, 1, 14,15,16,17,18,19, 16,
                    21,22,23,24,25,26,24,28,16,30,31,32,33,34,35,33,37])-1

  offset = 70*np.array([0,0	,0	,0,	0,	0,	1.65674000000000,	-1.80282000000000,	0.624770000000000,	2.59720000000000,	-7.13576000000000,	0,	2.49236000000000,	-6.84770000000000,	0,	0.197040000000000,	-0.541360000000000,	2.14581000000000,	0,	0,	1.11249000000000,	0,	0,	0,	-1.61070000000000,	-1.80282000000000,	0.624760000000000,	-2.59502000000000,	-7.12977000000000,	0,	-2.46780000000000,	-6.78024000000000,	0,	-0.230240000000000,	-0.632580000000000,	2.13368000000000,	0,	0,	1.11569000000000,	0,	0,	0,	0.0196100000000000,	2.05450000000000,	-0.141120000000000,	0.0102100000000000,	2.06436000000000,	-0.0592100000000000,	0,	0,0,	0.00713000000000000,	1.56711000000000,	0.149680000000000,	0.0342900000000000,	1.56041000000000,	-0.100060000000000,	0.0130500000000000,	1.62560000000000,	-0.0526500000000000,	0,	0,	0,	3.54205000000000,	0.904360000000000,	-0.173640000000000,	4.86513000000000,	0,	0,	3.35554000000000,	0,	0	,0	,0	,0	,0.661170000000000,	0,	0,	0.533060000000000,	0,	0	,0	,0	,0	,0.541200000000000	,0	,0.541200000000000,	0	,0	,0	,-3.49802000000000,	0.759940000000000,	-0.326160000000000,	-5.02649000000000	,0	,0,	-3.36431000000000,	0,0,	0,	0	,0	,-0.730410000000000,	0,	0	,-0.588870000000000,0	,0,	0,	0	,0	,-0.597860000000000	,0	,0.597860000000000])
  offset = offset.reshape(-1,3)

  rotInd = [[6, 5, 4],
            [9, 8, 7],
            [12, 11, 10],
            [15, 14, 13],
            [18, 17, 16],
            [21, 20, 19],
            [],
            [24, 23, 22],
            [27, 26, 25],
            [30, 29, 28],
            [33, 32, 31],
            [36, 35, 34],
            [],
            [39, 38, 37],
            [42, 41, 40],
            [45, 44, 43],
            [48, 47, 46],
            [51, 50, 49],
            [54, 53, 52],
            [],
            [57, 56, 55],
            [60, 59, 58],
            [63, 62, 61],
            [66, 65, 64],
            [69, 68, 67],
            [72, 71, 70],
            [],
            [75, 74, 73],
            [],
            [78, 77, 76],
            [81, 80, 79],
            [84, 83, 82],
            [87, 86, 85],
            [90, 89, 88],
            [93, 92, 91],
            [],
            [96, 95, 94],
            []]
  posInd=[]
  for ii in np.arange(38):
      if ii==0:
          posInd.append([1,2,3])
      else:
          posInd.append([])


  expmapInd = np.split(np.arange(4,118)-1,38)

  return parent, offset, posInd, expmapInd

class Ax3DPose(object):
  def __init__(self, ax, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Create a 3d pose visualizer that can be updated with new poses.

    Args
      ax: 3d axis to plot the 3d pose on
      lcolor: String. Colour for the left part of the body
      rcolor: String. Colour for the right part of the body
    """

    # Start and endpoints of our representation

    self.I = np.array(
        [1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17, 18, 19, 16, 21, 22, 23, 25, 26, 24, 28, 16, 30, 31,
         32, 33, 34, 35, 33, 37]) - 1
    self.J = np.array(
        [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27, 28, 29, 30, 31, 32,
         33, 34, 35, 36, 37, 38]) - 1
    # Left / right indicator
    self.LR = np.array(
        [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        dtype=bool)
    self.ax = ax

    vals = np.zeros((38, 3))

    # Make connection matrix
    self.plots = []
    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots.append(self.ax.plot(x, y, z, lw=2, c=lcolor if self.LR[i] else rcolor))

    self.ax.set_xlabel("x")
    self.ax.set_ylabel("y")
    self.ax.set_zlabel("z")

  def update(self, channels, lcolor="#3498db", rcolor="#e74c3c"):
    """
    Update the plotted 3d pose.

    Args
      channels: 96-dim long np array. The pose to plot.
      lcolor: String. Colour for the left part of the body.
      rcolor: String. Colour for the right part of the body.
    Returns
      Nothing. Simply updates the axis with the new pose.
    """

    assert channels.size == 114, "channels should have 96 entries, it has %d instead" % channels.size
    vals = np.reshape( channels, (38, -1) )

    for i in np.arange( len(self.I) ):
      x = np.array( [vals[self.I[i], 0], vals[self.J[i], 0]] )
      y = np.array( [vals[self.I[i], 1], vals[self.J[i], 1]] )
      z = np.array( [vals[self.I[i], 2], vals[self.J[i], 2]] )
      self.plots[i][0].set_xdata(x)
      self.plots[i][0].set_ydata(y)
      self.plots[i][0].set_3d_properties(z)
      self.plots[i][0].set_color(lcolor if self.LR[i] else rcolor)

    r = 1000;
    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    self.ax.set_xlim3d([-r+xroot, r+xroot])
    self.ax.set_zlim3d([-r+zroot, r+zroot])
    self.ax.set_ylim3d([-r+yroot, r+yroot])

    self.ax.set_aspect('equal')