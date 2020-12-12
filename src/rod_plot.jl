import PyPlot

function plot(r::basic_rod)

    # Setup quiver points
    xd = similar(r.d)
    xd[:,1] = r.x[:,1]
    xd[:,end] = r.x[:,end]
    xd[:,2:end-1] = 0.5*(r.x[:,2:end-2] + r.x[:,3:end-1])

    # Create a figure and axes
    fig = PyPlot.figure()
    ax = PyPlot.axes(projection="3d")
    ax.plot(r.x[1,:], r.x[2,:], r.x[3,:], "-ok")
    ax.quiver(xd[1,:], xd[2,:], xd[3,:], 
                  r.d[1,:], r.d[2,:], r.d[3,:],
                  length=0.1, normalize=true)
    PyPlot.xlabel("x")
    PyPlot.ylabel("y")
    PyPlot.zlabel("z")
    PyPlot.pygui(true)
    display(fig)
end