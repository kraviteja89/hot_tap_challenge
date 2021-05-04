from fenics import *
import mshr
import matplotlib.pyplot as plt
from os import mkdir, path
import numpy as np

"""
2D Axisymmetric heat transfer model of welding a cylindrical sleeve to a pipeline

The code organization is as follows:

Step 1 - Define parameters
Step 2 - Create geometry and mesh 
Step 3 - Create boundary subsets
Step 4 - Solve the stationary problem, for initial conditions
Step 5 - Solve the time-dependent problem

The following are the outputs of the program
domains.pvd      - ParaView data file showing the different domains marked 
                    1:pipeline 2:sleeve 3:weld
boundaries.pvd   - ParaView data file showing three boundary subsets marked
                    1:G-process 2:G-insulated 3:G-ambient
hot_weld_sol.pvd - Solutions of the time-dependent problem over the entire mesh  
"""
result_path = "Results"
if not path.exists(result_path):
    mkdir(result_path)

plt.rcParams["figure.figsize"] = (12, 8)

## Step 1 - Define Parameters
# geometric tolerance and time step
tol = 1E-8
T = 10.0
tsteps = 1000
dt = T/tsteps

# geometry parameters
r0 = 10.75 / 2  # inches
t_sleeve = t_wall = 0.188  # inches
t_gap = 0.02  # inches
l_sleeve = 10*t_sleeve
l_wall = 1.5*l_sleeve

# heat parameters
rho = 0.284  # lb/in^3
cp = 0.199  # BTU/lb-F
k = 31.95/3600/12  # BTU/s-in-F
u_process = 325.0  # F
u_amb = 70.0  # F
h_process = 48.0/3600/12/12  # BTU/s-in^2-F
h_amb = 9.0/3600/12/12  # BTU/s-in^2-F
f_max = 270  # BTU/s/in^3
u_melt = 2750  # F
u_haz = 1475  # F

## Step 2: Create Geometry and Mesh
# make geometry
weld_area = mshr.Polygon([Point(r0, 0), Point(r0, -t_sleeve - t_gap), Point(r0 + t_sleeve + t_gap, 0)])
sleeve_area = mshr.Rectangle(Point(r0 + t_gap, 0), Point(r0 + t_sleeve + t_gap, l_sleeve))
pipe_area = mshr.Rectangle(Point(r0 - t_wall, l_sleeve - l_wall), Point(r0, l_sleeve))
total_area = weld_area + sleeve_area + pipe_area

# make subdomain for 1) pipe, 2) sleeve 3) weld
total_area.set_subdomain(1, pipe_area)
total_area.set_subdomain(2, sleeve_area)
total_area.set_subdomain(3, weld_area)

# generate mesh
mesh_density = 64
mesh = mshr.generate_mesh(total_area, mesh_density)

# mark different domains and redefine the measure dx
domain_markers = MeshFunction('size_t', mesh, mesh.topology().dim(), mesh.domains())
dx = Measure('dx', domain=mesh, subdomain_data=domain_markers)
domain_file = File(result_path + '//domains.pvd')
domain_file << domain_markers

# Un-comment the next two lines to view the mesh
#plot(mesh, 'title = mesh')
#plt.show()


# Step 3 - make boundary subsets.
# The boundaries will be marked as 1) insulated 2) Process 3) Ambient
class FullBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary


class BoundaryInsulated(SubDomain):
    def inside(self, x, on_boundary):
        if on_boundary:
            # top and bottom faces
            if near(x[1],  l_sleeve, tol) or near(x[1], l_sleeve-l_wall, tol):
                return True
            # internal faces
            elif x[1] >= -tol and (near(x[0], r0, tol) or near(x[0], r0 + t_gap, tol)):
                return True
            # weld internal face
            elif near(x[1], 0, tol) and -tol <= x[0]-r0 <= t_gap + tol:
                return True
        return False


class BoundaryProcess(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and near(x[0], r0 - t_wall, tol)


bx0 = FullBoundary()
bx1 = BoundaryInsulated()
bx2 = BoundaryProcess()
boundary_markers = MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
boundary_markers.set_all(0)
bx0.mark(boundary_markers, 3)
bx1.mark(boundary_markers, 1)
bx2.mark(boundary_markers, 2)

# save boundary IDs to file
boundary_file = File(result_path + '//boundaries.pvd')
boundary_file << boundary_markers

# redefine the measure ds
ds = Measure('ds', domain=mesh, subdomain_data=boundary_markers)

## Step 4- initial solution
t = 0.0

# create function space
V = FunctionSpace(mesh, 'P', 1)
u = TrialFunction(V)
v = TestFunction(V)

# define the bilinear and linear parts of the equation
axsym1 = Expression('2*pi*x[0]', degree=2)
axsym = project(axsym1, V)

a = axsym*k*dot(grad(u), grad(v))*dx + axsym*h_amb*u*v*ds(3) + axsym*h_process*u*v*ds(2)
L = axsym*h_amb*u_amb*v*ds(3) + axsym*h_process*u_process*v*ds(2)

# solve for initial condition
u = Function(V)
u_n = Function(V)

solve(a == L, u)
u_n.assign(u)

# write out the inital solution
solution_file = File(result_path + '//hot_weld_solution.pvd')
solution_file << (u_n, 0)

## Step 5 - Time dependent solution

# Redefine the linear and bilinear parts for the time-dependent problem
f = Expression('t <= 10.0 ? f_max*(0.5 + 0.5*cos(pi*(5+t)/5)) : 0.0',
               degree=2, t=t, f_max=f_max)
u = TrialFunction(V)
v = TestFunction(V)

a = axsym*rho*cp*u*v*dx + axsym*dt*k*dot(grad(u), grad(v))*dx + \
    axsym*dt*h_amb*u*v*ds(3) + axsym*dt*h_process*u*v*ds(2)
L = axsym*rho*cp*u_n*v*dx + axsym*dt*f*v*dx(3) +\
    axsym*dt*h_amb*u_amb*v*ds(3) + axsym*dt*h_process*u_process*v*ds(2)

u = Function(V)

# Define points to evaluate temperature
eval_pts = 5
pipe_rvals = np.linspace(r0, r0 - t_wall, eval_pts)
sleeve_zvals = np.linspace(0, t_wall, eval_pts)
p_pipe = [(val, -0.5*(t_wall + t_gap)) for val in pipe_rvals]
p_sleeve = [(r0 + t_gap + 0.5*t_wall, val) for val in sleeve_zvals]
p_weld = (r0 + 0.5*t_sleeve, -0.5*t_sleeve)

# set up output variables
dt_output = 0.1
output_steps = int(dt_output/dt)
sleeve_temp = np.array([u_n(point) for point in p_sleeve])
pipe_temp = np.array([u_n(point) for point in p_pipe])
weld_temp = [u_n(p_weld)]
time_array = np.linspace(0, T, int(T/dt_output) + 1)
im_number = 1

# Run the time-dependent analysis
for n in range(tsteps):

    t += dt
    f.t = t

    solve(a == L, u)

    u_n.assign(u)

    if (n+1) % output_steps == 0:
        # output solution to paraview file
        solution_file << (u_n, t)

        # find the temperature at the specified points
        sleeve_temp1 = np.array([u_n(point) for point in p_sleeve])
        pipe_temp1 = np.array([u_n(point) for point in p_pipe])
        sleeve_temp = np.vstack((sleeve_temp, sleeve_temp1))
        pipe_temp = np.vstack((pipe_temp, pipe_temp1))
        weld_temp.append(u_n(p_weld))

        # create plots
        fig, ax = plt.subplots(2, 2)
        c = plot(u, title='Time = %.2fs' % t, mode='color', vmin=u_amb, vmax=u_melt)
        plt.colorbar(c, label='Temperature(F)')
        plt.ylim([-2*t_wall, t_wall])
        plt.xlabel('r (in)')
        plt.ylabel('z (in)')

        for n1 in range(eval_pts):
            ax[0, 1].plot(time_array[:pipe_temp.shape[0]], pipe_temp[:, n1],
                          label='depth = %.2f' % (r0-pipe_rvals[n1]))
            ax[0, 0].plot(time_array[:pipe_temp.shape[0]], sleeve_temp[:, n1],
                          label='depth = %.2f' % sleeve_zvals[n1])
        for n2 in range(2):
            ax[0, n2].plot(time_array, np.ones_like(time_array)*u_melt, 'k--',
                           label='melting point')
            ax[0, n2].plot(time_array, np.ones_like(time_array) * u_haz, 'r--',
                           label='HAZ cutoff')
            ax[0, n2].legend(loc='upper left')
            ax[0, n2].set_ylim([0, 1.4 * u_melt])
            ax[0, n2].set_xlim([0, T])
            ax[0, n2].set_xlabel('Time (s)')

        ax[1, 0].plot(time_array[:pipe_temp.shape[0]], weld_temp, label='Weld temperature')
        ax[1, 0].plot(time_array, np.ones_like(time_array) * u_melt, 'k--',
                       label='melting point')
        ax[1, 0].plot(time_array, np.ones_like(time_array) * u_haz, 'r--',
                       label='HAZ cutoff')
        ax[1, 0].legend(loc='upper left')
        ax[1, 0].set_ylim([0, 1.8 * u_melt])
        ax[1, 0].set_xlim([0, T])
        ax[1, 0].set_xlabel('Time (s)')

        ax[1, 0].set_title('Weld Temperature(F)')
        ax[0, 1].set_title('Temperatures in pipe(F)')
        ax[0, 0].set_title('Temperatures in sleeve(F)')
        plt.tight_layout()
        plt.savefig(result_path + "//linear_model%0*d.png" % (4, im_number), dpi=96)
        im_number += 1
        plt.close()





