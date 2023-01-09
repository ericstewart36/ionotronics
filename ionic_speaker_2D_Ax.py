"""
Code for hydrogel ionotronics.

- with the model comprising:
    > Large deformation compressible Gent elasticity
    > Dilute-solution mixing model for the 
        diffusing cation and anion
        and
    > Electro-quasistatics
    

- with the numerical degrees of freedom:
    > vector displacements
    > scalar electrochemical potentials
    > scalar electrostatic potential.
    
- with basic units:
    > Length: mu-m
    >   Time: mu-s
    >   Mass: mg
    >  Moles: n-mol
    > Charge: mC
    
    Eric M. Stewart    and    Sooraj Narayan,   
   (ericstew@mit.edu)        (soorajn@mit.edu)     
    
                   Fall 2022 
                  
   
Version history: 
    - v1, 10/26/22. Initial version for paper submission to JMPS.
    - v2, 12/14/22. Cleaned up extraneous features, added extra comments.
             
Code acknowledgments:
    
    - Jeremy Bleyer, whose `elastodynamics' code was a useful reference for 
      constructing our own implicit dynamic solution procedure.
     (https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html)
     
  
"""



# Fenics-related packages
from dolfin import *
import numpy as np
import math
# Plotting packages
import matplotlib.pyplot as plt
# Current time package
from datetime import datetime


# Set level of detail for log messages (integer)
# 
# Guide:
# CRITICAL  = 50, // errors that may lead to data corruption
# ERROR     = 40, // things that HAVE gone wrong
# WARNING   = 30, // things that MAY go wrong later
# INFO      = 20, // information of general interest (includes solver info)
# PROGRESS  = 16, // what's happening (broadly)
# TRACE     = 13, // what's happening (in detail)
# DBG       = 10  // sundry
set_log_level(30)


#-----------------------------------------------------------
# Global Fenics parameters
# parameters["form_compiler"]["cpp_optimize"]=True
# parameters["form_compiler"]["optimize"] = True
# set_log_level(30)

# The behavior of the form compiler FFC can be adjusted by prescribing
# various parameters. Here, we want to use the UFLACS backend of FFC::

# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["quadrature_degree"] = 2

'''''''''''''''''''''
DEFINE GEOMETRY
'''''''''''''''''''''

# Dimensions
scaleX = 5.0e4
xElem = 100

scaleY = 900.e0 

# N number of elements in y-direction     
int1 = 200.e0
int2 = scaleY/2.-int1
int3 = int1/2.0 + int2
                
N = 124

M1 = 60 #N/scaleY*int1
M2 = 2.0 #90 #int2/scaleY*N
M3 = M1/2
r1 = 1/1.5
r2 = 1/1.06     
r3 = r1
a1 = (1-r1)/(1-r1**M1)
a2 = (1-r2)/(1-r2**(M2)) 
a3 = (1-r3)/(1-r3**M3)
                
preMapLength = float(int1 + M2*(int1/M1))

mesh = RectangleMesh(Point(0.,-preMapLength),Point(scaleX, preMapLength),xElem,N)

xOrig = mesh.coordinates()
xMap1 = np.zeros((len(xOrig),2))
xMap2 = np.zeros((len(xOrig),2))
xMap3 = np.zeros((len(xOrig),2))

slope = 0.00

for i in range(0,len(xMap1)):

    xMap1[i,0] = xOrig[i,0] 
      
    if np.abs(xOrig[i,1]) < (preMapLength - int1)+1.e-8:
        xMap1[i,1] = (int2/int1)*(M1/M2)*xOrig[i,1] 
    else:
        xMap1[i,1] = xOrig[i,1] + np.sign(xOrig[i,1])*((int2/int1)*(M1/M2)- 1.0)*(preMapLength - int1)

for i in range(0,len(xMap2)):

    xMap2[i,0] = xMap1[i,0] 
      
    if np.abs(xMap1[i,1]) > int2 and np.abs(xMap1[i,1]) < int3 + 1e-8:
        xMap2[i,1] = np.sign(xMap1[i,1])*(int3-(int3-int2)*(a3*(r3**((int3-np.abs(xMap1[i,1]))/(int3-int2)*M3)-1)/(r3-1)))
    elif np.abs(xMap1[i,1]) < int2+1.e-8:
        xMap2[i,1] = xMap1[i,1] #np.sign(xOrig[i,1])*(int2*(a2*(r2**(np.abs(xOrig[i,1])/int2*(M2))-1)/(r2-1)))
        #xMap2[i,0] = xMap1[i,0]*2.0
    else:
        xMap2[i,1] = np.sign(xMap1[i,1])*(int3 + (scaleY/2 - int3)*(a3*(r3**((np.abs(xMap1[i,1])-int3)/(scaleY/2 - int3)*M3)-1)/(r3-1))) 
    
for i in range(0,len(xMap3)):

    xMap3[i,0] = xMap2[i,0] 
    xMap3[i,1] = xMap2[i,1] + (slope*scaleY*(xMap2[i,0]-scaleX)/scaleX)

mesh.coordinates()[:] = xMap3

# This says "spatial coordinates" but is really the referential coordinates,
# since the mesh does not convect in FEniCS.
x = SpatialCoordinate(mesh) 

'''''''''''''''''''''
     SUBDOMAINS
'''''''''''''''''''''

#################### ##################################################

#Pick up on the boundary entities of the created mesh
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0) and on_boundary
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1] - (slope*scaleY*(x[0]-scaleX)/scaleX),-scaleY/2.) and on_boundary
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],scaleX) and on_boundary
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1]- (slope*scaleY*(x[0]-scaleX)/scaleX),scaleY/2) and on_boundary  
class Center(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1]- (slope*scaleY*(x[0]-scaleX)/scaleX),0.)  
         
# Dirichlet boundary
# Mark boundary subdomians
facets = MeshFunction("size_t", mesh, 1)
facets.set_all(0)
DomainBoundary().mark(facets, 1)  # First, mark all boundaries with common index
# Next mark sepcific boundaries
Left().mark(facets, 2)
Bottom().mark(facets, 3)
Top().mark(facets,6)
Right().mark(facets,7)
Center().mark(facets,8)
ds = Measure('ds', domain=mesh, subdomain_data=facets)

tol = DOLFIN_EPS
class Omega_0(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= -int2 + tol

class Omega_1(SubDomain):
    def inside(self, x, on_boundary):
        return ((x[1] >= -int2 - tol) and (x[1]<= int2 + tol))
    
class Omega_2(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] >= int2 - tol)  
    
class Omega_3(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] >= int2 - tol) and (x[1]<= int2 + 0.2*int1)  
  
materials = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
subdomain_0 = Omega_0()
subdomain_1 = Omega_1()
subdomain_2 = Omega_2()
subdomain_3 = Omega_3()
subdomain_0.mark(materials, 0)
subdomain_1.mark(materials, 1)
subdomain_2.mark(materials, 2)
subdomain_3.mark(materials, 3)

dx = Measure('dx', domain=mesh, subdomain_data=materials)

"""
Userdefined expression for defining different materials
"""
class mat(UserExpression): 
    def __init__(self, materials, mat_0, mat_1, mat_2, **kwargs):
        super().__init__(**kwargs)
        self.materials = materials
        self.k_0 = mat_0
        self.k_1 = mat_1
        self.k_2 = mat_2
        
    def eval_cell(self, values, x, cell):
        if self.materials[cell.index] == 0:
            values[0] = self.k_0
        elif self.materials[cell.index] == 1:
            values[0] = self.k_1
        else:
            values[0] = self.k_2
            
    def value_shape(self):
        return ()        
    

'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''

# Material parameters    
# M-mass; L-length; T-time; #-number of moles; Q-charge; K- temperature

matInd = mat(materials, Constant(1.), Constant(0.), Constant(1.))
Gshear = mat(materials, Constant(0.003e-6), Constant(0.034e-6), Constant(0.003e-6), degree=0)  
Kbulk = mat(materials, Constant(2000*0.003e-6), Constant(2000*0.034e-6), Constant(2000.0*0.003e-6), degree=0)  
Gshear0 = 100.0e-6 
Im_gent = mat(materials, Constant(300), Constant(90.0), Constant(300), degree=0)
    
D = 1.e-2                    # Diffusivity [L2T-1]
RT = 8.3145e-3*(273.0+20.0)  # Gas constant*Temp [ML2T-2#-1]
Farad = 96485.e-6            # Faraday constant [Q#-1]

# Initial concentrations
cPos0 = 0.274              # Initial concentration [#L-3]
cNeg0 = cPos0              # Initial concentration [#L-3]
cMax = 10000*1e-9 

vareps0    = Constant(8.85e-12*1e-6)
vareps_num =  mat(materials, Constant(1.0e4), Constant(1.0), Constant(1.0e4))
vareps_r   = mat(materials, Constant(80), Constant(6.5), Constant(80))
vareps     = vareps0*vareps_r*vareps_num

# Mass density
rho = Constant(1e-9) # 1e3 kg/m^3 = 1e-9 mg/um^3,

# Generalized-alpha method parameters
alpha = Constant(0.2) # use moderate \alpha here to stabilize dynamic simulation
gamma   = Constant(0.5+alpha)
beta    = Constant((gamma+0.5)**2/4.)


#Simulation time related params (reminder: in microseconds)
ttd  = 0.01
# Step in time
t = 0.0         # initialization of time
#
T2_tot = 1*0.5e6 
dt = T2_tot/500

phi_norm = RT/Farad # "Thermal" Volt

'''''''''''''''''''''
   FUNCTION SPACES
'''''''''''''''''''''

# Define function space, scalar
U2 = VectorElement("Lagrange", mesh.ufl_cell(), 1)
P1 = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
D0 = FiniteElement("DG", mesh.ufl_cell(), 0)
D1 = FiniteElement("DG", mesh.ufl_cell(), 1)

# DOFs
TH = MixedElement([U2, P1, P1, P1])
ME = FunctionSpace(mesh, TH) # Total space for all DOFs

W = FunctionSpace(mesh,P1)   # Scalar space for visualization later
W2 = FunctionSpace(mesh,U2)   # Vector space for visualization later
W3 = FunctionSpace(mesh,D0)   # DG space for visualization later
W4 = FunctionSpace(mesh,D1)   # DG space for visualization later

# Define test functions in weak form
dw = TrialFunction(ME)                                   
(u_test, omgPos_test, phi_test, omgNeg_test)  = TestFunctions(ME)    # Test function

# Define actual functions with the required DOFs
w = Function(ME)
(u, omgPos, phi, omgNeg) = split(w)    # current DOFs

# A copy of functions to store values in last step for time-stepping.
w_old = Function(ME)
(u_old, omgPos_old, phi_old, omgNeg_old) = split(w_old)   # old DOFs

v_old = Function(W2)
a_old = Function(W2)

# Initial chemical potential
mu0 = ln(cPos0)
mu20 = ln(cNeg0)

# Update initial ("old") values for electrochemical potential
init_omgPos = Expression('abs(x[1])>=int2-tol?std::log((cPos0)):std::log((cNum))', int2=int2, tol = tol, cPos0 = cPos0, cNum=cPos0/1e3, degree=0)
omgPos_init = interpolate(init_omgPos,ME.sub(1).collapse())
assign(w_old.sub(1),omgPos_init)
#
init_omgNeg = Expression('abs(x[1])>=int2-tol?std::log((cNeg0)):std::log((cNum))', int2=int2, tol = tol, cNeg0 = cNeg0,  cNum=cPos0/1e3, degree=0)
omgNeg_init = interpolate(init_omgNeg,ME.sub(3).collapse())
assign(w_old.sub(3),omgNeg_init)

# Update initial guess for w to help the solver a bit
assign(w.sub(3),omgNeg_init)
assign(w.sub(1),omgPos_init)

'''''''''''''''''''''
     SUBROUTINES
'''''''''''''''''''''

# Quick-calculate sub-routines
    
# Gradient of vector field u   
def ax_grad_vector(u):
    grad_u = grad(u)
    return as_tensor([[grad_u[0,0], grad_u[0,1], 0],
                  [grad_u[1,0], grad_u[1,1], 0],
                  [0, 0, u[0]/x[0]]]) 
#
# Gradient of scalar field y
# (just need an extra zero for dimensions to work out)
def ax_grad_scalar(y):
    grad_y = grad(y)
    return as_vector([grad_y[0], grad_y[1], 0.])


# Axisymmetric deformation gradient 
def F_ax_calc(u):
    dim = len(u)
    Id = Identity(dim)          # Identity tensor
    F = Id + grad(u)            # 2D Deformation gradient
    F33 = (x[0]+u[0])/x[0]      # axisymmetric F33, R/R0    
    return as_tensor([[F[0,0], F[0,1], 0],
                  [F[1,0], F[1,1], 0],
                  [0, 0, F33]]) # Full axisymmetric F


def Piola(F,phi):
    
    Id = Identity(3)
    J = det(F)
    
    C = F.T*F
    Cdis = J**(-2/3)*C
    I1 = tr(Cdis)
         
    eR = -phi_norm*ax_grad_scalar(phi)
    e_sp = inv(F.T)*eR
    #
    T_max = vareps0*vareps_r*J*(outer(e_sp,e_sp) - 1/2*(inner(e_sp,e_sp))*Id)*inv(F.T) 
    
    # Piola stress (Gent)
    TR = J**(-2/3)*Gshear*(Im_gent/(Im_gent+3-I1))*(F - 1/3*tr(Cdis)*inv(F.T))\
        + Kbulk*ln(J)*inv(F.T) + T_max
    
    return TR

# variable time step
dk = Constant(0.0)

# Update formula for acceleration
# a = 1/(2*beta)*((u - u0 - v0*dt)/(0.5*dt*dt) - (1-2*beta)*a0)
def update_a(u, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dk
        beta_ = beta
    else:
        dt_ = float(dk)
        beta_ = float(beta)
    return (u-u_old-dt_*v_old)/beta_/dt_**2 - (1-2*beta_)/2/beta_*a_old

# Update formula for velocity
# v = dt * ((1-gamma)*a0 + gamma*a) + v0
def update_v(a, u_old, v_old, a_old, ufl=True):
    if ufl:
        dt_ = dk
        gamma_ = gamma
    else:
        dt_ = float(dk)
        gamma_ = float(gamma)
    return v_old + dt_*((1-gamma_)*a_old + gamma_*a)

def update_fields(u_proj, u_proj_old, v_old, a_old):
    """Update fields at the end of each time step."""

    # Get vectors (references)
    u_vec, u0_vec  = u_proj.vector(), u_proj_old.vector()
    v0_vec, a0_vec = v_old.vector(), a_old.vector()

    # use update functions using vector arguments
    a_vec = update_a(u_vec, u0_vec, v0_vec, a0_vec, ufl=False)
    v_vec = update_v(a_vec, u0_vec, v0_vec, a0_vec, ufl=False)

    # Update (u_old <- u)
    v_old.vector()[:], a_old.vector()[:] = v_vec, a_vec
    #u_old.vector()[:] = u_proj.vector()

def ppos(x):
    return (x+abs(x))/2.

def avg(x_old, x_new, alpha):
    return alpha*x_old + (1-alpha)*x_new


'''''''''''''''''''''''''''''''''''''''''
  KINEMATICS & CONSTITUTIVE RELATIONS
'''''''''''''''''''''''''''''''''''''''''

# Residual computations
a_new = update_a(u, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

# get avg fields for generalized-alpha method
u_avg = avg(u_old, u, alpha)
v_avg = avg(v_old, v_new, alpha)
omgPos_avg = avg(omgPos_old, omgPos, alpha)
phi_avg    = avg(phi_old, phi, alpha)
omgNeg_avg = avg(omgNeg_old, omgNeg, alpha)

# Explicit concentration updates
cPos     = exp(omgPos_avg - Farad*phi_avg*phi_norm/RT)
cNeg     = exp(omgNeg_avg + Farad*phi_avg*phi_norm/RT)
cPos_old = exp(omgPos_old - Farad*phi_old*phi_norm/RT)
cNeg_old = exp(omgNeg_old + Farad*phi_old*phi_norm/RT)


# Kinematics
F = F_ax_calc(u_avg)
C = F.T*F
Ci = inv(C)
F_old = F_ax_calc(u_old) 
J = det(F)
J_old = det(F_old)

'''''''''''''''''''''''
       WEAK FORMS
'''''''''''''''''''''''

# Turn on implicit dynamics
dynSwitch = Constant(1.0)

L0 = inner(Piola(F_ax_calc(u_avg), phi_avg), ax_grad_vector(u_test))*x[0]*dx \
    + dynSwitch*rho*inner(a_new, u_test)*x[0]*dx 
   
L1 = dot((cPos-cPos_old)/dk,omgPos_test)*x[0]*dx \
    + matInd*D*(cPos_old)*inner(Ci*ax_grad_scalar(omgPos_avg),ax_grad_scalar(omgPos_test))*x[0]*dx

L2 = dot(vareps*phi_norm*J*Ci*ax_grad_scalar(phi_avg),ax_grad_scalar(phi_test))*x[0]*dx \
     -matInd*dot(Farad*cMax*(cPos-cNeg),phi_test)*x[0]*dx 

L3 = dot((cNeg-cNeg_old)/dk,omgNeg_test)*x[0]*dx \
    + matInd*D*(cNeg_old)*dot(Ci*ax_grad_scalar(omgNeg_avg),ax_grad_scalar(omgNeg_test))*x[0]*dx
  
eta = 1.5e-14
L_damp = eta*inner(v_avg, u_test)*x[0]*dx

# Total weak form
L = (1/Gshear0)*L0 + L1 + 1e3*L2 + L3  + (1/Gshear0)*L_damp 
#
# Here we multiply L2 by a factor 1e3 purely for numerical purposes;
# this extra factor renders the residual for the electrical problem a similar
# order of magnitude to the mechanical and chemical problems which we found
# helps the solver.


# Automatic differentiation tangent:
a = derivative(L, w, dw)

'''''''''''''''''''''''
BOUNDARY CONDITIONS
''''''''''''''''''''''' 

# Boundary condition definitions
bcs_2 = DirichletBC(ME.sub(2), 0., facets, 6) # Ground top of device
bcs_3 = DirichletBC(ME.sub(2), 0., facets, 3) # Ground bottom of device

bcs_a = DirichletBC(ME.sub(0).sub(0),0.,facets,2)
bcs_b = DirichletBC(ME.sub(0),Constant((0.,0.)),facets,7) # right face built-in

bcs = [bcs_3, bcs_2, bcs_a, bcs_b]

phiRamp = Expression(("3000.e3/phi_norm*(1-exp(-3*t/Tramp))"),
                phi_norm = phi_norm, Tramp = T2_tot, t = 0.0, degree=1)
freq = 30 # Hz
phiRamp2 = Expression(("3000.e3/phi_norm + 1000.0e3/phi_norm*sin(2*pi*f*t/(2*Tramp))"),
                phi_norm = phi_norm, pi=np.pi, f=freq, Tramp = T2_tot, t = 0.0, degree=1)

bcs_f = DirichletBC(ME.sub(2),phiRamp,facets,6) # Ramp up phi on top face
bcs_f2 = DirichletBC(ME.sub(2),phiRamp2,facets,6) # Ramp up phi on top face


bcs2 = [bcs_3, bcs_a, bcs_b, bcs_f]

'''''''''''''''''''''
    RUN ANALYSIS
'''''''''''''''''''''

# Output file setup
file_results = XDMFFile("results/speaker_2D_ax_30Hz.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True

# initialize counter
ii = 0

# Give fields descriptive names
u_v = w_old.sub(0)
u_v.rename("displacement","")

omgPos_v = w_old.sub(1)
omgPos_v.rename("omega(+)", "")

phi_v = w_old.sub(2)
phi_v.rename("phi", "")

omgNeg_v = w_old.sub(3)
omgNeg_v.rename("omega(-)", "")

# Write elastic properties to file at time t=0 to see different materials
Gshear_v = project(1e6*Gshear,W3)
Gshear_v.rename("G0, MPa","")
file_results.write(Gshear_v,t)
#
Kbulk_v = project(1e6*Kbulk,W3)
Kbulk_v.rename("K, MPa","")
file_results.write(Kbulk_v,t)

# Initialize arrays for plotting later
Ny = N+1
x_plot = scaleX
y = np.sort(np.array(mesh.coordinates()[np.where(mesh.coordinates()[:,0] == x_plot),1])) 
y = y[0,:]
y2 = np.linspace(0, scaleY/10, Ny) 

voltage_out = np.zeros(100000)
disp_out = np.zeros(100000)
time_out    = np.zeros(100000)

# function to write results to XDMF at time t
def writeResults(t):
    # Displacement
    file_results.write(u_v,t)
    
    # Convert phi to Volts before saving
    phi_Dim = phi_norm/1.e3*phi
    phi_Dim = project(phi_Dim,W)
    phi_Dim.rename("phi, V","")
    file_results.write(phi_Dim, t)
    
    # Electrochemical potentials
    file_results.write(omgPos_v, t)
    file_results.write(omgNeg_v, t)
        

while (round(t,2) <= round( (3.0 + 2.0/freq*20)*T2_tot + 0.01*1e6,2)):

    # Output storage, also outputs intial state at t=0
    writeResults(t)

    # Constant time-step for ramp-up, 10 ms
    dt = 0.01*1e6
    
    # Conditional for DC versus AC steps of analysis
    if t+dt<=3.0*T2_tot:
        phiRamp.t = t - float(alpha*dt)
    else:
        bcs2 = [bcs_3, bcs_a, bcs_b, bcs_f2]
        dt = 0.01*1e6/freq # new time step, 1/100 of actuation signal period
        phiRamp2.t = t - float(alpha*dt) - 3.0*T2_tot
    
    # Update compiled constant for time step.
    dk.assign(dt)
    # Re-compile the solver to capture when BC setup changes.
    StressProblem = NonlinearVariationalProblem(L, w, bcs2, J=a)
    # Set up the non-linear solver
    solver  = NonlinearVariationalSolver(StressProblem)
    # Solver parameters
    prm = solver.parameters
    prm['nonlinear_solver'] = 'newton'
    prm['newton_solver']['linear_solver'] = 'petsc'  
    prm['newton_solver']['absolute_tolerance'] = 1.E-6
    prm['newton_solver']['relative_tolerance'] = 1.E-6
    prm['newton_solver']['maximum_iterations'] = 60
    
    # Solve the problem
    (iter, converged) = solver.solve()
    
    # Time history outputs
    phi_top = w(x_plot, scaleY/2)[3]*phi_norm
    phi_btm = w(x_plot, -scaleY/2)[3]*phi_norm
    voltage_out[ii] = phi_top - phi_btm
    disp_out[ii]    = w(0.0, scaleY/2 - slope*scaleY)[1]
    time_out[ii]    = t
    
    # Update fields for next step
    u_proj = project(u, W2)
    u_proj_old = project(u_old, W2)
    update_fields(u_proj, u_proj_old, v_old, a_old)
    w_old.vector()[:] = w.vector()
    
    # increment time
    t += dt
    ii = ii + 1
    
    
    # Print progress of calculation
    #if ii%10 == 0:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Step: Actuate   |   Simulation Time: {} s  |     Iterations: {}".format(t/1e6, iter))
    print()
    

# Final step XDMF output
writeResults(t)

# output final measured device voltage
phi_top = w(x_plot, scaleY/2)[3]*phi_norm
phi_btm = w(x_plot,-scaleY/2.)[3]*phi_norm
disp_out[ii] = w(scaleX, scaleY/2 - slope*scaleY)[0]

'''''''''''''''''''''
    VISUALIZATION
'''''''''''''''''''''

# Set up font size, initialize colors array
font = {'size'   : 16}
plt.rc('font', **font)
#
prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

# PLOT 1: Through-thickness electrostatic potential at end of simulation.
phi_plot = [w_old(x_plot, xi)[3] for xi in y]
plt.figure()
fig = plt.gcf()
plt.plot( np.array(phi_plot)*phi_norm/1e3, y + 0.5*scaleY, linewidth=3., c=colors[2])
plt.axhline(-int2 + 0.5*scaleY, c='k', linewidth=1.)
plt.axhline(int2+ 0.5*scaleY, c='k', linewidth=1.)
plt.ylabel("$y$-coordinate, $\mu$m")
plt.xlabel("Electric potential  $\\phi$, V") 

# save figure to file
fig = plt.gcf()
fig.set_size_inches(6, 5)
plt.tight_layout()
plt.savefig("plots/speaker_phi_contour.png", dpi=600)


# PLOT 2: Time histories of voltage and displacement during DC phase.

# only plot as far as time_out has time history for.
ind = np.argmax(time_out) 

# Set up figure
fig, (ax1, ax2) = plt.subplots(2,1, sharex='col')
#
color = 'b'
ax1.set_ylabel(r"$V$ (kV)")
ax1.plot(time_out[0:ind]/1.e6, np.array(voltage_out[0:ind])/1e6,linewidth=3.0, color=color)
ax1.tick_params(axis='y')
ax1.axvline(3.0*T2_tot/1e6, c='k', linewidth=1.)
ax1.set_yticks([0,1,2,3,4])
ax1.set_ylim(-0.05, 4.5)
#
color = 'r'
ax2.set_ylabel(r"$\delta$ (mm)")
ax2.plot(time_out[0:ind]/1.e6, -disp_out[0:ind]/1e3,linewidth=3., color=color)
ax2.tick_params(axis='y')
ax2.axvline(3.0*T2_tot/1e6, c='k', linewidth=1.)
ax2.set_xlabel(r"Time (s)")
ax2.set_xlim(0, 3.0*T2_tot/1e6 - 0.01)
ax2.set_yticks([0,2.5,5,7.5,10])
ax2.set_ylim(-0.05,10.05)

# save figure to file
fig = plt.gcf()
fig.set_size_inches(9, 5)
plt.tight_layout()
plt.savefig("plots/speaker_DC.png", dpi=600)


# PLOT 3: Time histories of voltage and displacement during AC phase.

# Set up figure
fig, (ax1, ax2) = plt.subplots(2,1, sharex='col')
#
color = 'b'
ax1.set_ylabel(r"$V$ (kV)")
ax1.plot(time_out[np.where(time_out>=dt)]/1.e6, np.array(voltage_out[np.where(time_out>=dt)])/1e6,linewidth=3.0, color=color)
ax1.tick_params(axis='y')
ax1.axvline(2.0*T2_tot/1e6, c='k', linewidth=1.)
#
color = 'r'
ax2.set_ylabel(r"$\delta$ (mm)")
ax2.plot(time_out[np.where(time_out>=dt)]/1.e6 , -disp_out[np.where(time_out>=dt)]/1e3,linewidth=3., color=color)
ax2.tick_params(axis='y')
ax2.axvline(3.0*T2_tot/1e6, c='k', linewidth=1.)
ax2.set_xlabel(r"Time (s)")
ax2.set_xlim(3.0*T2_tot/1e6, (3.0+20.0*2.0/freq)*T2_tot/1e6)
ax2.set_yticks([0,2.5,5,7.5,10])
ax2.set_ylim(-0.05,10.05)

# save figure to file
fig = plt.gcf()
fig.set_size_inches(9, 5)
plt.tight_layout()
plt.savefig("plots/speaker_AC.png", dpi=600)
