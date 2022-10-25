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
                   
   
Note: This code will yield revelant results from the ionotronics paper, but is
  not yet fully cleaned up. In the near future I plan to make edits to render 
  it more readable, remove extraneous features, and provide more detailed 
  explanatory comments.

                   
Code acknowledgments:
    
    - Jeremy Bleyer, whose `elastodynamics' code was a useful reference for 
      constructing our own implicit dynamic solution procedure.
     (https://comet-fenics.readthedocs.io/en/latest/demo/elastodynamics/demo_elastodynamics.py.html)
         
  
"""


# FEniCS package
from dolfin import *
# NumPy for arrays and array operations
import numpy as np
# MatPlotLib for plotting
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
#
set_log_level(30)

# Global FEniCS parameters:
# Optimization options for the form compiler
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["representation"] = "uflacs"
parameters["form_compiler"]["cpp_optimize_flags"] = "-O3 -ffast-math -march=native"
parameters["form_compiler"]["quadrature_degree"] = 2

'''''''''''''''''''''
DEFINE GEOMETRY
'''''''''''''''''''''

# Create a uniform box mesh and map the coordinates to achieve 
# the desired refinement of the electrochemical boundary layers.

# Overall dimensions of rectangular prism device
scaleX = 2.0e4   # 2 cm
scaleY = 900.e0  # 0.9 mm
scaleZ = 1.0e4   # 1 cm

# N number of elements in y-direction    
int1 = 200.e0 # 
int2 = scaleY/2.-int1
int3 = int1/2.0 + int2
                
N = 124

# parameters used to define the bias strategy to refine the mesh near the interfaces.
M1 = 60 
M2 = 2.0 
M3 = M1/2
r1 = 1/1.5
r2 = 1/1.06     
r3 = r1
a1 = (1-r1)/(1-r1**M1)
a2 = (1-r2)/(1-r2**(M2)) 
a3 = (1-r3)/(1-r3**M3)
             
# Thickness of the pre-mapping mesh   
preMapLength = float(int1 + M2*(int1/M1))

# Define a uniformly spaced box mesh
mesh = BoxMesh(Point(0.,-preMapLength, 0.0),Point(scaleX, preMapLength, scaleZ),1,N, 1)


# Map the coordinates of the uniform box mesh to the biased spacing
xOrig = mesh.coordinates()
xMap1 = np.zeros((len(xOrig),3))
xMap2 = np.zeros((len(xOrig),3))

# Mapping functions, evaluated simply using for-loops and conditionals.
for i in range(0,len(xMap1)):

    xMap1[i,0] = xOrig[i,0] 
    xMap1[i,2] = xOrig[i,2] 
      
    if np.abs(xOrig[i,1]) < (preMapLength - int1)+1.e-8:
        xMap1[i,1] = (int2/int1)*(M1/M2)*xOrig[i,1] 
    else:
        xMap1[i,1] = xOrig[i,1] + np.sign(xOrig[i,1])*((int2/int1)*(M1/M2)- 1.0)*(preMapLength - int1)

for i in range(0,len(xMap2)):

    xMap2[i,0] = xMap1[i,0] 
    xMap2[i,2] = xMap1[i,2] 
      
    if np.abs(xMap1[i,1]) > int2 and np.abs(xMap1[i,1]) < int3 + 1e-8:
        xMap2[i,1] = np.sign(xMap1[i,1])*(int3-(int3-int2)*(a3*(r3**((int3-np.abs(xMap1[i,1]))/(int3-int2)*M3)-1)/(r3-1)))
    elif np.abs(xMap1[i,1]) < int2+1.e-8:
        xMap2[i,1] = xMap1[i,1]
    else:
        xMap2[i,1] = np.sign(xMap1[i,1])*(int3 + (scaleY/2 - int3)*(a3*(r3**((np.abs(xMap1[i,1])-int3)/(scaleY/2 - int3)*M3)-1)/(r3-1))) 

# Over-write the original mesh coordinates with the mapped coordinates
mesh.coordinates()[:] = xMap2

# Store the coordinates for future use
x = SpatialCoordinate(mesh) 


'''''''''''''''''''''
     SUBDOMAINS
'''''''''''''''''''''

#----------------------------------------------------------
# Define the mesh subdomains, used for applying BCs and 
#  spatially-varying material properties.


#Pick up on the boundary entities of the created mesh
class Left(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],0) and on_boundary
class Bottom(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],-scaleY/2.) and on_boundary
class Right(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[0],scaleX) and on_boundary
class Top(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[1],scaleY/2) and on_boundary  
class Front(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2],scaleZ) and on_boundary
class Back(SubDomain):
    def inside(self, x, on_boundary):
        return near(x[2],0.) and on_boundary


# Mark boundary subdomians
facets = MeshFunction("size_t", mesh, 2)
facets.set_all(0)
DomainBoundary().mark(facets, 1)  # First, mark all boundaries with common index

# Next mark sepcific boundaries
Left().mark(facets, 2)
Bottom().mark(facets, 3)
Top().mark(facets,6)
Right().mark(facets,7)
Front().mark(facets,4)
Back().mark(facets, 5)

# Define a ds measure for each face, necessary for applying traction BCs.
ds = Measure('ds', domain=mesh, subdomain_data=facets)

# Define functions for each volumetric subdomain for spatially-varying
#  material properties.
tol = 1e-12
class Omega_0(SubDomain):
    def inside(self, x, on_boundary):
        return x[1] <= -int2 + tol

class Omega_1(SubDomain):
    def inside(self, x, on_boundary):
        return ((x[1] >= -int2 - tol) and (x[1]<= int2 + tol))
    
class Omega_2(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] >= int2 - tol)  
    
# this last volumetric sub-domain is the ``pill-box'' subdomain, 
#   which extends into 20% of a layer thickness of the upper I.H.
class Omega_3(SubDomain):
    def inside(self, x, on_boundary):
        return (x[1] >= int2 - tol) and (x[1]<= int2 + 0.2*int1)
  
# index the volumetric subdomains.
materials = MeshFunction("size_t", mesh, mesh.topology().dim(), 0)
subdomain_0 = Omega_0()
subdomain_1 = Omega_1()
subdomain_2 = Omega_2()
subdomain_3 = Omega_3()
subdomain_0.mark(materials, 0)
subdomain_1.mark(materials, 1)
subdomain_2.mark(materials, 2)
subdomain_3.mark(materials, 3)

# Define different volume measures for integration.
dx = Measure('dx', domain=mesh, subdomain_data=materials)


# User-defined expression for spatially-varing material properties.
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
    
# Index which evaluates to 1 in the ``pill-box'' domain, and 0 elsewhere.
class integralIndex(UserExpression): 
    def __init__(self, materials, intInd_0, intInd_1, **kwargs):
        super().__init__(**kwargs)
        self.materials = materials
        self.k_0 = intInd_0
        self.k_1 = intInd_1
        
    def eval_cell(self, values, x, cell):
        if self.materials[cell.index] == 3:
            values[0] = self.k_0
        else:
            values[0] = self.k_1
            
    def value_shape(self):
        return ()      

intInd = integralIndex(materials, Constant(1.), Constant(0.), degree=0)


'''''''''''''''''''''
MATERIAL PARAMETERS
'''''''''''''''''''''
# M-mass; L-length; T-time; #-number of moles; Q-charge; K- temperature

# matInd defined so that 1 == ionic hydrogel, 0 == dielectric elastomer.
matInd = mat(materials, Constant(1.), Constant(0.), Constant(1.))

# Elastic constants for the two different materials. 
Gshear = mat(materials, Constant(0.003e-6), Constant(0.034e-6), Constant(0.003e-6), degree=0)  
Kbulk = mat(materials, Constant(2000*0.003e-6), Constant(2000*0.034e-6), Constant(2000.0*0.003e-6), degree=0)  
Im_gent = mat(materials, Constant(300.0), Constant(90.0), Constant(300.0), degree=0)
Gshear0 = 100.0e-6 # spatially uniform normalization factor for the weak form.

# electrochemical material parameters.    
#
D = 1.e-2                    # Diffusivity [L2T-1]
RT = 8.3145e-3*(273.0+20.0)  # Gas constant*Temp [ML2T-2#-1]
Farad = 96485.e-6            # Faraday constant [Q#-1]
cPos0 = 0.274                # Initial concentration [#L-3]
cNeg0 = cPos0                # Initial concentration [#L-3]
cMax = 10000*1e-9            # Reference concentration [#L-3]
#
# Electrical permittivity
vareps0    = Constant(8.85e-12*1e-6)
vareps_num = mat(materials, Constant(1e4), Constant(1.0), Constant(1e4), degree=1)
vareps_r   = mat(materials, Constant(80), Constant(6.5), Constant(80))
vareps     = vareps0*vareps_r*vareps_num

# Mass density
rho = Constant(1e-9) # 1e3 kg/m^3 = 1e-9 mg/um^3

# alpha-method parameters
alpha   = Constant(0.0) 
gamma   = Constant(0.5+alpha)
beta    = Constant((gamma+0.5)**2/4.)


#Simulation time related params (reminder: in microseconds)
ttd  = 0.01
# Step in time
t = 0.0         # initialization of time
T_tot = 0e6*1.0e6 #200.0 #0.11        # total simulation time 

#

t1 = 15.0e6
t2 = 35.0e6
t3 = 2.5e6
t4 = 52.5e6
T2_tot = 1.0e6 #0.0001*1.e6 #t1+t2+t3+t4
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

# Initial electro-chemical potentials
init_omgPos = Expression('abs(x[1])>=int2-tol?std::log((cPos0)):std::log((cNum))', int2=int2, tol = tol, cPos0 = cPos0, cNum=DOLFIN_EPS, degree=0)
omgPos_init = interpolate(init_omgPos,ME.sub(1).collapse())
assign(w_old.sub(1),omgPos_init)


init_omgNeg = Expression('abs(x[1])>=int2-tol?std::log((cNeg0)):std::log((cNum))', int2=int2, tol = tol, cNeg0 = cNeg0,  cNum=DOLFIN_EPS, degree=0)
omgNeg_init = interpolate(init_omgNeg,ME.sub(3).collapse())
assign(w_old.sub(3),omgNeg_init)

# Update initial guess for w, this helps the solver a bit on the first time step
assign(w.sub(3),omgNeg_init)
assign(w.sub(1),omgPos_init)


'''''''''''''''''''''
     SUBROUTINES
'''''''''''''''''''''


def F_calc(u):
    dim = len(u)
    Id = Identity(dim) # Identity tensor
    
    F = Id + grad(u) # 3D Deformation gradient
    return F # Full 3D F

    
def Piola(F,phi):
    
    Id = Identity(3)
    J = det(F)
    
    C = F.T*F
    Cdis = J**(-2/3)*C
    I1 = tr(Cdis)
         
    eR = -phi_norm*grad(phi)
    e_sp = inv(F.T)*eR
    #
    T_max = vareps0*vareps_r*J*(outer(e_sp,e_sp) - 1/2*(inner(e_sp,e_sp))*Id)*inv(F.T) 
    
    # Piola stress (Gent)
    TR = J**(-2/3)*Gshear*(Im_gent/(Im_gent+3-I1))*(F - 1/3*tr(C)*inv(F.T))\
        + Kbulk*ln(J)*inv(F.T) + T_max
    
    return TR

# Mass form
def m(u, u_test):
    return rho*inner(u, u_test)*dx

# Elastic stiffness form
def k(u, u_test,cPos,cNeg, phi):
    return inner(Piola(F_calc(u),phi), grad(u_test))*dx

# Rayleigh damping form
def c(u, u_test,cPos,cNeg, phi):
    return eta_m*m(u, u_test) + eta_k*k(u, u_test, cPos,cNeg, phi)

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

# Newmark-beta kinematical update
a_new = update_a(u, u_old, v_old, a_old, ufl=True)
v_new = update_v(a_new, u_old, v_old, a_old, ufl=True)

# get avg fields for generalized-alpha method
u_avg = avg(u_old, u, alpha)
v_avg = avg(v_old, v_new, alpha)
omgPos_avg = avg(omgPos_old, omgPos, alpha)
phi_avg    = avg(phi_old, phi, alpha)
omgNeg_avg = avg(omgNeg_old, omgNeg, alpha)

# Explicit concentration updates
cPos     = exp(omgPos_avg - phi_avg)
cNeg     = exp(omgNeg_avg + phi_avg)
cPos_old = exp(omgPos_old - phi_old)
cNeg_old = exp(omgNeg_old + phi_old)


# Kinematics
F = F_calc(u_avg)
C = F.T*F
Ci = inv(C)
F_old = F_calc(u_old) #grad(u_old) + Identity(3)
J = det(F)
J_old = det(F_old)



'''''''''''''''''''''''
       WEAK FORMS
'''''''''''''''''''''''

dynSwitch = Constant(1.0)
     
L0 = inner(Piola(F_calc(u_avg), phi_avg), grad(u_test))*dx \
         + dynSwitch*rho*inner(a_new, u_test)*dx 
   
L1 = dot((cPos-cPos_old)/dk/D,omgPos_test)*dx \
    + (cPos_old)*inner(Ci*matInd*grad(omgPos_avg),grad(omgPos_test))*dx
   
L3 = dot(vareps*phi_norm*J*Ci*grad(phi_avg),grad(phi_test))*dx \
     -dot(Farad*matInd*cMax*(cPos-cNeg),phi_test)*dx 

L4 = dot((cNeg-cNeg_old)/dk/D,omgNeg_test)*dx \
    + (cNeg_old)*dot(Ci*matInd*grad(omgNeg_avg),grad(omgNeg_test))*dx
   

# Total weak form

L = (1/Gshear0)*L0 + L1 + L3 + L4 


# Automatic differentiation tangent:
a = derivative(L, w, dw)

'''''''''''''''''''''''
BOUNDARY CONDITIONS
'''''''''''''''''''''''  

# Boundary condition expressions as necessary

phiRamp = Expression(("(250/phi_norm)*t/Tramp"),
                  phi_norm = phi_norm, Tramp = T_tot, t = 0.0, degree=1)

# Boundary condition definitions

bcs_1 = DirichletBC(ME.sub(2), 0., facets, 8) # Ground center of device
bcs_2 = DirichletBC(ME.sub(2), 0., facets, 6) # Ground top of device
bcs_3 = DirichletBC(ME.sub(2), 0., facets, 3) # Ground bottom of device

bcs_4 = DirichletBC(ME.sub(2), phiRamp, facets, 6) # Ground bottom of device

bcs_a = DirichletBC(ME.sub(0).sub(0),0.,facets,2)


x_plot = scaleX
phi_eq = w_old(x_plot, scaleY/2, scaleZ/2)[4]
#phiRamp = Expression(("1.0e3/phi_norm*(1-exp(-5*t/Tramp)) "),
#                 phi_norm = phi_norm,  Tramp = T2_tot, t = 0.0, degree=1)
phiRamp = Expression(("min(1.0e3/phi_norm*(t/Tramp), 1.0e3/phi_norm)"),
                 phi_eq=phi_eq, phi_norm = phi_norm, pi=np.pi, Tramp = T2_tot, t = 0.0, degree=2)
#disp = Expression(("5.5*scaleX*t/Tramp"),
#                  scaleX = scaleX, Tramp = T2_tot, t = 0.0, degree=1)

disp = Expression(("0.02*scaleX*2*abs(t/Tramp - floor(t/Tramp + 0.5))"),
                  scaleX = scaleX, Tramp = T2_tot, t = 0.0, degree=1)
    
bcs_f = DirichletBC(ME.sub(2),phiRamp,facets,6) # Ramp up phi on top face

#bcs_b1 = DirichletBC(ME.sub(0),Constant((0.,0., 0.)),facets,2) # left face built-in
bcs_b1 = DirichletBC(ME.sub(0).sub(0),0.0,facets,2) # pull right face
bcs_b2 = DirichletBC(ME.sub(0).sub(0),disp,facets,7) # pull right face
bcs_b3 = DirichletBC(ME.sub(0).sub(1),0.0,facets,3)
bcs_b4 = DirichletBC(ME.sub(0).sub(2),0.0,facets,5)



bcs2 = [bcs_3, bcs_a, bcs_b1, bcs_b2, bcs_b3, bcs_b4, bcs_f]

'''''''''''''''''''''
    RUN ANALYSIS
'''''''''''''''''''''


# Output file setup
file_results = XDMFFile("results/suo_capacitor_3D_triangle.xdmf")
file_results.parameters["flush_output"] = True
file_results.parameters["functions_share_mesh"] = True



# initialize counter
ii = 0
u_v = w_old.sub(0)
u_v.rename("disp","")

omgPos_v = w_old.sub(1)
omgPos_v.rename("omega", "")

phi_v = w_old.sub(2)
phi_v.rename("phi", "")

omgNeg_v = w_old.sub(3)
omgNeg_v.rename("omega2", "")

Gshear_v = project(1e6*Gshear,W3)
Gshear_v.rename("Gshear","")
file_results.write(Gshear_v,t)

Kbulk_v = project(1e6*Kbulk,W3)
Kbulk_v.rename("Kbulk","")
file_results.write(Kbulk_v,t)

intInd_v = project(intInd,W3)
intInd_v.rename("IntIndex","")
file_results.write(intInd_v,t)



# Initialize arrays for plotting later
Ny = N+1
x_plot = scaleX
y = np.sort(np.array(mesh.coordinates()[np.where(mesh.coordinates()[:,0] == x_plot),1])) #np.linspace(-scaleY/2, scaleY/2, Ny)
y = y[0,:]
y2 = np.linspace(0, scaleY/10, Ny) 

output_step = [0.,0.001,0.005,5.01, 50.01]
op_dim = len(output_step)
phi_var = np.zeros((Ny, op_dim))
cPos_var = np.zeros((Ny, op_dim))
#cNeg_var = np.zeros((Ny, op_dim))

voltage_out = np.zeros(100000)
charge_out = np.zeros(100000)
disp_out = np.zeros(100000)
time_out    = np.zeros(100000)
trac_out    = np.zeros(100000)

i=0
ii=0

trac_out    = np.zeros(100000)
ocv_out    = np.zeros(100000)
t_out    = np.zeros(100000)
t = T_tot
jj = 0
spike = 0

# Turn on implicit dynamics
dynSwitch.assign(Constant(1.0))

voltage_out = np.zeros(100000)
charge_out = np.zeros(100000)
disp_out = np.zeros(100000)
time_out    = np.zeros(100000)
trac_out    = np.zeros(100000)

#while (round(t,2) <= round(T_tot + 1.0*T2_tot,2)):
while (round(t,2) <= round(T_tot + 11.0*T2_tot,2)):
    
    # Output storage
    if True:
        file_results.write(u_v,t)
        file_results.write(omgPos_v, t)
        file_results.write(phi_v, t)
        phi_nonDim = phi_norm/1.e3*phi
        phi_nonDim = project(phi_nonDim,W)
        phi_nonDim.rename("phi_Orig","")
        file_results.write(phi_nonDim, t)
        file_results.write(omgNeg_v, t)
        file_results.write(Gshear_v,t)
        
        #
        _w_0,_w_1, _w_2, _w_3 = w_old.split()

    dt = T2_tot/20
    
    
    #dt = T2_tot/1000
    dk.assign(dt)
    
    
    if t-T_tot<=T2_tot:
        disp.t = 0.0
    else:
        disp.t = t - float(alpha*dt) - T_tot - T2_tot 
    #disp2.t = min(t - T_tot, T2_tot)
    phiRamp.t = t - float(alpha*dt) - T_tot
    
    StressProblem = NonlinearVariationalProblem(L, w, bcs2, J=a)
    
    # Set up the non-linear solver
    solver  = NonlinearVariationalSolver(StressProblem)
    
    # Solver parameters
    prm = solver.parameters
    prm['nonlinear_solver'] = 'newton'
    prm['newton_solver']['linear_solver'] = 'petsc'   #'petsc'   #'gmres'
    prm['newton_solver']['absolute_tolerance'] = 1.E-6    # 1.e-10
    prm['newton_solver']['relative_tolerance'] = 1.E-6    # 1.e-10
    prm['newton_solver']['maximum_iterations'] = 60
    
    
    # Solve the problem
    (iter, converged) = solver.solve()
    
    
    # output measured device voltage, etc.
    phi_top = w(x_plot, scaleY/2, scaleZ/2)[4]*phi_norm
    phi_btm = w(x_plot, -scaleY/2, scaleZ/2)[4]*phi_norm
    voltage_out[ii] = phi_top - phi_btm
    disp_out[ii]    = w(scaleX, scaleY/2, scaleZ/2)[0]
    time_out[ii]    = t
    #trac_out[jj]    = trac_final*(t-T_tot)/(T2_tot)
    ocv_out[jj] = phi_top - phi_btm
    t_out[jj]   = t - T_tot -float(alpha*dt)
    #
    
    
    u_proj = project(u, W2)
    u_proj_old = project(u_old, W2)
    update_fields(u_proj, u_proj_old, v_old, a_old)
    w_old.vector()[:] = w.vector()
    
    
    charge_out[ii] = assemble(Farad*cMax*(cPos-cNeg)*dx(3))
    
    # increment time
    t += dt
    ii = ii + 1
    jj = jj + 1
    
    
    # Print progress of calculation
    #if ii%10 == 0:
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Step: Sense   |   Simulation Time: {} s  |     Iterations: {}".format(t/1e6, iter))
    print()
    

# Final step paraview output
file_results.write(u_v,t)
file_results.write(omgPos_v, t)
file_results.write(phi_v, t)
file_results.write(omgNeg_v, t)
file_results.write(Gshear_v,t)

# output final measured device voltage
phi_top = w(x_plot, scaleY/2, scaleZ/2)[4]*phi_norm
phi_btm = w(x_plot,-scaleY/2., scaleZ/2)[4]*phi_norm
disp_out[ii]    = w(scaleX, scaleY/2, scaleZ/2)[0]
#voltage_out[ii] = phi_top - phi_btm
#charge_out[ii] = assemble(Farad*cMax*(cPos-cNeg)*dx(3))
#time_out[ii]    = t



font = {'size'   : 16}

plt.rc('font', **font)

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']
font = {'size'   : 16}

# Time signals
fig, (ax1, ax2, ax3) = plt.subplots(3,1, sharex='col')
 

color = colors[2]
ax1.set_ylabel(r"$V$ (V)") #, color=color)
ax1.plot(np.insert(time_out[np.where(time_out>T_tot)]/1.e6 - T_tot/1e6, 0, 0 ), np.insert(np.array(voltage_out[np.where(time_out>T_tot)]-ocv_out[0])/1e3, 0, 0),linewidth=3.0, color=color, 
         label=r"$V$")
ax1.axvline(T2_tot/1e6, c='k', linewidth=1.)
ax1.set_ylim(-0.15, 1.15)
#ax1.tick_params(axis='y', labelcolor=color)

color = colors[3]
ax3.set_ylabel(r"Q$_{{D}}$ (pC)") #, color=color)
ax3.plot(np.insert(time_out[np.where(time_out>T_tot)]/1.e6 - T_tot/1e6, 0, 0), np.insert(1e9*charge_out[np.where(time_out>T_tot)],0 ,0),linewidth=3.0, color=color, 
         label=r"Q$_{{D}}$")
#ax2.tick_params(axis='y', labelcolor=color)
ax3.axvline(T2_tot/1e6, c='k', linewidth=1.)
#plt.xlim(0,50)
ax3.set_ylim(22,24)
#ax3.set_yticks(np.array([0, 12.5, 25]))

'''
fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.savefig("plots/OCV_disp.png", dpi=600)
'''
stretch_out = (scaleX + disp_out[np.where(time_out>T_tot)])/scaleX
color = colors[0]
ax2.set_ylabel(r"$\lambda$ (-)") #, color=color)
ax2.plot(np.insert(time_out[np.where(time_out>T_tot)]/1.e6 - T_tot/1e6, 0, 0), np.insert(stretch_out, 0, 1), color=color, linewidth=3.0)
ax2.axvline(T2_tot/1e6, c='k', linewidth=1.)
ax2.set_ylim(0.99, 1.03)
ax2.set_yticks(np.array([1, 1.01, 1.02]))


ax3.set_xlabel(r"Time (s)")
ax3.set_xlim(0*T2_tot/1e6, 11.0*T2_tot/1e6)

# save figure to file
fig = plt.gcf()
fig.set_size_inches(8, 5)
plt.tight_layout()
plt.savefig("plots/cyclic_signals.png", dpi=600)



# Experimental comparison
expData = np.genfromtxt('uniaxial_cap_cycle_data.csv', delimiter=',')

plt.figure()
fig = plt.gcf()
fig.set_size_inches(7.5,6)
plt.scatter(expData[:,0]*10/expData[-1,0], expData[:,1]/expData[0,1], s=100,
                     edgecolors=(0.0, 0.0, 0.0,1),
                     color=(1, 1, 1, 1),
                     label='Experiment', linewidth=2.0)
C0 = (1e12)*np.array(charge_out[np.where(time_out==T_tot + T2_tot)])/np.array(voltage_out[np.where(time_out==T_tot + T2_tot)])
#plt.plot(time_out[np.where(voltage_out!=0)], charge_out[np.where(voltage_out!=0)],linewidth=3.)
plt.plot(np.array(time_out[np.where(time_out>=T_tot)] - T2_tot + dt)/T2_tot, (1e12)*np.array(charge_out[np.where(time_out>=T_tot)])/np.array(voltage_out[np.where(time_out>=T_tot)])/C0,\
         label='Simulation', linewidth=3.0, color='k')
plt.axvline(0/1e6, c='k', linewidth=1.)
plt.xlim(0,10)
plt.ylim(0.9975,1.030)
#plt.axis('tight')
plt.xlabel(r"Cycles")
plt.ylabel(r"C$_{{D}} /$ C$_{{D,0}}$")
#plt.grid(linestyle="--", linewidth=0.5, color='b')
plt.legend()

# save figure to file
fig = plt.gcf()
fig.set_size_inches(8, 5)
plt.tight_layout()
plt.savefig("plots/cyclic_capacitance.png", dpi=600)
