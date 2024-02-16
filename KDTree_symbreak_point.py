import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.patches as patches
from tqdm import trange

class AGranPoint:
    def __init__(self,Lx = 100.0,Ly=100.0,rho=0.25, r_tr = 8, r0 = 1, v0 = 1.2, eta = 1500,xi = 10,mu = 0.1, mur = 1,mu_tr = 0.0001,ka = 0.1,v_drag=1,k = 1, mode='drag'):
        self.Lx = Lx   # system size
        self.Ly = Ly
    
        rho = rho          # system packing fraction


        self.v0 = v0   # propulsion speed
        self.dt = 0.1        # time discretization
        self.eta = eta          # noise strength
        self.xi = xi
        self.r0 = r0
        self.mu = mu         # mobility
        self.mu_tr = mu_tr
        self.mur = mur             # angular mobility
        self.v_drag = v_drag
        self.mode =mode
        self.k = k
        self.ka = ka


        self.pos_tr = np.zeros(2)
        self.pos_tr[0] = self.Lx/2
        self.pos_tr[1] = self.Ly/2


        self.N = int(rho*(self.Lx*self.Ly))
        # N=10
        self.initialize()

    def initialize(self):
        self.pos = np.zeros((self.N,2))

        self.pos[:,0] = np.random.uniform(-self.Lx/2+2*self.r0,self.Lx/2-2*self.r0,size=self.N)
        self.pos[:,1] = np.random.uniform(-self.Ly/2,self.Ly/2,size=self.N)
        self.orient = np.random.uniform(-np.pi, np.pi,size=self.N)
        self.periodic()

        
        # self.set_coord()
#         self.relax()


        

    
    def periodic(self):
        self.pos[:,0] = self.pos[:,0]%self.Lx
        self.pos[:,1] = self.pos[:,1]%self.Ly
        self.pos_tr[0] = self.pos_tr[0]%self.Lx
        self.pos_tr[1] = self.pos_tr[1]%self.Ly
        
    def nematic(self):
        tree = cKDTree(self.pos,boxsize=[self.Lx,self.Ly])
        dist = tree.sparse_distance_matrix(tree, max_distance=self.r0,output_type='coo_matrix')
        
        dtheta = -self.orient[dist.row] + self.orient[dist.col]
        
        dx = -self.pos[dist.row,0]+self.pos[dist.col,0]
        dy = -self.pos[dist.row,1]+self.pos[dist.col,1]
        
        # torque = sparse.coo_matrix((self.mur*np.sin(2*dtheta)/(dx**2+dy**2+0.1),(dist.row,dist.col)), shape=dist.get_shape())
        torque = sparse.coo_matrix((self.mur*np.sin(2*dtheta),(dist.row,dist.col)), shape=dist.get_shape())
        
        TAU = np.squeeze(np.asarray(torque.sum(axis=1)))#/np.squeeze(np.asarray(torque.getnnz(axis=1)))
        density = np.squeeze(np.asarray(torque.getnnz(axis=1)))                                                                        
        
        return TAU,density
    
    def WCA(self,rsq,r_unit,k):
        return self.k*(6*(r_unit)**6/(rsq)**(7/2) - 12*(r_unit)**12/(rsq)**(13/2))

    def Fvol(self,p1,p2,k,r0):


        tree1 = cKDTree(p1,boxsize=[self.Lx,self.Ly])
        tree2 = cKDTree(p2,boxsize=[self.Lx,self.Ly])
        dist = tree1.sparse_distance_matrix(tree2, max_distance=(r0)*2**(1/6),output_type='coo_matrix')

        dx = -p1[dist.row,0]+p2[dist.col,0]
        dy = -p1[dist.row,1]+p2[dist.col,1]

        dx[dx>self.Lx/2] -=self.Lx
        dx[dx<-self.Lx/2]+=self.Lx
        dy[dy>self.Ly/2] -=self.Ly
        dy[dy<-self.Ly/2]+=self.Ly

        force = -k*(2*(-np.sqrt(dx**2+dy**2)+r0)/r0)**2
        # force = self.WCA(dx**2+dy**2,r0,k)

        # filt = (~np.isnan(force))
        
        angle = np.angle(dx+1j*dy)
        # fx = sparse.coo_matrix((force[filt]*np.cos(angle[filt]),(dist.row[filt],dist.col[filt])), shape=dist.get_shape())
        # fy = sparse.coo_matrix((force[filt]*np.sin(angle[filt]),(dist.row[filt],dist.col[filt])), shape=dist.get_shape())
        
        fx = sparse.coo_matrix((force*np.cos(angle),(dist.row,dist.col)), shape=dist.get_shape())
        fy = sparse.coo_matrix((force*np.sin(angle),(dist.row,dist.col)), shape=dist.get_shape())
        Fx = np.squeeze(np.asarray(fx.sum(axis=1)))
        Fy = np.squeeze(np.asarray(fy.sum(axis=1)))

        return (Fx,Fy)


        
    def wall_repul(self):
        dx = (self.pos[:,0]-self.pos_tr[0]-self.Lx/2)%self.Lx-self.Lx/2
        dy = (self.pos[:,1]-self.pos_tr[1]-self.Ly/2)%self.Ly-self.Ly/2
        FX = np.zeros(self.N)
        FY = np.zeros(self.N)

        r0 = 3
        inter = (dx**2<r0**2)*(dy**2<r0**2)
        filt1 = inter*(dx>0)
        filt2 = inter*(dx<0)
        filt3 = inter*(dx>0)
        filt4 = inter*(dx<0)
        FX[filt1] = 3*(r0-dx[filt1])**2
        FX[filt2] = -3*(dx[filt2]+r0)**2
        FY[filt3] = 3*(r0-dy[filt3])**2
        FY[filt4] = -3*(dy[filt4]+r0)**2
        # Fx[filt] = self.WCA(dx[filt],r_unit)
        return (FX*self.k, FY*self.k)
        
    def wall_torque(self):
        dx = (self.pos[:,0]-self.pos_tr[0]-self.Lx/2)%self.Lx-self.Lx/2
        dy = (self.pos[:,1]-self.pos_tr[1]-self.Ly/2)%self.Ly-self.Ly/2
        TAU = np.zeros(self.N)
        r0 = 3
        inter = (dx**2<r0**2)*(dy**2<r0**2)
        filt1 = inter*(dx>0)
        filt2 = inter*(dx<0)
        filt3 = inter*(dy>0)
        filt4 = inter*(dy<0)
        TAU[filt1] = 6*(r0-dx[filt1])*np.sin(2*self.orient[filt1])
        TAU[filt2] = 6*(dx[filt2]+r0)*np.sin(2*self.orient[filt2])
        TAU[filt3] = -6*(r0-dy[filt3])*np.sin(2*self.orient[filt3])
        TAU[filt4] = -6*(dy[filt4]+r0)*np.sin(2*self.orient[filt4])
        return TAU

    def update(self):

        # interaction
        FX = np.zeros(self.N)
        FY = np.zeros(self.N)
        TAU = np.zeros(self.N)
        
        # nematic alignment
        tau, dens = self.nematic()
        TAU += self.ka*tau/dens
        
        # volume exclusion
        (Fxvol,Fyvol) = self.Fvol(self.pos,self.pos,0.1,0.5)
        FX += Fxvol
        FY += Fyvol

        # tracer dynamics
        Fx_wall,Fy_wall = self.wall_repul()
        FX += Fx_wall
        FY += Fy_wall
        TAU += self.wall_torque()


        if self.mode=='drag':
            self.pos[:,0] -= self.v_drag*self.dt
            # self.pos_tr += self.v_drag*self.dt
        elif self.mode=='free':
            self.pos_tr -=self.mu_tr*np.sum(self.wall_repul())*self.dt
            
        # self.body_tr[:,0] = self.pos_tr[0]+self.body_shape[:,0]
            

        # propulsion force
        FX += self.v0*np.cos(self.orient)/self.mu
        FY += self.v0*np.sin(self.orient)/self.mu


        # noise
        FX += self.eta*np.sqrt(self.dt)*np.random.uniform(-1, 1, size=self.N)
        FY += self.eta*np.sqrt(self.dt)*np.random.uniform(-1, 1, size=self.N)
        TAU += self.xi*np.sqrt(self.dt)*np.random.uniform(-1, 1, size=self.N)
        


        # position update
        self.pos[:,0] += self.mu*FX*self.dt#*100/(1+dens)
        self.pos[:,1] += self.mu*FY*self.dt#*100/(1+dens)
        self.orient += self.mur*TAU*self.dt#*10/(1+dens)


    # periodic boundary
        self.pos[:,0] = self.pos[:,0]%self.Lx
        self.pos[:,1] = self.pos[:,1]%self.Ly
        self.orient = self.orient%(2*np.pi)

    def measure(self):
        tree1 = cKDTree(self.marker,boxsize=[self.Lx,self.Ly])
        tree2 = cKDTree(self.pos,boxsize=[self.Lx,self.Ly])
        dist = tree1.sparse_distance_matrix(tree2, max_distance=self.r0*4,output_type='coo_matrix')

        rho = np.ones(self.N)[dist.col]
        rho_mat = sparse.coo_matrix((rho,(dist.row,dist.col)), shape=dist.get_shape())
        self.rho = np.squeeze(np.asarray(rho_mat.sum(axis=1)))

        px = np.cos(self.orient[dist.col])
        py = np.sin(self.orient[dist.col])
        px_mat = sparse.coo_matrix((px,(dist.row,dist.col)), shape=dist.get_shape())
        py_mat = sparse.coo_matrix((py,(dist.row,dist.col)), shape=dist.get_shape())
        self.px = np.squeeze(np.asarray(px_mat.sum(axis=1)))
        self.py = np.squeeze(np.asarray(py_mat.sum(axis=1)))








