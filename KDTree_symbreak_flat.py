import numpy as np
import scipy as sp
from scipy import sparse
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation
import matplotlib.patches as patches
from tqdm import trange

class AGranFlat:
    def __init__(self,Lx = 100.0,Ly=100.0,AR = 1.5,r0 = 1., rb = 0.3,rho=0.25, r_tr = 8, T = 10,factor = 0.1, v0 = 1.2, eta = 1500,mua = 0.0007, mup = 0.001, mur = 0.0001,mu_tr = 0.0001,trN = 1, k = 200,kw = 1, dw = 8, v_drag=1, mode='drag',lfrac = 0.8,pfrac = 0.2,Nrelax = 20000):
        self.Lx = Lx   # system size
        self.Ly = Ly
        self.AR = AR   # aspect ratio
        self.r0 = r0   # particle size
        self.rb = rb
        self.r_tr = r_tr  # tracer size
        self.anoise = False
    
        rho = rho          # system packing fraction


        factor =factor       # one step jump length
        self.v0 = v0   # propulsion speed
        self.dt = r0*factor/(self.v0)        # time discretization
        self.T=2/self.dt  # number of steps for damping

        self.eta = eta          # noise strength
        self.mua = mua         # mobility
        self.mup = mup
        self.mu_tr = mu_tr
        self.mur = mur             # angular mobility
        self.f0 = self.v0/(self.T*self.mua*self.dt)
        # w0 = 0.2
        # tau0 = w0*10/mur
        self.k = k              # wca strength
        self.kw = kw            # wall stiffness
        self.dw = dw            # wall thickness
        self.v_drag = v_drag
        self.mode =mode
        self.lfrac  = lfrac
        self.Nrelax = Nrelax




        self.pos_tr = np.zeros(2)
        self.pos_tr[0] = self.Lx/2
        self.pos_tr[1] = self.Ly/2

        # N_tr = int(trN*self.Ly/self.r0)
        
#         self.body_shape = np.zeros((2*N_tr,2))
#         # self.body_shape[:,0] = np.linspace(-self.Lx*self.lfrac/2,self.Lx*self.lfrac/2,2*N_tr)/10
#         self.body_shape[:,1] = np.linspace(-self.Ly*self.lfrac/2,self.Ly*self.lfrac/2,2*N_tr)
        
# #         seg1  = np.zeros((2*N_tr,2))
# #         seg1[:,0] = np.linspace(-self.Lx*self.lfrac/2,0,2*N_tr)/5
# #         seg1[:,1] = np.linspace(0,self.Ly*self.lfrac/2,2*N_tr)/2
# #         seg2  = np.zeros((2*N_tr,2))
# #         seg2[:,0] = np.linspace(0,self.Lx*self.lfrac/2,2*N_tr)/5
# #         seg2[:,1] = np.linspace(self.Ly*self.lfrac/2,0,2*N_tr)/2
# #         seg3  = np.zeros((2*N_tr,2))
# #         seg3[:,0] = np.linspace(-self.Lx*self.lfrac/2,0,2*N_tr)/5
# #         seg3[:,1] = np.linspace(0,-self.Ly*self.lfrac/2,2*N_tr)/2
# #         seg4  = np.zeros((2*N_tr,2))
# #         seg4[:,0] = np.linspace(0,self.Lx*self.lfrac/2,2*N_tr)/5
# #         seg4[:,1] = np.linspace(-self.Ly*self.lfrac/2,0,2*N_tr)/2
        
# #         self.body_shape = np.concatenate([seg1,seg2,seg3,seg4,seg1*2/3,seg2*2/3,seg3*2/3,seg4*2/3,seg1/3,seg2/3,seg3/3,seg4/3])
        
#         self.body_tr = np.zeros((len(self.body_shape),2))
#         self.body_tr[:,0] = self.pos_tr[0]+self.body_shape[:,0]
#         self.body_tr[:,1] = self.pos_tr[1]+self.body_shape[:,1]

       

        self.pfrac = pfrac
        self.N = int(rho*(self.Lx*self.Ly-3*self.Ly*self.r0)/((1-self.pfrac)*self.AR*np.pi*self.r0**2+self.pfrac*np.pi*self.rb**2))
        self.Np = int(self.N*self.pfrac)
        # N=10
        self.initialize()
        self.relax()

    def initialize(self):
        self.pos = np.zeros((self.N,2))
#         print(self.Lx/2)
#         print(-self.Lx/2)

        self.pos[:,0] = np.random.uniform(-self.Lx/2+2*self.r0,self.Lx/2-2*self.r0,size=self.N)
        self.pos[:,1] = np.random.uniform(-self.Ly/2,self.Ly/2,size=self.N)
        self.orient = np.random.uniform(-np.pi, np.pi,size=self.N-self.Np)
        self.mom_trans = np.zeros((self.N,2))
        self.mom_trans[:,0] = np.cos(self.orient)*self.v0/self.mua
        self.mom_trans[:,1] = np.sin(self.orient)*self.v0/self.mua

        self.mom_ang = np.zeros(self.N-self.Np)

        self.VX_avg = 0
        self.VY_avg = 0
        
        
        x,y = np.meshgrid(np.linspace(1,self.Lx-1,40),np.linspace(1,self.Ly-1,40))
        self.marker = np.zeros((1600,2))
        self.marker[:,0] = x.reshape(-1)
        self.marker[:,1] = y.reshape(-1)
        
        self.set_coord()
#         self.relax()


        

    
    def set_coord(self):
        self.pos[:,0] = self.pos[:,0]%self.Lx
        self.pos[:,1] = self.pos[:,1]%self.Ly
        self.pos_tr[0] = self.pos_tr[0]%self.Lx
        self.pos_tr[1] = self.pos_tr[1]%self.Ly

        self.armb1 = (self.AR-(1/self.AR))*self.r0*np.array([np.cos(self.orient),np.sin(self.orient)]).T
        self.armb2 = - (self.AR-(1/self.AR))*self.r0*np.array([np.cos(self.orient),np.sin(self.orient)]).T
    #     armb1 = (AR-(1/AR))*r0*np.array([np.cos(orient+np.pi/2),np.sin(orient+np.pi/2)]).T
    #     armb2 = - (AR-(1/AR))*r0*np.array([np.cos(orient+np.pi/2),np.sin(orient+np.pi/2)]).T


        self.pos1 = self.pos[self.Np:] + self.armb1
        self.pos2 = self.pos[self.Np:] + self.armb2
        self.pos1[:,0] = self.pos1[:,0]%self.Lx
        self.pos1[:,1] = self.pos1[:,1]%self.Ly
        self.pos2[:,0] = self.pos2[:,0]%self.Lx
        self.pos2[:,1] = self.pos2[:,1]%self.Ly
        self.postot = np.concatenate([self.pos1,self.pos2])

# #         if self.mode=='free':

# #             self.body_tr[:,0] = self.pos_tr[0]+self.body_shape[:,0]
# #             self.body_tr[:,1] = self.pos_tr[1]+self.body_shape[:,1]

# #         self.body_tr[:,0] = self.body_tr[:,0]%self.Lx
# #         self.body_tr[:,1] = self.body_tr[:,1]%self.Ly


    def WCA(self,rsq,r_unit,k):
        return self.k*(6*(r_unit)**6/(rsq)**(7/2) - 12*(r_unit)**12/(rsq)**(13/2))

    def FWCA(self,p1,p2,k,r_ref,typ):


        tree1 = cKDTree(p1,boxsize=[self.Lx,self.Ly])
        tree2 = cKDTree(p2,boxsize=[self.Lx,self.Ly])
        dist = tree1.sparse_distance_matrix(tree2, max_distance=(r_ref)*2**(1/6),output_type='coo_matrix')

        dx = -p1[dist.row,0]+p2[dist.col,0]
        dy = -p1[dist.row,1]+p2[dist.col,1]

        dx[dx>self.Lx/2] -=self.Lx
        dx[dx<-self.Lx/2]+=self.Lx
        dy[dy>self.Ly/2] -=self.Ly
        dy[dy<-self.Ly/2]+=self.Ly

        force = self.WCA(dx**2+dy**2,r_ref,self.k)
        if typ==0:
            filt = (~np.isnan(force))*(np.abs(force)<np.abs(self.WCA((r_ref*0.8)**2,r_ref,self.k)))
        else:
            filt = (~np.isnan(force))
        angle = np.angle(dx+1j*dy)
        fx = sparse.coo_matrix((force[filt]*np.cos(angle[filt]),(dist.row[filt],dist.col[filt])), shape=dist.get_shape())
        fy = sparse.coo_matrix((force[filt]*np.sin(angle[filt]),(dist.row[filt],dist.col[filt])), shape=dist.get_shape())
        Fx = np.squeeze(np.asarray(fx.sum(axis=1)))
        Fy = np.squeeze(np.asarray(fy.sum(axis=1)))

        return (Fx,Fy)






    def Torque(self,Fx,Fy, rx,ry):
        return (rx*Fy-ry*Fx)


    




    def update(self):



        # interaction
        FX = np.zeros(self.N)
        FY = np.zeros(self.N)
        TAU = np.zeros(self.N-self.Np)
        self.set_coord()


        # volume exclusion (2body)

        (Fxvol1,Fyvol1) = self.FWCA(self.pos1,self.postot,self.k,self.r0*2/self.AR,0)
        FX[self.Np:] += Fxvol1
        FY[self.Np:] += Fyvol1
        TAU += self.Torque(Fxvol1,Fyvol1,self.armb1[:,0],self.armb1[:,1])
        (Fxvol2,Fyvol2) = self.FWCA(self.pos2,self.postot,self.k,self.r0*2/self.AR,0)
        FX[self.Np:] += Fxvol2
        FY[self.Np:] += Fyvol2
        TAU += self.Torque(Fxvol2,Fyvol2,self.armb2[:,0],self.armb2[:,1])
        
        (Fxvol10,Fyvol10) = self.FWCA(self.pos1,self.pos[self.Np:],self.k,self.r0*(1+1/self.AR),0)
        FX[self.Np:] += Fxvol10
        FY[self.Np:] += Fyvol10
        TAU += self.Torque(Fxvol10,Fyvol10,self.armb1[:,0],self.armb1[:,1])
        (Fxvol20,Fyvol20) = self.FWCA(self.pos2,self.pos[self.Np:],self.k,self.r0*(1+1/self.AR),0)
        FX[self.Np:] += Fxvol20
        FY[self.Np:] += Fyvol20
        TAU += self.Torque(Fxvol20,Fyvol20,self.armb2[:,0],self.armb2[:,1])
        
        # (Fxvol10,Fyvol10) = self.FWCA(self.pos1,self.pos[:self.Np],self.k,self.r0*(1/self.AR)+self.rb,0)
        # FX[self.Np:] += Fxvol10
        # FY[self.Np:] += Fyvol10
        # TAU += self.Torque(Fxvol10,Fyvol10,self.armb1[:,0],self.armb1[:,1])
        # (Fxvol20,Fyvol20) = self.FWCA(self.pos2,self.pos[:self.Np],self.k,self.r0*(1/self.AR)+self.rb,0)
        # FX[:self.Np] += Fxvol20
        # FY[:self.Np] += Fyvol20
        # TAU += self.Torque(Fxvol20,Fyvol20,self.armb2[:,0],self.armb2[:,1])
        
        (Fxvol01,Fyvol01) = self.FWCA(self.pos[self.Np:],self.pos1,self.k,self.r0*(1+1/self.AR),0)
        FX[self.Np:] += Fxvol01
        FY[self.Np:] += Fyvol01
        (Fxvol02,Fyvol02) = self.FWCA(self.pos[self.Np:],self.pos2,self.k,self.r0*(1+1/self.AR),0)
        FX[self.Np:] += Fxvol02
        FY[self.Np:] += Fyvol02
        
        # (Fxvol01,Fyvol01) = self.FWCA(self.pos[:self.Np],self.pos1,self.k,self.r0*(1/self.AR)+self.rb,0)
        # FX[:self.Np] += Fxvol01
        # FY[:self.Np] += Fyvol01
        # (Fxvol02,Fyvol02) = self.FWCA(self.pos[:self.Np],self.pos2,self.k,self.r0*(1/self.AR)+self.rb,0)
        # FX[:self.Np] += Fxvol02
        # FY[:self.Np] += Fyvol02
        
        # (Fxvol00,Fyvol00) = self.FWCA(self.pos[:self.Np],self.pos[:self.Np],self.k,self.rb*2,0)
        # FX[:self.Np] += Fxvol00
        # FY[:self.Np] += Fyvol00
        # (Fxvol00,Fyvol00) = self.FWCA(self.pos[:self.Np],self.pos[self.Np:],self.k,self.r0+self.rb,0)
        # FX[:self.Np] += Fxvol00
        # FY[:self.Np] += Fyvol00
        # (Fxvol00,Fyvol00) = self.FWCA(self.pos[self.Np:],self.pos[:self.Np],self.k,self.r0+self.rb,0)
        # FX[self.Np:] += Fxvol00
        # FY[self.Np:] += Fyvol00
        (Fxvol00,Fyvol00) = self.FWCA(self.pos[self.Np:],self.pos[self.Np:],self.k,self.r0*2,0)
        FX[self.Np:] += Fxvol00
        FY[self.Np:] += Fyvol00


        # tracer dynamics
        dx1 = (self.pos1[:,0]-self.pos_tr[0])%self.Lx
        dx2 = (self.pos2[:,0]-self.pos_tr[0])%self.Lx
        dx0 = (self.pos[:,0]-self.pos_tr[0])%self.Lx
        
        dx1[dx1>self.Lx/2]-=self.Lx
        dx2[dx2>self.Lx/2]-=self.Lx
        dx0[dx0>self.Lx/2]-=self.Lx
        
        filt1l = (0>dx1)*(dx1>-self.dw)
        filt2l = (0>dx2)*(dx2>-self.dw)
        filt0l = (0>dx0)*(dx0>-self.dw)

        filt1r = (0<dx1)*(dx1<self.dw)
        filt2r = (0<dx2)*(dx2<self.dw)
        filt0r = (0<dx0)*(dx0<self.dw)
        
        Fxtr1 = np.zeros(self.N)
        Fxtr2 = np.zeros(self.N)
        Fxtr0 = np.zeros(self.N)
        
        Fxtr1[filt1l] -= self.kw*(self.dw+dx1[filt1l])
        Fxtr2[filt2l] -= self.kw*(self.dw+dx2[filt2l])
        Fxtr0[filt0l] -= self.kw*(self.dw+dx0[filt0l])

        Fxtr1[filt1r] -= self.kw*(-self.dw+dx1[filt1r])
        Fxtr2[filt2r] -= self.kw*(-self.dw+dx2[filt2r])
        Fxtr0[filt0r] -= self.kw*(-self.dw+dx0[filt0r])
        
        FX[self.Np:] += Fxtr0+Fxtr1+Fxtr2
        TAU+=self.Torque(Fxtr1,np.zeros(self.N),self.armb1[:,0],self.armb1[:,1])+ self.Torque(Fxtr2,np.zeros(self.N),self.armb2[:,0],self.armb2[:,1])
        
        VX = -self.mu_tr*np.sum(Fxtr0+Fxtr1+Fxtr2)
        VY=0
        
        
        
#         (Fxtr1,Fytr1) = self.FWCA(self.pos1,self.body_tr,self.k,self.r0*(1+1/self.AR),1)
#         (Fxtr2,Fytr2) = self.FWCA(self.pos2,self.body_tr,self.k,self.r0*(1+1/self.AR),1)
        
#         (Fxtr0,Fytr0) = self.FWCA(self.pos[self.Np:],self.body_tr,self.k,self.r0*2,1)
#         (Fxtrp,Fytrp) = self.FWCA(self.pos[:self.Np],self.body_tr,self.k,self.r0+self.rb,1)

#         FX[:self.Np] += Fxtrp
#         FX[self.Np:] += Fxtr0+Fxtr1+Fxtr2
#         FY[:self.Np] += Fytrp
#         FY[self.Np:] += Fytr0+Fytr1+Fytr2
#         TAU += self.Torque(Fxtr1,Fytr1,self.armb1[:,0],self.armb1[:,1])+ self.Torque(Fxtr2,Fytr2,self.armb2[:,0],self.armb2[:,1])
#         VX = -self.mu_tr*(np.sum(Fxtrp)+np.sum(Fxtr0+Fxtr1+Fxtr2))
#         VY = -self.mu_tr*(np.sum(Fytrp)+np.sum(Fytr0+Fytr1+Fytr2))

    #     pos_tr[0]-=mu_tr*np.sum(Fxtr0+Fxtr1+Fxtr2)
    #     pos_tr[1]-=mu_tr*np.sum(Fytr0+Fytr1+Fytr2)

    
    
    
    
    
    
    
    
        # in object frame
        # self.pos[:,0] +=-VX
        # self.pos[:,1] +=-VY
        if self.mode=='drag':
            self.pos[:,0] -= self.v_drag*self.dt
            # self.pos_tr[0] += self.v_drag*self.dt
        elif self.mode=='free':
            self.pos_tr[0] +=VX*self.dt
            self.pos_tr[1] +=VY*self.dt
            
        # self.body_tr[:,0] = self.pos_tr[0]+self.body_shape[:,0]

            

        self.VX_avg = 0*(1-1/self.T)*self.VX_avg+VX
        self.VY_avg = 0*(1-1/self.T)*self.VY_avg+VY
        
        
        # drag force
        FX -= self.mom_trans[:,0]/(self.T*self.dt)  
        FY -= self.mom_trans[:,1]/(self.T*self.dt)
        TAU -= self.mom_ang/(self.T*self.dt)
        
        # propulsion force
        FX[self.Np:] += self.f0*np.cos(self.orient)
        FY[self.Np:] += self.f0*np.sin(self.orient)
    #     TAU += tau0
    #     TAU[:int(N/2)] +=tau0
    #     TAU[int(N/2):] -=tau0




        

        # noise
        FX[:self.Np] += self.eta*np.sqrt(self.dt)*np.random.uniform(-1, 1, size=self.Np)
        FY[:self.Np] += self.eta*np.sqrt(self.dt)*np.random.uniform(-1, 1, size=self.Np)
        if self.anoise==True:
            FX[self.Np:] += self.eta*np.sqrt(self.dt)*np.random.uniform(-1, 1, size=self.N-self.Np)
            FY[self.Np:] += self.eta*np.sqrt(self.dt)*np.random.uniform(-1, 1, size=self.N-self.Np)
#         TAU += 3*self.eta*np.sqrt(self.dt)*np.random.uniform(-np.pi, np.pi, size=self.N)/self.mur




        # momentum update
        self.mom_trans[:,0]+=FX*self.dt
        self.mom_trans[:,1]+=FY*self.dt
        self.mom_ang[:]+=TAU*self.dt


        # position update
        self.pos[:self.Np,0] += self.mup*self.mom_trans[:self.Np,0]*self.dt
        self.pos[:self.Np,1] += self.mup*self.mom_trans[:self.Np,1]*self.dt
        self.pos[self.Np:,0] += self.mua*self.mom_trans[self.Np:,0]*self.dt
        self.pos[self.Np:,1] += self.mua*self.mom_trans[self.Np:,1]*self.dt
    #     orient   += mur*TAU*dt
        self.orient += self.mur*self.mom_ang[:]*self.dt
        # self.pos[:,0] +=self.mu*FX*self.dt
        # self.pos[:,1] +=self.mu*FY*self.dt
        # self.orient +=self.mur*TAU*self.dt


    # periodic boundary
        self.pos[:,0] = self.pos[:,0]%self.Lx
        self.pos[:,1] = self.pos[:,1]%self.Ly
        self.orient = self.orient%(2*np.pi)

        self.set_coord()
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




    # relaxation of initial position to avoid overlapping
    def relax(self):
        Lx = self.Lx
        Ly = self.Ly
        mode = self.mode
        Lscale = 1
        self.Lx = Lscale*Lx
        self.Ly = Lscale*Ly
        self.mode = 'relax'
        
        self.initialize()
        self.set_coord()
        tree = cKDTree(self.pos,boxsize=[self.Lx,self.Ly])
        # treeall = cKDTree(np.concatenate([self.pos,self.body_tr]),boxsize=[self.Lx,self.Ly])
        dist = tree.sparse_distance_matrix(tree, max_distance=self.r0*2*2**(1/6),output_type='coo_matrix')
        while (len(dist.col)>self.N):
            filt = (dist.col!=dist.row)
            self.pos[dist.col[filt][0]][0] = np.random.uniform(-self.Lx/2,self.Lx/2,size=1)
            self.pos[dist.col[filt][0]][1] = np.random.uniform(-self.Ly/2,self.Ly/2,size=1)
            self.set_coord()

            tree = cKDTree(self.pos,boxsize=[self.Lx,self.Ly])
            # treeall = cKDTree(np.concatenate([self.pos,self.body_tr]),boxsize=[self.Lx,self.Ly])
            dist = tree.sparse_distance_matrix(tree, max_distance=self.r0*1.9*2**(1/6),output_type='coo_matrix')
            
            
        
        for i in trange(self.Nrelax):
            self.Lx = (Lscale-(Lscale-1)*(i+1)/self.Nrelax)*Lx
            self.Ly = (Lscale-(Lscale-1)*(i+1)/self.Nrelax)*Ly
            
            self.pos[:,:] *=(Lscale-(Lscale-1)*(i+1)/self.Nrelax)/(Lscale-(Lscale-1)*(i)/self.Nrelax)
            self.pos[:,0] +=(Lx/2)*(1-(Lscale-(Lscale-1)*(i+1)/self.Nrelax)/(Lscale-(Lscale-1)*(i)/self.Nrelax))
            self.update()

        self.mode = mode




