import numpy as np
from numpy.fft import rfft2,irfft2,fftshift
from scipy.signal import convolve2d

class LaplaceVoltage:
    boundary_conditions=["box","circulant"]
    def __init__(self, masks:list, volts:list, sidelengths:tuple=(100,100)):
        """
        masks : list
            A list of boolean ndarrays
        volts : list
            A list of floats, same len as masks
        sidelengths : tuple
            (x,y), the length of x, the length of y; gives length scale
        """
        assert len(volts)==len(masks), "len masks and voltages don't match"
        self.shape=masks[0].shape
        self.masks=[mask.copy() for mask in masks]# deepcopy
        self.volts=volts.copy()
        self.mask_checks() # throw error if user put something stupid
        self.set_boundary_conditions("circulant")
        self.nx   =self.shape[0]
        self.ny   =self.shape[1]
        self.lenx =sidelengths[0]
        self.leny =sidelengths[1]
        self.dx   =self.nx/self.lenx
        self.dy   =self.ny/self.leny
        self.v    =np.zeros(self.shape) # init solution to laplacian
        self.rho  =np.zeros(self.shape) # init charges
        self.recip_r=None # init the recip of r array
        self.set_recip_r() # fills reciprical of r array
        self.laplace_ker=None # init laplacian kernel
        self.set_laplace_kernel() # fill self.laplace_ker
        return
    def mask_checks(self):
        # perform checks on masks
        for m in self.masks: 
            assert m.shape==self.shape, "masks must all have same shape" 
        # make sure no masks intersect
        mask_union=np.full(self.shape, False)
        for m in self.masks:
            assert (mask_union*m).flatten().any()==False, "masks intersect!"
            mask_union|=m # take the union
        return
    def set_boundary_conditions(self,bd_condition="box"):
        if bd_condition=="box":
            # add a mask with v=0
            mask_edge=np.full(self.shape, False)
            mask_edge[:,0] =True
            mask_edge[:,-1]=True
            mask_edge[0,:] =True
            mask_edge[-1,:]=True
            v_edge         =0.0
            self.masks=self.masks+[mask_edge]
            self.volts=self.volts+[v_edge]
        # if it's circulant, no need to do anything
        return
    def set_recip_r(self):
        # x and y should never be zero
        x=np.linspace(-self.lenx/2,self.lenx/2,self.nx) #+ self.dx/2 # +dx/2 puts charges between grid spacing
        x=fftshift(x) # center zero on the corner
        X=np.outer(x,np.ones(self.ny))
        y=np.linspace(-self.lenx/2,self.leny/2,self.ny) #+ self.dy/2 # +dy/2 puts charges between grid spacing
        y=fftshift(y) # center zero on the corner
        Y=np.outer(np.ones(self.nx),y)
        #if 0.0 in x and 0.0 in y: Warning("Zero value encountered in r")
        self.recip_r=1/np.sqrt(X*X+Y*Y) # Define the reciprical of r
        return
    @staticmethod 
    def convolve2(a:np.ndarray, b:np.ndarray) -> np.ndarray:
        # Convloves a with b, two 2d arrays
        A,B=rfft2(a),rfft2(b)
        return irfft2(A*B) 
    def set_laplace_kernel(self):
        self.laplace_ker=np.zeros((3,3)) # First order laplacian kernel
        self.laplace_ker[0,1]=1/self.dx**2 # d/dx^2
        self.laplace_ker[2,1]=1/self.dx**2 
        self.laplace_ker[1,0]=1/self.dy**2
        self.laplace_ker[1,2]=1/self.dy**2
        self.laplace_ker[1,1]=2/self.dx**2 + 2/self.dy**2
        return 
    def set_v_bd(self):
        # Force the voltages to be the values specified on boundary
        for mask,v in zip(self.masks,self.volts):
            self.v[mask]=v
        return
    def update_rho(self):
        # Compute charge density on boundary
        laplacian=convolve2d(self.v, self.laplace_ker, 
                mode='same', boundary='wrap')
        laplacian=np.roll(laplacian,-1,0)
        laplacian=np.roll(laplacian,-1,1)
        print(f"\nDEBUG: {laplacian.shape}")
        # TODO: might have to np.roll
        for mask in self.masks:
            # weighted average of a,b
            def avg(a,b,wa=0.9):
                wb=1-wa
                return wa*a + wb*b
            self.rho[mask]=avg(self.rho[mask],-laplacian[mask])
        return 
    def update_v(self):
        # update the voltage by solving convolution
        self.v=LaplaceVoltage.convolve2(self.rho, self.recip_r)
        return
    def solve_step(self):
        # solve the laplace equation, 1 iteration
        self.set_v_bd() # Set the voltage on the boundary
        self.update_rho() # compute rho that does that
        self.update_v() # compute the potential of these charges
        return 

    def tests(self):
        return
            

if __name__=="__main__":
    import matplotlib.pyplot as plt
    shape=np.array([100,100])
    sidelengths=(100,100)
    rho=np.zeros(shape)
    mask1=np.full(shape, False)
    v1=1.0
    mask2=np.full(shape, False)
    v2=-1.0
    for i in range(30,70):
        mask1[i,40]=True
        mask2[i,60]=True
    masks=[mask1,mask2]
    volts=[v1,v2]
    # Initiate solver
    p=LaplaceVoltage(masks,volts,sidelengths)
    
    plt.clf()
    plt.figure()
    for i in range(10):
        p.set_v_bd()
        plt.clf()
        plt.title("Set v bd")
        plt.imshow(p.v)
        plt.colorbar()
        plt.pause(.5)

        p.update_rho()
        plt.clf()
        plt.title("Update rho")
        plt.imshow(p.v)
        plt.colorbar()
        plt.pause(.3)

        p.update_v()
        plt.clf()
        plt.title("Update Voltage")
        plt.imshow(p.v)
        plt.colorbar()
        plt.pause(.3)


#
#class LaplaceCharges:
#    def __init__(self, rho:np.ndarray, sidelengths:tuple=(100,100)):
#        """
#        rho : np.ndarray
#            rho is the charge density. The grid is determined by the shape of rho.
#        """
#        self.lenx=sidelengths[0]
#        self.leny=sidelengths[1]
#        self.nx=rho.shape[0]
#        self.ny=rho.shape[1]
#        self.rho=rho.copy()
#        self.dx=self.nx/self.lenx
#        self.dy=self.nx/self.leny
#        self.V=np.empty((self.nx,self.ny)) # init voltage array
#        self.recip_r=None # Declare and 
#        self.compute_recip_r() # initiate self.recip_r
#    def compute_recip_r(self):
#        # x and y should never be zero
#        x=np.linspace(-self.lenx/2,self.lenx/2,self.nx) #+ self.dx/2 # +dx/2 puts charges between grid spacing
#        x=fftshift(x) # center zero on the corner
#        X=np.outer(x,np.ones(self.ny))
#        y=np.linspace(-self.lenx/2,self.leny/2,self.ny) #+ self.dy/2 # +dy/2 puts charges between grid spacing
#        y=fftshift(y) # center zero on the corner
#        Y=np.outer(np.ones(self.nx),y)
#        #if 0.0 in x and 0.0 in y: Warning("Zero value encountered in r")
#        self.recip_r=1/np.sqrt(X*X+Y*Y) # Define the reciprical of r
#        return
#    @staticmethod 
#    def convolve2(a:np.ndarray, b:np.ndarray) -> np.ndarray:
#        # Convloves a with b, two 2d arrays
#        A,B=rfft2(a),rfft2(b)
#        return irfft2(A*B) 
#    def compute_voltage(self):
#        self.V=LaplaceCharges.convolve2(self.rho, self.recip_r)
#        return 
#
#
#if __name__=="__main__":
#    import matplotlib.pyplot as plt
#    shape=np.array([150,150])
#    sidelengths=(100,100)
#    rho=np.zeros(shape)
#    mask=np.empty()
#    rho[50,75]=1.0
#    rho[51,75]=1.0
#    rho[52,75]=1.0
#    rho[53,75]=1.0
#    rho[54,75]=1.0
#    #rho[mask.T]=1.0
#    laplace=LaplaceCharges(rho,sidelengths)
#    laplace.compute_voltage()
#    plt.figure()
#    plt.imshow(laplace.V)
#    plt.show(block=True)





