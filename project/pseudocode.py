from numpy.fft import rfft,irfft,rfft2,irfft2


class Grid:
    def __init__(self,shape:tuple,boundary:np.ndarray): 
        assert boundary.shape()=shape
        self.domain = np.zeros((m,n))
        self.boundary = boundary.copy() # deep-copy, no dubious user behaviour
        return

class LaplaceGrid:
    def __init__(self,grid:Grid):
        self.grid = grid
        return

class Shape:
    names=["circle","line segment","square"]
    def __init__(self): 
        self.perimiter=0

# some chapes
class Circle(Shape):
    def __init__(self,c:np.ndarray,r:float):
        self.c=c.copy() # center of the circle
        self.r=r
    def distance(self,p:np.ndarray):
        # returns distance from self to p
        vec_pc=self.c-p # the vector from p to c
        return np.sqrt(vec_pc@vec_pc)

class LineSeg(Shape):
    def __init__(self,a:np.ndarray,b:np.ndarray):
        assert a.shape==(2,) and b.shape==(2,)
        self.a=a.copy()
        self.b=b.copy()
    def distance(self,z:np.ndarray):
        # get distance from point z to instance
        ab=b-a # vector from a to b
        za=a-z # vector from z to a
        zb=b-z # vector from z to b
        if za@ab >= 0: # z is 'under' the ab's orthogonal sweep
            return np.sqrt(za@za)
        elif zb@ab <= 0: # z is 'over' the ab's orthogonal sweep
            return np.sqrt(zb@zb)
        # Otherwise, compute distance to perp
        return np.sqrt(za@za-(za@ab)**2/ab@ab)
        

class Square(Shape):
    def __init__(self,c:np.ndarray,l:float):
        self.c=c # center of the square
        self.l=l # sidelength
        self.line_segs=None # init line segments that make the square
        self.get_line_segs() # fills self.line_segs
    def get_line_segs(self):
        a=c+np.array([l/2,l/2])
        b=c+np.array([l/2,-l/2])
        l1=LineSeg(a,b)
        a,b=b,c+np.array([-l/2,-l/2])
        l2=LineSeg(a,b)
        a,b=b,c+np.array([-l/2,l/2])
        l3=LineSeg(a,b)
        a,b=b,c+np.array([l/2,l/2])
        l4=LineSeg(a,b)
        self.line_segs=[l1,l2,l3,l4]
        return
    def distance(self,p:np.ndarray): # returns distance from p to square
        return min([l.distance(,p) for l in self.line_segs])
        
        

# class for boundaries of multiple shapes
class Boundary:
    def __init__(self,boundary:list):
        self.boundary=boundary # a list of Shapes
        self.distance_matrix=None
    def distance(self,p):
        # compute distance from p to the boundary
        d=np.inf
        for obj in self.boundary:
            # compute distance from segment to boundary
            d=min(obj.distance(p))
        return d
    def compute_distance_matrix(self,gridX:np.ndarray,gridY:np.ndarray):
        # takes 2d grid and computes distance to each point in grid
        print("INFO: filling distance matrix...",end=" ")
        assert gridX.shape==gridY.shape
        shape=gridX.shape # this is the shape of the grid of points for dist matrix
        distance_matrix=np.empty(shape) # initiate distance matrix
        # compute the distance to each point in the grid
        for i in range(shape[0]):
            for j in range(shape[1]):
                x,y=gridX[i,j],gridY[i,j]
                dist=self.distance(np.array([x,y]))
                distance_matrix[i,j]=dist
        print("Done")
        return 
        



        



class LaplaceMCMC:
    def __init__(self,boundary:Boundary):
        self.boundary=boundary # a Boundary object

    

