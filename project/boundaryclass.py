import numpy as np

# Bounding volume hierarchy
class Bvh:
    def __init__(lines):
        # Split the lines up into segments
        self.n_segments=0
        segments=[]
        for line in lines:
            for i,j in zip(line[:-1],line[1:]):
                segments.append((i,j))
                self.n_segments+=1
        self.segments=np.array(segments)


    def create_bvh(self):
        # helper, return smallest rectangle that contains segments
        def get_bounding_volume(segments):
            # segments have shape (n_segments,dim_space)
            # returns [[xmin,ymin,zmin],[xmax,ymax,zmax]]
            bd_vol=np.vstack(
                    [np.min(segments,axis=0),
                    np.max(segments,axis=0)])
            return bd_vol

        # reccursively create bvh
        def create_bvh_rec(segments):
            bd_vol=get_bounding_volume(segments)
            if len(segments)==1:
                return bd_vol,segments[0]
            # dummy, randomly put them into some order
            # partition segments into two groups
            # TODO: implement partitioning algorithm
            subset0=segments[:len(segments)//2]
            subset1=segments[len(segments)//2:]
            bvh0=create_bch_rec(subset0)
            bvh1=create_bch_rec(subset1)
            return bd_vol,(bvh0,bvh1)
            

    # associated function, class method / unbound method
    def distance_to_segment(p:np.ndarray, segment:np.ndarray):
        """returns distance from point p to segment (a,b)"""
        def norm(vec):
            return np.sqrt(vec@vec)
        assert p.shape==(segment.shape[-1],) # dims must agree
        assert segment.shape[0]==2 # a segment has two points
        # nomenclature, segment has points a,b
        ab=segment[1]-segment[0] # vec form a to b
        bp=p-segment[1]
        ap=p-segment[0]
        if ab@bp>=0: # dist equal dist to b
            return norm(bp)
        elif ab@ap<=0: # dist equal dist to a
            return norm(ap)
        else:
            cos_sq_theta=(ab@ap)**2/(ab@ab * ap@ap) # cos squared of theta
            sin_theta=np.sqrt(1-cos_sq_theta)
            h=sin_theta*norm(ap) # height of isosolese triangle
            return h



class Boundary:
    def __init__(lines=[],margin:float=1.0):
        """Optionally provide a list of arrays of dots, 
        implicitly joined by lines"""
        self.lines=lines.copy() # a list of np arrays
        self.margin=1.0
        # Determine the bounding volume hierarchy
        self.bvh=

