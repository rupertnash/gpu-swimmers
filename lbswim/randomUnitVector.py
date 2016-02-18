"""RandomState -- numpy.random.RandomState, with a unitVector method
added.

unitVector -- produce random unit vectors without creating a
RandomState instance.
"""
import numpy as N

class RandomState(N.random.RandomState):
    """numpy.random.RandomState, with a unitVector method added."""
    def unitVector(self, dim=3, size=1):
        """Want a random unit vector with uniform probability.
        So, pick a random n-vector in [-1, 1]^n , reject if
        it's magnitude is > 1 (i.e. outside a sphere)
        then normalise."""
        if isinstance(size, int):
            size = (size,)
            
        ans = self.normal(size=size+(dim,))
        return ans/N.sqrt(N.sum(ans**2, axis=-1))[...,N.newaxis]
    

    def __reduce_ex__(self, int):
        supTup = super(RandomState, self).__reduce_ex__(int)
        ans = (self.__class__,) + supTup[1:]
        return ans
    
    
def unitVector(dim=3, size=1):
    """Want a random unit vector with uniform probability.
    So, pick a random n-vector in [-1, 1]^n , reject if
    it's magnitude is > 1 (i.e. outside a sphere)
    then normalise."""
    
    rs = RandomState()
    return rs.unitVector(dim, size)
