import numpy as np
import pytest
from scripts.prepare_cuda_fixed_cube_bounded_worker import HELPER

namespace={"np":np};exec(HELPER,namespace);fingerprint=namespace['numeric_sensitivity_fingerprint']

def test_numeric_equality_normalizes_signedzero_dtype_endian_and_chunk_boundaries():
 a=np.array([0.,-0.,1.,-2.,np.finfo(np.float32).tiny],np.float32)
 b=a.astype('>f8');b[0]=-0.;b[1]=0.
 assert np.array_equal(a,b)
 assert fingerprint(a,2)==fingerprint(b,3)
 assert fingerprint(a[::-1],1)==fingerprint(b[::-1],4)

def test_real_differences_shape_and_smallest_nonzero_are_detected():
 a=np.zeros(6,np.float32);b=a.copy();b[3]=np.nextafter(np.float32(0),np.float32(1))
 assert fingerprint(a)!=fingerprint(b)
 assert fingerprint(a)!=fingerprint(a.reshape(2,3))

@pytest.mark.parametrize('value',(np.nan,np.inf,-np.inf))
def test_nonfinite_sensitivity_cannot_create_false_positive_live_signal(value):
 with pytest.raises(ValueError,match='Nonfinite'):fingerprint(np.array([value],np.float32))
