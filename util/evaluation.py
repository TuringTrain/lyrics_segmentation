import sys
import logging
import warnings
import numpy as np

_module = sys.modules['__main__'].__file__
_logger = logging.getLogger(_module)


def pr_measures(expected, predicted):

    n = len(expected)
    assert n == len(predicted)
    
    different = expected != predicted
    not_nil = expected >= 0
    not_abstain = predicted >= 0
    
    fp = (different & not_abstain).sum()
    fn = (different & not_nil).sum()
    tp = not_nil.sum() - fn
    
    #    tp = np.logical_and(expected == predicted, expected >= 0).sum()
    #    fp = np.logical_and(expected != predicted, predicted >= 0).sum()
    #    fn = np.logical_and(expected != predicted, expected >= 0).sum()
    #    tn = np.logical_and(expected == predicted, expected < 0).sum()
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = tp / (tp + fp)
        r = tp / (tp + fn)
        f1 = 2 * p * r / (p + r)

    return np.array([p, r, f1])
    

def pr_fscore(p, r, beta=1):
    beta2 = beta * beta
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nan_to_num((1 + beta2) * p * r / (beta2 * p + r))

    
def pr_approximate_randomization(expected, predicted1, predicted2, iterations=1000):
     
    n = len(expected)
    assert n == len(predicted1)
    assert n == len(predicted2)

    def compare (e, p):
        tp = ((e == p) & (e >= 0)).astype(int)
        fp = ((e != p) & (p >= 0)).astype(int)
        fn = ((e != p) & (e >= 0)).astype(int)
        return tp, fp,fn

    def test_statistics(tp1, fp1, fn1, tp2, fp2, fn2):
        tp1c = tp12c + tp1.sum()
        tp2c = tp12c + tp2.sum()
        fp1c = fp12c + fp1.sum()
        fp2c = fp12c + fp2.sum()
        fn1c = fn12c + fn1.sum()
        fn2c = fn12c + fn2.sum()
        if (tp1c == 0):
            p1 = r1 = f1 = 0
        else:
            p1 = tp1c / (tp1c + fp1c)
            r1 = tp1c / (tp1c + fn1c)
            f1 = 2 * p1 * r1 / (p1 + r1)
        if (tp2c == 0):
            p2 = r2 = f2 = 0
        else:
            p2 = tp2c / (tp2c + fp2c)
            r2 = tp2c / (tp2c + fn2c)
            f2 = 2 * p2 * r2 / (p2 + r2)
        return np.abs(np.array([p1 - p2, r1 - r2, f1 - f2]))

    e  = np.array(expected)
    p1 = np.array(predicted1)
    p2 = np.array(predicted2)
   
    sp = (p1 == p2) | ((p1 != e) & (p2 != e) & (p1 >= 0) & (p2 >= 0))
    dp = ~sp
    de = e[dp]
    m  = len(de)
    
    tp12,  fp12,  fn12  = compare(e[sp], p1[sp])
    tp12c, fp12c, fn12c = tp12.sum(), fp12.sum(), fn12.sum()

    tp1, fp1, fn1 = compare(de, p1[dp])
    tp2, fp2, fn2 = compare(de, p2[dp])
    
    tref = test_statistics(tp1, fp1, fn1, tp2, fp2, fn2)

    tp1c = tp12c + tp1.sum()
    tp2c = tp12c + tp2.sum()
    fp1c = fp12c + fp1.sum()
    fp2c = fp12c + fp2.sum()
    fn1c = fn12c + fn1.sum()
    fn2c = fn12c + fn2.sum()
    
    tpd = tp1 - tp2
    fpd = fp1 - fp2
    fnd = fn1 - fn2
     
    np.random.seed(n)
    r = np.zeros(len(tref))
    
    for i in range(0, iterations):
        mask = np.random.randint(low=0, high=2, size=m)
        tpdm = np.inner(tpd, mask)
        fpdm = np.inner(fpd, mask)
        fndm = np.inner(fnd, mask)
        
        tp3c = tp1c - tpdm
        tp4c = tp2c + tpdm
        fp3c = fp1c - fpdm
        fp4c = fp2c + fpdm
        fn3c = fn1c - fndm
        fn4c = fn2c + fndm
        
        if (tp3c == 0):
            p3 = r3 = f3 = 0
        else:
            p3 = tp3c / (tp3c + fp3c)
            r3 = tp3c / (tp3c + fn3c)
            f3 = 2 * p3 * r3 / (p3 + r3)
        if (tp4c == 0):
            p4 = r4 = f4 = 0
        else:
            p4 = tp4c / (tp4c + fp4c)
            r4 = tp4c / (tp4c + fn4c)
            f4 = 2 * p4 * r4 / (p4 + r4)
        t = np.abs(np.array([p3 - p4, r3 - r4, f3 - f4]))

#        mask = np.random.randint(low=0, high=2, size=m)
#        tpdm = tpd * mask; tp3 = tpdm + tp2; tp4 = tp1 - tpdm
#        fpdm = fpd * mask; fp3 = fpdm + fp2; fp4 = fp1 - fpdm
#        fndm = fnd * mask; fn3 = fndm + fn2; fn4 = fn1 - fndm
#        t = test_statistics(tp3, fp3, fn3, tp4, fp4, fn4)
        r += t >= tref            
    
    return (r + 1) / (iterations + 1)
    

def bootstrap_confidence_intervals(expected, predicted, measure_function, confidence=95, iterations=1000):
    
    n = len(expected)
    assert n == len(predicted)
    
    expected = np.asarray(expected)
    predicted = np.asarray(predicted)
    
    mref = measure_function(expected, predicted)
    m = len(mref)
    
    np.random.seed(n)
    
    msampled = np.zeros((iterations, m))
    for i in range(0, iterations):
        indexes = np.random.randint(low=0, high=n, size=n)
        e = expected[indexes]
        p = predicted[indexes]
        msampled[i,:] = measure_function(e, p)
    
    lower_percentile = (100 - confidence) / 2
    upper_percentile = 100 - lower_percentile
    
    lower_margin = mref - np.percentile(msampled, lower_percentile, axis=0)    
    upper_margin = np.percentile(msampled, upper_percentile, axis=0) - mref
    margin = np.max(np.stack((lower_margin, upper_margin)), axis=0)
    
    return np.stack((mref, margin, lower_margin, upper_margin))


def approximate_randomization(expected, predicted_baseline, predicted_system, measure_function, twosided=True, iterations=1000):
    
    n = len(expected)
    assert n == len(predicted_baseline)
    assert n == len(predicted_system)
    
    expected = np.array(expected)
    predicted_baseline = np.array(predicted_baseline)
    predicted_system = np.array(predicted_system)
    
    mb = measure_function(expected, predicted_baseline)
    ms = measure_function(expected, predicted_system)
    tref = np.abs(ms - mb)
    
    np.random.seed(n)
    r = np.zeros(len(tref))
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(0, iterations):
            indexes = np.random.randint(low=0, high=2, size=n)
            temp = (predicted_baseline - predicted_system) * indexes
            predicted1 = temp + predicted_system
            predicted2 = predicted_baseline - temp
            m1 = measure_function(expected, predicted1)
            m2 = measure_function(expected, predicted2)
            t = np.abs(m1 - m2)
            r += t >= tref
        
    pvalues = (r + 1) / (iterations + 1)
    if (not twosided):
        pvalues = pvalues / 2
      
    return pvalues