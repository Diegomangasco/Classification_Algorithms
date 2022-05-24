import pylab
import numpy

llrs = numpy.load('commedia_llr_infpar.npy')
Labels = numpy.load('commedia_labels_infpar.npy')

thresholds = numpy.array(llrs)
thresholds.sort()
thresholds = numpy.concatenate([numpy.array([-numpy.inf]), thresholds, numpy.array([numpy.inf])])
FPR = numpy.zeros(thresholds.size)
TPR = numpy.zeros(thresholds.size)

for idx, t in enumerate(thresholds):
    Pred = numpy.int32(llrs>t)
    Conf = numpy.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            Conf[i, j] = ((Pred == i)*(Labels == j)).sum()
    TPR[idx] = Conf[1, 1] / (Conf[1, 1] + Conf[0, 1])
    FPR[idx] = Conf[1, 0] / (Conf[1, 0] + Conf[0, 0])

pylab.plot(FPR, TPR)
pylab.show()