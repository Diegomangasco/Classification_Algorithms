import numpy
import scipy.special

llCond = numpy.load('commedia_ll.npy')
llJoint = llCond + numpy.log(1.0/3.0)
Labels = numpy.load('commedia_labels.npy')
llMarginal = scipy.special.logsumexp(llJoint, axis = 0)
Post = numpy.exp(llJoint-llMarginal)
Pred = numpy.argmax(Post, axis=0)
Conf = numpy.zeros((3, 3))
for i in range(3):
    for j in range(3):
        Conf[i, j] = ((Pred == i)*(Labels == j)).sum()

print(Conf)