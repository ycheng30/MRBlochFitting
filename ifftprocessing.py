#Ifft processing module
from numpy import *
import cjlib
import scipy
from scipy import optimize
import scipy.fftpack

## Define FLEX Equation
def fitfuncComplexSuperLorGau(p, x): 

	tfid1 = zeros( size(x) ) + p[0]
	tfid2 = zeros( size(x) ) + p[0]
	
	for pooli in range( (len(p)-1)/6 ):
		firstVal = pooli * 6 + 1
		sp = 3*cos(p[firstVal+5])**2 - 1
		tfid1 = tfid1 + ((p[firstVal]*cos(2*pi*p[firstVal+1]*x+p[firstVal+2]))*exp(-p[firstVal+3]*fabs(x))*exp(-(p[firstVal+4]*(x)/sp)**2))
		tfid2 = tfid2 + ((p[firstVal]*cos(2*pi*p[firstVal+1]*x+(p[firstVal+2]-pi/2)))*exp(-p[firstVal+3]*fabs(x))*exp(-(p[firstVal+4]*(x)/sp)**2))
	
	return tfid1, tfid2


def fitfuncComplexLorGau(p, x): 
	
	tfid1 = zeros( size(x) ) + p[0]
	tfid2 = zeros( size(x) ) + p[0]
	
	for pooli in range( (len(p)-1)/5 ):
		firstVal = pooli * 5 + 1
		tfid1 = tfid1 + ((p[firstVal]*cos(2*pi*p[firstVal+1]*x+p[firstVal+2]))*exp(-p[firstVal+3]*fabs(x))*exp(-(p[firstVal+4]*(x))**2))
		tfid2 = tfid2 + ((p[firstVal]*cos(2*pi*p[firstVal+1]*x+(p[firstVal+2]-pi/2)))*exp(-p[firstVal+3]*fabs(x))*exp(-(p[firstVal+4]*(x))**2))
	
	return tfid1, tfid2

def fitfuncComplexLor(p, x): 
	
	tfid1 = zeros( size(x) ) + p[0]
	tfid2 = zeros( size(x) ) + p[0]
	
	for pooli in range( (len(p)-1)/4 ):
		firstVal = pooli * 4 + 1
		tfid1 = tfid1 + ((p[firstVal]*cos(2*pi*p[firstVal+1]*x+p[firstVal+2]))*exp(-p[firstVal+3]*fabs(x)))
		tfid2 = tfid2 + ((p[firstVal]*cos(2*pi*p[firstVal+1]*x+(p[firstVal+2]-pi/2)))*exp(-p[firstVal+3]*fabs(x)))
	
	return tfid1, tfid2

def FlexLSFitComplex(tevol, mm1, mm2, p0, fitfunc = 'Lor'):
	# Ensure type(p0) == list else do p0.tolist()
	if fitfunc =='Lor':
		errfunc = lambda p, x, y1, y2: (fitfuncComplexLor(p, x)[0] - y1)*(fitfuncComplexLor(p, x)[1] - y2)
	elif fitfunc =='LorGau':
		errfunc = lambda p, x, y1, y2: (fitfuncComplexLorGau(p, x)[0] - y1)*(fitfuncComplexLorGau(p, x)[1] - y2)
	elif fitfunc =='SuperLorGau':
		errfunc = lambda p, x, y1, y2: (fitfuncComplexSuperLorGau(p, x)[0] - y1)*(fitfuncComplexSuperLorGau(p, x)[1] - y2)
	p1, success = optimize.leastsq(errfunc, p0[:], args=(tevol, mm1, mm2))
	return p1

def FlexFuncMinComplex(tevol, mm1, mm2, p0, bounds, weighting = 500, fitfunc = 'Lor' ):

	def errfunc(p, x, y1, y2):
		p = p*scalings
#		w = tevol*weighting + 0.1
		w = tevol[-1]*weighting - tevol*weighting + 0.1
		if fitfunc =='Lor':
			dd = sum( w * (((fitfuncComplexLor(p, x)[0] - y1) + (fitfuncComplexLor(p, x)[1] - y2))**2)) # Distance to the target function
		elif fitfunc =='LorGau':
			dd = sum( w * (((fitfuncComplexLorGau(p, x)[0] - y1) + (fitfuncComplexLorGau(p, x)[1] - y2))**2)) # Distance to the target function
		elif fitfunc =='SuperLorGau':
			dd = sum( w * (((fitfuncComplexSuperLorGau(p, x)[0] - y1) + (fitfuncComplexSuperLorGau(p, x)[1] - y2))**2)) # Distance to the target function
		return dd

	scalings = abs(p0)
	
	# re-scale the bounds
	bounds_rescaled = []
	for bndi, bnd in enumerate( bounds ):
		bounds_rescaled.append( (bnd[0]/scalings[bndi], bnd[1]/scalings[bndi] ) )
	p1 = optimize.fmin_l_bfgs_b(errfunc, p0/scalings, args=(tevol, mm1, mm2), bounds=bounds_rescaled, approx_grad=True, factr = 1e-4)[0]
	return p1*scalings

def GetFitComponenetsLorGau(FitData, tevol):
	Components = zeros((2, len(FitData)/5, len(tevol)))
	counter=1
	for ii in range(len(FitData)/5):
		Components[0, ii] = ifftshift((FitData[counter]*cos(2*pi*FitData[counter+1]*tevol+FitData[counter+2]))*exp(-FitData[counter+3]*abs(tevol))*exp(-(FitData[counter+4]*tevol)**2))
		Components[1, ii] = ifftshift((FitData[counter]*cos(2*pi*FitData[counter+1]*tevol+(FitData[counter+2]-pi/2)))*exp(-FitData[counter+3]*abs(tevol))*exp(-(FitData[counter+4]*abs(tevol))**2))
		counter = counter+5
	return Components

def GetFitComponenets(FitData, tevol, fitfunc = 'Lor'):
	if fitfunc == 'Lor':
		Components = zeros((2, len(FitData)/4, len(tevol)))
		counter=1
		for ii in range(len(FitData)/4):
			Components[0, ii] = scipy.fftpack.ifftshift((FitData[counter]*cos(2*pi*FitData[counter+1]*tevol+FitData[counter+2]))*exp(-FitData[counter+3]*abs(tevol)))
			Components[1, ii] = scipy.fftpack.ifftshift((FitData[counter]*cos(2*pi*FitData[counter+1]*tevol+(FitData[counter+2]-pi/2)))*exp(-FitData[counter+3]*abs(tevol)))
			counter = counter+4
	elif fitfunc == 'LorGau':
		Components = zeros((2, len(FitData)/5, len(tevol)))
		counter=1
		for ii in range(len(FitData)/5):
			Components[0, ii] = scipy.fftpack.ifftshift((FitData[counter]*cos(2*pi*FitData[counter+1]*tevol+FitData[counter+2]))*exp(-FitData[counter+3]*abs(tevol))*exp(-(FitData[counter+4]*tevol)**2))
			Components[1, ii] = scipy.fftpack.ifftshift((FitData[counter]*cos(2*pi*FitData[counter+1]*tevol+(FitData[counter+2]-pi/2)))*exp(-FitData[counter+3]*abs(tevol))*exp(-(FitData[counter+4]*abs(tevol))**2))
			counter = counter+5
	elif fitfunc == 'SuperLorGau':
		Components = zeros((2, len(FitData)/6, len(tevol)))
		counter=1
		suf = fabs(3*cos(FitData[counter+5])**2 - 1)
		for ii in range(len(FitData)/6):
			Components[0, ii] = scipy.fftpack.ifftshift((FitData[counter]*cos(2*pi*FitData[counter+1]*tevol+FitData[counter+2]))*exp(-FitData[counter+3]*abs(tevol))*exp(-(FitData[counter+4]*abs(tevol)*(1/suf))**2))
			Components[1, ii] = scipy.fftpack.ifftshift((FitData[counter]*cos(2*pi*FitData[counter+1]*tevol+(FitData[counter+2]-pi/2)))*exp(-FitData[counter+3]*abs(tevol))*exp(-(FitData[counter+4]*abs(tevol)*(1/suf))**2))
			counter = counter+6
	return Components

def PrintResults(p, fitfunc = 'Lor'):
	print 'DC correction:', round(p[0], 3)
	if fitfunc == 'Lor':
		for pooli in range((len(p)-1)/4):
			firstVal = pooli * 4 + 1
			print 'Frequency: {freq:3.0f} Hz \t Amplitude: {amp:5.2f} % \t Phase: {phase:3.2f} \t k+R2: {decay:4.0f} s-1'.format(freq = p[firstVal+1], amp = p[firstVal]*100, phase = p[firstVal+2], decay = p[firstVal+3])
	elif fitfunc == 'LorGau':
		for pooli in range((len(p)-1)/5):
			firstVal = pooli * 5 + 1
			print 'Frequency: {freq:3.0f} Hz \t Amplitude: {amp:5.2f} % \t Phase: {phase:3.2f} \t k+R2: {decay1:4.0f} s-1 \t k+R2: {decay2:4.0f} s-1'.format(freq = p[firstVal+1], amp = p[firstVal]*100, phase = p[firstVal+2], decay1 = p[firstVal+3], decay2 = p[firstVal+4])
	elif fitfunc == 'SuperLorGau':
		for pooli in range((len(p)-1)/6):
			firstVal = pooli * 6 + 1
			print 'Frequency: {freq:3.0f} Hz \t Amplitude: {amp:5.2f} % \t Phase: {phase:3.2f} \t k+R2: {decay1:4.0f} s-1 \t k+R2: {decay2:4.0f} s-1 \t k+R2: {freq2:4.4f} s-1'.format(freq = p[firstVal+1], amp = p[firstVal]*100, phase = p[firstVal+2], decay1 = p[firstVal+3], decay2 = p[firstVal+4], freq2 = p[firstVal+5])

