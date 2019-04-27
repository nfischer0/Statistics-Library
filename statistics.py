import sympy
import matplotlib.pyplot as plt
import numpy as np
import math
from math import ceil

def F(xaxis, f):
    '''accepts xaxis numbers and function f, returns y-vals in range of xaxis'''
    x = sympy.Symbol('x')
    y = []
    for num in xaxis:
        y.append(float(f.subs(x, num)))
    return y

def invNorm(area, mu = 0, sigma = 1):
    '''returns place on ppf of area (in other words the z-score of an area along N(mu,sigma) )'''
    x = sympy.Symbol('x')
    z = sympy.Symbol('z')
    n = sympy.exp(-1 * (x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * sympy.pi))
    inv = sympy.solve(-1 * area + sympy.integrate(n, (x, -1 * sympy.oo, z)), z)
    return inv[0]

def graphPPF(mu = 0, sigma = 1, lowerBound=.00001, upperBound=.99999, numPoints=100):
    '''graphs the percent-point function (inverse of CDF) for a given mu, sigma.
    numPoints is set to 100 for time efficiency only'''
    xaxis = np.linspace(lowerBound, upperBound, numPoints)
    yaxis = []
    for val in xaxis:
         yaxis.append(invNorm(val, mu, sigma))

    plt.plot(xaxis, yaxis)
    plt.title('Percent-Point Function')
    plt.ylabel('Z Score')
    plt.xlabel('Area (P(x))')
    plt.show()


def graphNormalComparison(mu, sigma, alpha, mu0=0, sigma0=1, lowerBound=-5, upperBound=5):
    '''graphs standared N(0,1) or modified F(mu0, sigma0) in dashed red and new transformed normal curve in blue for comparison'''
    x = sympy.Symbol('x')

    n = sympy.exp(-1 * (x - mu0)**2 / (2*sigma0**2)) / (sigma0 * math.sqrt(2 * sympy.pi)) #standard normal curve
    f = sympy.exp(-1 * (x - mu)**2 / (2*sigma**2)) / (sigma * math.sqrt(2 * sympy.pi)) #custom normal curve
    xaxis = np.linspace(lowerBound,upperBound,10000)
    plt.plot(xaxis, F(xaxis, f), color='r') #Ha
    plt.plot(xaxis, F(xaxis, n), color='b') #Ho
    if mu < mu0:
        z = float(invNorm(alpha, mu=mu0, sigma=sigma0))
        plt.vlines(z, 0, 1, colors='k', linestyles='dashed', alpha=.4)
        hoTail = F(np.linspace(lowerBound, z), n)
        haTail = F(np.linspace(z, upperBound), f)
        plt.fill_between(np.linspace(lowerBound, z), hoTail, color='b', alpha=.6)  # type 1
        plt.fill_between(np.linspace(z, upperBound), haTail, color='r', alpha=.6)  # type 2
        t2 = float(sympy.integrate(f, (x, z, upperBound)))
    else:
        z = float(invNorm(1 - alpha, mu=mu0, sigma=sigma0))
        plt.vlines(z, 0, 1, colors='k', linestyles='dashed', alpha=.4)
        hoTail = F(np.linspace(z, upperBound), n)
        haTail = F(np.linspace(lowerBound, z), f)
        plt.fill_between(np.linspace(z, upperBound), hoTail, color='b', alpha=.6)  # type 1
        plt.fill_between(np.linspace(lowerBound, z), haTail, color='r', alpha=.6)  # type 2
        t2 = float(sympy.integrate(f, (x, lowerBound, z)))
    plt.text(mu0 - .02, .41 / sigma0, 'Ho')
    plt.text(mu - .02, .41 / sigma, 'Ha')
    plt.text(z + .3, .52*sigma, '$ \\alpha $  = ' + str(alpha))
    plt.text(lowerBound, 1, 'P(T1 Error) = ' + str(alpha))
    plt.text(lowerBound, .8, 'P(T2 Error) = ' + str(round(t2, 4)))
    plt.show()

def graphNorm_CDF(z1, z2, mu = 0, sigma = 1, lowerBound=-3, upperBound=3):
    '''graphs N(mu, sigma) and shades and integrates the area between z scores'''
    x = sympy.Symbol('x')
    f = sympy.exp(-1 * (x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * sympy.pi))
    xaxis = np.linspace(lowerBound, upperBound, 1000)
    plt.plot(xaxis, F(xaxis, f))
    if lowerBound < z1 < upperBound:
        if lowerBound < z2 < upperBound:
            range = np.linspace(z1, z2, 1000)
        else:
            range = np.linspace(z1, upperBound, 1000)
    elif lowerBound < z2 < upperBound:
        range = np.linspace(lowerBound, z2, 1000)
    integral = float(sympy.integrate(f, (x, z1, z2)))

    funcTail = F(range, f)
    plt.fill_between(range, funcTail, alpha=.7)
    plt.xlabel('Z-Score')
    plt.text(mu - .9*sigma, .41*sigma, 'P(' + str(z1) + ' < x < ' + str(z2) +') = ' + str(integral))
    plt.title('Integral from ' + str(z1) + ' to ' + str(z2))
    plt.axis([lowerBound, upperBound, 0, sigma])
    plt.show()


def norm_CDF(z1, z2, mu = 0, sigma = 1):
    '''integrates N(mu, sigma) between z1 and z2'''
    x = sympy.Symbol('x')
    f = sympy.exp(-1 * (x - mu) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * sympy.pi))
    integral = float(sympy.integrate(f, (x, z1, z2)))
    return integral

def graphT_PDF(df, lowerBound=-5, upperBound=5):
    "graphs the student t distribution given the degrees of freedom"
    x = sympy.Symbol('x')
    f = (sympy.factorial( ((df+1) / 2) - 1) * (1 + (x**2 / df))**(-1 * ( (df+1) / 2)) )  / (sympy.factorial( (df / 2 ) - 1) * ((df*sympy.pi)**.5))
    xaxis = np.linspace(lowerBound, upperBound, 1000)
    plt.plot(xaxis, F(xaxis, f))
    plt.show()

def t_CDF(t1, t2, df):
    """integrates the student t distribution from t1 to t2 with given degrees of freedom
    This method relies on sympy.nsimplify which turns decimals into fractions allowing it to be properly integrated
    *(This used to rely on Riemann sums due to the difficulty/computational intensity of the integral)*"""
    x = sympy.Symbol('x')
    f = (sympy.factorial(((df + 1) / 2) - 1) * (1 + ((x ** 2) / df)) ** (-1 * ((df + 1) / 2))) / (sympy.factorial((df / 2) - 1) * ((df * sympy.pi) ** .5))
    f = sympy.nsimplify(f)
    integral = sympy.Integral(f, (x, t1, t2))
    #integral = integral.as_sum(method='trapezoid', n=30000) #DEPRECATED computes trapezoidal riemann sum
    return float(integral)

def graphT_CDF(t1, t2, df, lowerBound=-5, upperBound=5):
    """graphs the student t distribution with given degrees of freedom, calculates the integral (CDF)
    from t1 to t2, and shades that area on the returned graph"""
    x = sympy.Symbol('x')
    f = (sympy.factorial(((df + 1) / 2) - 1) * (1 + ((x ** 2) / df)) ** (-1 * ((df + 1) / 2))) / (sympy.factorial((df / 2) - 1) * ((df * sympy.pi) ** .5))
    f = sympy.nsimplify(f)
    integral = t_CDF(t1, t2, df)
    lowerBound = -5
    upperBound = -1 * lowerBound
    xAxis = np.linspace(lowerBound, upperBound, 1000)
    plt.plot(xAxis, F(xAxis, f))
    range = []
    if lowerBound < t1 < upperBound:
        if lowerBound < t2 < upperBound:
            range = np.linspace(t1, t2, 1000)
        else:
            range = np.linspace(t1, upperBound, 1000)
    elif lowerBound < t2 < upperBound:
        range = np.linspace(lowerBound, t2, 1000)
    funcTail = F(range, f)
    plt.fill_between(range, funcTail, alpha=.7)
    plt.title('Student-t distribution with df = ' + str(df))
    plt.text(-2, float(f.subs(x, 0)) + .005, 'P(' + str(lowerBound) + ' < x < ' + str(upperBound) + ') = ' +str(integral))
    plt.show()

def compare_T(df, num, lowerBound=-5, upperBound=5):
    """graphs num number of comparisons with df / num number of distributions"""
    step = float( df / num )
    x = sympy.Symbol('x')
    d = sympy.Symbol('d')
    up = ceil(num / 2)
    down = num - up
    f = (sympy.factorial(((d + 1) / 2) - 1) * (1 + (x ** 2 / d)) ** (-1 * ((d + 1) / 2))) / (sympy.factorial((d / 2) - 1) * ((d * sympy.pi) ** .5))
    xaxis = np.linspace(lowerBound, upperBound, 1000)
    dfStepUp = df + step
    dfStepDown = df - step
    plt.plot(xaxis, F(xaxis, f.subs(d, df)))
    while up > 0:
        plt.plot(xaxis, F(xaxis, f.subs(d, dfStepUp)))
        dfStepUp += step
        up -= 1
    while down > 0:
        plt.plot(xaxis, F(xaxis, f.subs(d, dfStepDown)))
        dfStepDown -= step
        down -= 1
    plt.show()

def chiSquared_CDF(X, df):
    x = sympy.Symbol('x')
    f = ((x ** ((df / 2) - 1)) * sympy.exp(-1 * x / 2)) / (2 ** (df / 2) * sympy.factorial((df / 2) - 1))
    integral = sympy.integrate(f, (x, X, sympy.oo))
    integral = sympy.simplify(integral)
    return float(integral)

def graphChiSquared_PDF(df, lowerBound=0, upperBound=20):
    """graphs the student t PDF squared with given degrees of freedom"""
    x = sympy.Symbol('x')
    f = ( (x**( (df/2) - 1)) * sympy.exp(-1*x / 2)) / (2**(df/2) * sympy.factorial( (df/2) - 1))
    xAxis = np.linspace(lowerBound, upperBound, 1000)
    plt.plot(xAxis, F(xAxis, f))
    plt.title('$\chi^{2}$ distribution with df = ' + str(df))
    plt.show()


def graphChiSquared_CDF(X, df, lowerBound=0, upperBound=20):
    x = sympy.Symbol('x')
    f = ((x ** ((df / 2) - 1)) * sympy.exp(-1 * x / 2)) / (2 ** (df / 2) * sympy.factorial((df / 2) - 1))
    xAxis = np.linspace(lowerBound, upperBound, 1000)
    integral = chiSquared_CDF(X, df)
    plt.plot(xAxis, F(xAxis, f))
    range = np.linspace(X, upperBound, 1000)
    funcTail = F(range, f)
    plt.fill_between(range, funcTail, alpha=.7)
    plt.title('$\chi^{2}$ distribution with df = '+ str(df))
    plt.text(xAxis[500], float(f.subs(x, 1)), 'P($\chi^{2}$ > ' + str(X) + ') = ' + str(integral))
    plt.show()

def compareChiSquared(df, num, lowerBound=0, upperBound=20):
    step = float(df / num)
    x = sympy.Symbol('x')
    d = sympy.Symbol('d')
    up = ceil(num / 2)
    down = num - up
    f = ((x ** ((d / 2) - 1)) * sympy.exp(-1 * x / 2)) / (2 ** (d / 2) * sympy.factorial((d / 2) - 1))
    xaxis = np.linspace(lowerBound, upperBound, 1000)
    dfStepUp = df + step
    dfStepDown = df - step
    plt.plot(xaxis, F(xaxis, f.subs(d, df)))
    while up > 0:
        plt.plot(xaxis, F(xaxis, f.subs(d, dfStepUp)))
        dfStepUp += step
        up -= 1
    while down > 0:
        plt.plot(xaxis, F(xaxis, f.subs(d, dfStepDown)))
        dfStepDown -= step
        down -= 1
    plt.show()