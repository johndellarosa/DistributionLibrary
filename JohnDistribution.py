import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import itertools
import math
import copy


def convolve_single_point(dist1, dist2, t):
    def integrand(x, t, f, g):
        return f.PDF(x) * g.PDF(t - x)

    return scipy.integrate.quad(integrand, -np.inf, np.inf, args=(t, dist1, dist2))[0]


def convolve(f, g, min_x, max_x, interval):
    x = np.arange(min_x, max_x + interval, interval)
    y = [convolve_single_point(f, g, t) for t in x]
    return np.array([x, y])

def exp(distribution):
    if type(distribution) == Gaussian:
        return Log_Normal(mu=distribution.mu, sigma=distribution.sigma)
    else:
        return 'Not defined for this input'


class Discrete_Distribution:

    def __init__(self):
        self.mean = 0
        self.variance = 0
        self.median = 0

    def PMF(self, x, *args, **kwargs):
        return 0


    def CDF(self, x, start=0):
        accum = 0
        for i in range(start, x+1):
            accum += self.PMF(i)
        return accum

    def plot(self, x, kind='scatter'):

        y = [self.PMF(i) for i in x]
        if kind == 'scatter':
            plt.scatter(x, y)
        elif kind == 'bar':
            plt.bar(x, y)
        return x,y
        # print(x)
        # print(y)


class Continuous_Distribution:
    #     def __init__():

    def __init__(self):
        self.mean = 0
        self.variance = 0
        self.median = 0
        self.kurtosis = 0
        self.skewness = 0


    def __add__(self, other):
        if issubclass(type(other), Continuous_Distribution):

            '''
            Treated as if it's two independent variables. Don't try to add same distribution to itself if you want to multiply
            '''

            if type(self) == Gaussian and type(other) == Gaussian:
                return Gaussian(mu=self.mu + other.mu, sigma=np.sqrt(self.variance + other.variance))

            else:
                return Convolution(self, other)
        elif type(other) == float or type(other) == int:
            if type(self) == Gaussian:
                return Gaussian(self.mu + other, self.sigma)
            elif type(self) == Laplace:
                return Laplace(self.mu + other, self.beta)
        else:
            return 0

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if type(other) == float or type(other) == int:
            if type(self) == Laplace:
                return Laplace(other * self.mu, abs(other) * self.b)
            elif type(self) == Gaussian:
                return Gaussian(mu=other * self.mu, sigma=abs(other) * self.sigma)
            return 'asdf'
        elif issubclass(type(other), Continuous_Distribution):

            return 'asdf'
        else:
            return 'asdf'

    def __rmul__(self, other):
        return self.__mul__(other)

    def PDF(self, x, *args, **kwargs):
        return 0

    def CDF(self, x):
        return 0

    def MGF(self, t):
        return 0



    def Quantile(self, p, min_x=0, stepsize=0.01):
        x = min_x  # change to min of support
        while (self.CDF(x) < p):
            x += stepsize
        return x

    def Sample(self):
        U = np.random.uniform()
        return self.Quantile(U)

    def plot(self, min_x, max_x, interval):
        x = np.arange(min_x, max_x + interval, interval)
        y = np.array([self.PDF(i) for i in x])
        plt.plot(x, y)
        # print(x)
        # print(y)

    #         plt.show()

    def calculate_moment(self, k):

        '''
        Doesn't work well yet
        '''

        # def integrand(self,x, power):
        #     return x ** power * self.PDF(x)
        return scipy.integrate.quad(lambda x: x ** k * self.PDF(x), -np.inf, np.inf,limit=100)[0]

    def calculate_central_moment(self, k):
        return scipy.integrate.quad(lambda x: (x - self.mean) ** k * self.PDF(x), -np.inf, np.inf,limit=100)[0]

    @staticmethod
    def param_requirements():
        print('hello')
        return ''

class Gaussian(Continuous_Distribution):
    def __init__(self, mu, sigma):
        #         super.__init__(self,min_x,max_x, interval)
        self.mean = mu
        self.mu = mu
        self.sigma = sigma
        self.std = sigma
        self.variance = sigma ** 2
        self.skewness = 0
        self.entropy = 0.5 * np.log(2 * np.pi * self.variance) + 0.5
        self.kurtosis = 3

    def __mul__(self, other):
        return Gaussian(other*self.mu,other*self.sigma)


    def __repr__(self):
        return f"N(mu={self.mu}, sigma={self.sigma})"

    def PDF(self, x):
        return 1 / (self.sigma * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - self.mean) / self.sigma) ** 2)

    def CDF(self, x):
        return scipy.norm.cdf(x)

    def MGF(self, t):
        return np.exp(self.mu * t + self.sigma**2 * t**2/2)
    @staticmethod
    def param_requirements():
        print('hello')
        return '''
        Param 1 (mu): float (-infty, infty)
        Param 2 (sigma): float (0, infty)
        '''

class Cauchy(Continuous_Distribution):
    def __init__(self, gamma, x_0):
        self.gamma = gamma
        self.x_0 = x_0

    def __repr__(self):
        return f"Cauchy Distribution: x_0={self.x_0}, gamma={self.gamma}"

    def PDF(self, x):
        return 1 / (np.pi * self.gamma * (1 + ((x - self.x_0) / self.gamma) ** 2))

    def CDF(self, x):
        return 1 / np.pi * np.arctan((x - self.x_0) / self.gamma)
    @staticmethod
    def param_requirements():
        print('hello')
        return '''
        Param 1 (Gamma): float (0, infty)
        Param 2 (x_0): float (-infty, infty)
        '''

class Exponential(Continuous_Distribution):
    def __init__(self, beta, *args):
        self.beta = beta
        self.mean = beta
        self.variance = beta ** 2

    def __repr__(self):
        return f'Exponential Distribution with Lambda = {1 / self.beta}'

    def PDF(self, x):
        if x >= 0:
            return 1 / self.beta * np.exp(-x / self.beta)
        else:
            return 0

    def Quantile(self, p):
        if 0 <= p < 1:
            return -1 * np.log(1 - p) * self.beta
        else:
            return "Choose p such that 0<= p < 1"

    def Sample(self):
        U = np.random.uniform()
        return self.Quantile(U)
    @staticmethod
    def param_requirements():
        return '''
        Param 1 (Beta): float (0, infty)
        '''

class Erlang(Continuous_Distribution):
    def __init__(self, k, param_lambda):
        if (not (type(k) == int or float(k).is_integer())) or k < 1:
            self.PDF = lambda x:0
        self.k = int(k)
        self.param_lambda = param_lambda

    def __repr__(self):
        return f"Erlang Distribution: k={self.k}, lambda={self.param_lambda}"

    def PDF(self, x):

        if x >= 0 :
            return (self.param_lambda ** self.k) * (x ** (self.k - 1)) * np.exp(
                -self.param_lambda * x) / math.factorial(self.k - 1)
        else:
            return 0

    @staticmethod
    def param_requirements():
        return '''
        Param 1 (k): int [1,infty)
        Param 2 (lambda): float (0, infty)
        '''
class Gamma(Continuous_Distribution):
    def __init__(self, alpha, beta):
        if (alpha <= 0):
            raise ('Alpha must be > 0')
        if (beta <= 0):
            raise ('Beta must be > 0')
        self.alpha = alpha
        self.beta = beta
        self.mean = alpha / beta
        self.variance = alpha / (beta ** 2)
        self.skewness = 2 / np.sqrt(alpha)
        self.support = '(0,inf)'

    def __repr__(self):
        return f"Gamma(alpha={self.alpha}, beta={self.beta})"

    def PDF(self, x):
        if x > 0:
            return (self.beta ** self.alpha) / scipy.special.gamma(self.alpha) * x ** (self.alpha - 1) * np.exp(
                -self.beta * x)
        else:
            return 0
    @staticmethod
    def param_requirements():
        return '''
        Param 1 (alpha): float (0, infty)
        Param 2 (beta): float (0, infty)
        '''

class Inverse_Gamma(Continuous_Distribution):
    def __init__(self, alpha, beta):
        if (alpha <= 0):
            raise ('Alpha must be > 0')
        if (beta <= 0):
            raise ('Beta must be > 0')
        self.alpha = alpha
        self.beta = beta
        if alpha > 0:
            self.mean = beta / (alpha - 1)
        else:
            self.mean = 'Undefined'
        if alpha > 2:
            self.variance = beta ** 2 / ((alpha - 1) ** 2 * (alpha - 2))
        else:
            self.variance = 'Undefined'

    def __repr__(self):
        return f'Inv-Gamma(alpha={self.alpha},beta={self.beta})'

    def PDF(self, x):
        if x > 0:
            return (self.beta ** self.alpha) / scipy.special.gamma(self.alpha) * x ** (-self.alpha - 1) * np.exp(
                -self.beta / x)
        else:
            return 0
    @staticmethod
    def param_requirements():
        return '''
        Param 1 (alpha): float (0, infty)
        Param 2 (beta): float (0, infty)
        '''

class Log_Normal(Continuous_Distribution):
    def __init__(self, mu, sigma):
        self.mu = mu
        self.mean = np.exp(mu + sigma ** 2 / 2)
        self.sigma = sigma
        self.median = np.exp(mu)

    def __repr__(self):
        return f"LogNorm(mu={self.mu}, sigma={self.sigma})"

    def PDF(self, x):
        if x > 0:
            return 1 / (x * self.sigma * np.sqrt(2 * np.pi)) * np.exp(
                -(np.log(x) - self.mu) ** 2 / (2 * self.sigma ** 2))
        else:
            return 0
    @staticmethod
    def param_requirements():
        return '''
        Param 1 (mu): float (-infty, infty)
        Param 2 (sigma): float (0, infty)
        '''

class Levy(Continuous_Distribution):
    def __init__(self, mu, c):
        self.mu = mu
        self.c = c

    def __repr__(self):
        return f'Levy(mu={self.mu},c={self.c})'

    def Sample(self):
        U = np.random.uniform()
        return self.c / (scipy.stats.norm.ppf(1 - U / 2)) ** 2 + self.mu

    def PDF(self, x, *args, **kwargs):
        return np.sqrt(self.c / (2 * np.pi)) * np.exp(-self.c / (2 * (x - self.mu))) / ((x - self.mu) ** (3 / 2))
    @staticmethod
    def param_requirements():
        return '''
        Param 1 (mu): float (-infnty, infty)
        Param 2 (c): float (0, infty)
        '''

class Laplace(Continuous_Distribution):
    def __init__(self, mu, b):
        self.mu = mu
        self.b = b
        self.mean = mu
        self.median = mu
        self.mode = mu
        self.variance = 2 * (b ** 2)
        self.skewness = 0

    def PDF(self, x):
        return 1 / (2 * self.b) * np.exp(-abs(x - self.mu) / self.b)

    def CDF(self, x):
        if x <= self.mu:
            return 0.5 * np.exp((x - self.mu) / self.b)
        else:
            return 1 - 0.5 * np.exp(-(x - self.mu) / self.b)

    def Quantile(self, p):
        return self.mu - self.b * np.sign(p - 0.5) * np.log(1 - 2 * abs(p - 0.5))
    @staticmethod
    def param_requirements():
        return '''
        Param 1 (mu): float (-infty, infty)
        Param 2 (b): float (0, infty)
        '''

class Pareto(Continuous_Distribution):

    def __init__(self, x_m, alpha):
        self.x_m = x_m
        self.alpha = alpha
        self.mean = np.where(alpha > 1, alpha*x_m/(alpha-1),np.inf)
    @staticmethod
    def param_requirements():
        return '''
        Param 1 (x_m): float (0, infty)
        Param 2 (alpha): float (0, infty)
        '''


class Beta_Binomial(Discrete_Distribution):
    def __init__(self, n, alpha, beta):
        #         super.__init__(self,min_x,max_x, interval)
        self.n = n
        self.alpha = alpha
        self.beta = beta

        self.mean = n * alpha / (alpha + beta)

        self.variance = (n * alpha * beta * (alpha + beta + n)) / ((alpha + beta) ** 2 * (alpha + beta + 1))
        self.support = np.array([i for i in range(0, n + 1)])

        # self.skewness = 0
        # self.entropy = 0.5 * np.log(2 * np.pi * self.variance) + 0.5
        # self.kurtosis = 3

    def PMF(self, k):
        if k >= 0 and (type(k) == int or float(k).is_integer()):
            return math.comb(self.n, k) * scipy.special.beta(k + self.alpha, self.n - k + self.beta) / scipy.special.beta(
            self.alpha, self.beta)
        else: return 0

    def plot(self, kind='scatter'):
        super().plot(self.support, kind=kind)


class Binomial(Discrete_Distribution):
    def __init__(self, n, p):
        #         super.__init__(self,min_x,max_x, interval)
        self.n = n
        self.p = p
        self.q = 1 - p

        self.mean = n * p

        self.variance = n * p * (1 - p)
        self.support = np.array([i for i in range(0, n + 1)])

        self.skewness = (self.q - self.p) / np.sqrt(n * self.p * self.q)
        # self.entropy = 0.5 * np.log(2 * np.pi * self.variance) + 0.5
        # self.kurtosis = 3

    def PMF(self, k):
        if (0<=k<=self.n) and (type(k) == int or float(k).is_integer()):
            return math.comb(self.n, k) * (self.p ** k) * (1 - self.p) ** (self.n - k)
        else:
            return 0

    def CDF(self, k):
        accum = 0
        for i in range(0, k + 1):
            accum += self.PDF(i)
        return accum

    def plot(self, kind='scatter'):
        super().plot(self.support, kind=kind)


class Hypergeometric(Discrete_Distribution):

    def __init__(self, N, K, n):
        self.N = N
        self.K = K
        self.n = n
        self.mean = self.n * self.K / self.N

    def PMF(self, k: int):
        if k >= 0 and (type(k) == int or float(k).is_integer()):
            return math.comb(self.K, k) * math.comb(self.N - self.K, self.n - k) / math.comb(self.N, self.n)
        else: return 0

class Geometric(Discrete_Distribution):

    def __init__(self, p):
        assert 0 < p < 1
        self.p = p
        self.mean = 1 / p
        self.variance = (1 / p) / (p ** 2)
        self.skewness = (2 - p) / np.sqrt(1 - p)

    def PMF(self, k: int):
        return (1 - self.p) ** (k - 1) * self.p

    def CDF(self, k: int):
        if k>=1 and (type(k) == int or float(k).is_integer()):
            return 1 - (1 - self.p) ** (k)
        else:
            return 0

class Negative_Geometric(Discrete_Distribution):

    def __init__(self, N, K, r):
        self.N = N
        self.K = K
        self.r = r
        self.mean = r*self.K/(self.N-self.K+1)


    def PMF(self, k):
        if k >= 0 and (type(k) == int or float(k).is_integer()):
            return math.comb(k+self.r-1,k)*math.comb(self.N-self.r-k,self.K-self.k)/math.comb(self.N,self.K)
        else: return 0

class Beta_Negative_Binomial(Discrete_Distribution):

    def __init__(self,alpha, beta, r):
        assert alpha >0
        assert beta >0
        assert r >0
        self.alpha = alpha
        self.beta = beta
        self.r = r
        self.mean = np.where(alpha>1, r*beta/(alpha-1),np.inf)

    def PMF(self,k):
        if k>=0 and (type(k) == int or float(k).is_integer()):
            k = int(k)
            return scipy.special.beta(self.r+k,self.alpha+self.beta)/scipy.special.beta(self.r,self.alpha)*scipy.special.gamma(k+self.beta)/(math.factorial(k)*scipy.special.gamma(self.beta))
        else:
            return 0

class Negative_Binomial(Discrete_Distribution):

    def __init__(self,r,p):
        self.r = r
        self.p = p
        self.mean = r*(1-p)/p
        self.variance = r*(1-p)/(p**2)
        self.skewness = (1+p)/np.sqrt(p*r)


    def PMF(self,k):
        if k>=0 and (type(k) == int or float(k).is_integer()):
            return math.comb(k+self.r-1,k)*(1-self.p)**k * self.p**self.r
        else: return 0

class Poisson(Discrete_Distribution):

    def __init__(self,param_lambda):
        assert param_lambda>0
        self.param_lambda = param_lambda
        self.mean = param_lambda
        self.variance = param_lambda
        self.skewness = 1/np.sqrt(param_lambda)

    def __add__(self, other):
        if type(other) == int:
            return (lambda x: self.PMF(x-other))
        else:
            return "Unsupported"

    def __radd__(self, other):
        return self.__add__(other)

    def PMF(self,k):
        if (type(k) == int or float(k).is_integer()):
            return np.exp(k*np.log(self.param_lambda)-self.param_lambda-np.log(scipy.special.gamma(k+1)))
        # return self.param_lambda**k * np.exp(-self.param_lambda)/math.factorial(k)
        else:
            return 0


    def Sample(self):
        # Knuth algorithm
        L = np.exp(-self.param_lambda)
        k = 0
        p = 1
        while (p>L):
            k=k+1
            U = np.random.uniform()
            p = p*U
        return k-1


class Convolution(Continuous_Distribution):
    '''
    Currently not very efficient
    '''

    def __init__(self, parent1, parent2):
        self.parent_1 = copy.deepcopy(parent1)
        self.parent_2 = copy.deepcopy(parent2)
        self.pdf_memo = dict()
        if type(parent1.mean) != str and type(parent2.mean) != str:
            self.mean = parent1.mean + parent2.mean

    def PDF(self, x,limit=100):
        if x in self.pdf_memo.keys():
            return self.pdf_memo[x]
        else:

            def integrand(tau, x):
                return self.parent_1.PDF(tau) * self.parent_2.PDF(x - tau)

            self.pdf_memo[x] = scipy.integrate.quad(integrand, -np.inf, np.inf, args=(x),limit=limit)[0]
            return self.pdf_memo[x]

class General_Discrete_Distribution():

    def __init__(self, PMF_1=(lambda x: 0), CDF=(lambda x:0),PMF_2=None):
        self.PMF_1 = PMF_1
        if PMF_2 is None:
            self.underlying_PMF = PMF_1
            self.PMF_2 = lambda x:0
            self.is_combination = False
        else:
            self.underlying_PMF = self.Discrete_Convolve
            self.is_combination = True
            self.PMF_2 = PMF_2
        self.underlying_CDF = CDF


        # self.support = support

        self.memo = dict()

    def Discrete_Convolve(self,k):
        accum = 0
        for i in range(0,k+1):
            accum += self.PMF_1(i)*self.PMF_2(k-i)
        return accum

    def __call__(self, *args, **kwargs):
        return self.PMF(*args,**kwargs)

    def PMF(self,k):
        if k in self.memo.keys():
            return self.memo[k]
        else:
            self.memo[k] = self.underlying_PMF(k)
            return self.memo[k]
            # if self.is_combination:
            #     self.memo[k] = self.underlying_PMF(k)
            #     return self.memo[k]
            # else:




    def __add__(self, other):
        if type(other) == int:
            return General_Discrete_Distribution(PMF_1=lambda x: self.PMF(x-other), CDF=lambda x: self.CDF(x-other))
        elif type(other) == General_Discrete_Distribution:
            new_dist = General_Discrete_Distribution(PMF_1 = self.underlying_PMF, PMF_2=other.underlying_PMF)
            return new_dist



    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if type(other) == int:
            return General_Discrete_Distribution(lambda x:self.PMF(x/other), lambda x: self.CDF(x/other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def power(self,exponent:int):
        if exponent % 2:
            return General_Discrete_Distribution(lambda x: 2*self.PMF(np.power(x,1/exponent)), lambda x: 2*self.CDF(np.power(x,1/exponent))-0.5)
        else:
            return General_Discrete_Distribution(lambda x: self.PMF(np.power(x, 1 / exponent)),
                                                 lambda x:  self.CDF(np.power(x, 1 / exponent)))

    def __pow__(self, power, modulo=None):
        assert type(power) == int
        if power % 2:
            return General_Discrete_Distribution(lambda x: 2 * self.PMF(np.power(x, 1 / power)),
                                                 lambda x: 2 * self.CDF(np.power(x, 1 / power)) - 0.5)
        else:
            return General_Discrete_Distribution(lambda x: self.PMF(np.power(x, 1 / power)),
                                                 lambda x: self.CDF(np.power(x, 1 / power)))

    def root(self,n):
        assert type(n) == int
        new_dist =  General_Discrete_Distribution(lambda x: self.PMF(np.power(x,n)),
                                                 lambda x: self.CDF(np.power(x, n)))
        return new_dist

    def __abs__(self):
        new_dist = General_Discrete_Distribution(lambda x: np.where(x!=0,self.PMF(x)+self.PMF(-x),self.PMF(0))[0]
                                                 )
        return new_dist

    '''
    Not sure how to handle cdf
    '''
    # def pointwise_multiplication(self,other):
    #     return General_Discrete_Distribution(lambda x: self.PMF(x)*other.PMF(x))


class General_Continuous_Distribution():

    def __init__(self, PDF, CDF=lambda x:0):
        self.PDF = PDF
        self.CDF = CDF
        self.mean = 0
        # self.support = support

    def __call__(self, *args, **kwargs):
        return self.PMF(*args,**kwargs)

    def __add__(self, other):
        if type(other) == int:
            new_dist = General_Continuous_Distribution(PDF=lambda x: self.PDF(x-other), CDF=lambda x: self.CDF(x-other))
            new_dist.mean = self.mean + other
            return new_dist

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if type(other) == int:
            new_dist =  General_Continuous_Distribution(lambda x:self.PDF(x/other), lambda x: self.CDF(x/other))
            new_dist = self.mean * other
            return new_dist

    def __rmul__(self, other):
        return self.__mul__(other)

    def __abs__(self):
        new_dist = General_Continuous_Distribution(lambda x: self.PDF(x)+self.PDF(-x),
                                                 )
        return new_dist




    # def exp(self):
    #     new_dist =


    '''
    Not sure how to handle cdf
    '''
    # def pointwise_multiplication(self,other):
    #     return General_Discrete_Distribution(lambda x: self.PMF(x)*other.PMF(x))

    def Quantile(self, p, min_x=0, stepsize=0.01):
        x = min_x  # change to min of support
        while (self.CDF(x) < p):
            x += stepsize
        return x

    def Sample(self):
        U = np.random.uniform()
        return self.Quantile(U)