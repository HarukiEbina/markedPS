import numpy as np

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.special import factorial2,legendre,binom,gamma
from scipy.integrate import simps
from velocileptors.Utils.loginterp import loginterp

from velocileptors.EPT.ept_fftw import EPT
from velocileptors.EPT.ept_fullresum_fftw import REPT

import math
from velocileptors.Utils.spherical_bessel_transform_fftw import SphericalBesselTransform
import os, json
from os.path import exists
import matplotlib.pyplot as plt

from numpy import polynomial as P

class mark:

    '''
    Class to compute IR-resummed RSD power spectrumusing the moment expansion appraoch in EPT.
    Instead of summing the velocity moments separately, the full 1-loop power spectrum with
    linear velocities resummed is computed.
    
    Based on the EPT class.
    
    Default output with tables is (k,mu) array with mu_deg = 0,2,4,6,8    
    '''
    
    def __init__(self, k, p, pnw=None, *args, rbao = 110, kmin = 1e-2, kmax = 0.5, nk = 100, sbao=None,Cn=None,\
                 name='toy',basedir = '.',R=15, **kw):
        
        self.nk, self.kmin, self.kmax = nk, kmin, kmax
        self.rbao = rbao
        
        #call usual power spectrum code (resummed)
        self.rept = REPT(k, p, pnw=pnw, rbao = rbao, kmin = kmin, kmax = kmax, nk = nk, sbao=sbao, **kw)
        self.ept = self.rept.ept
        
        if pnw is None:
            knw = self.ept.kint
            Nfilter =  np.ceil(np.log(7) /  np.log(knw[-1]/knw[-2])) // 2 * 2 + 1 # filter length ~ log span of one oscillation from k = 0.01
            pnw = savgol_filter(self.ept.pint, int(Nfilter), 4)
        else:
            knw, pnw = k, pnw
            
        # self.ept_nw = EPT( knw, pnw, kmin=kmin, kmax=kmax, nk = nk, third_order=True, **kw)
        self.ept_nw = self.rept.ept_nw
        
        self.beyond_gauss = self.ept.beyond_gauss
        
        self.kv = self.ept.kv
        self.plin  = loginterp(k, p)(self.kv)
        self.plin_nw = loginterp(knw, pnw)(self.kv)
        self.plin_w = self.plin - self.plin_nw
        if sbao is None:
            self.sigma_squared_bao = np.interp(self.rbao, self.ept_nw.qint, self.ept_nw.Xlin + self.ept_nw.Ylin/3.)
        else:
            self.sigma_squared_bao = sbao
            
        # FoG, sigma_squared_bao is sigma_FoG^2
        self.damp_exp = - 0.5 * self.kv**2 * self.sigma_squared_bao
        self.damp_fac = np.exp(self.damp_exp)
        
        self.pktable_nw = self.ept_nw.pktable_ept
        self.pktable_w  =  self.ept.pktable_ept - self.pktable_nw
        self.pktable_w[:,0] = self.kv
        self.pktable = self.pktable_nw + self.pktable_w; self.pktable[:,0] = self.kv
        
        self.vktable_nw = self.ept_nw.vktable_ept
        self.vktable_w  = self.ept.vktable_ept - self.vktable_nw
        self.vktable_w[:,0] = self.kv
        self.vktable = self.vktable_nw + self.vktable_w; self.vktable[:,0] = self.kv
        
        self.s0ktable_nw = self.ept_nw.s0ktable_ept
        self.s0ktable_w  =  self.ept.s0ktable_ept - self.s0ktable_nw
        self.s0ktable_w[:,0] = self.kv
        self.s0ktable = self.s0ktable_nw + self.s0ktable_w; self.s0ktable[:,0] = self.kv
        
        self.s2ktable_nw = self.ept_nw.s2ktable_ept
        self.s2ktable_w  = self.ept.s2ktable_ept - self.s2ktable_nw
        self.s2ktable_w[:,0] = self.kv
        self.s2ktable = self.s2ktable_nw + self.s2ktable_w; self.s2ktable[:,0] = self.kv
        
        if self.beyond_gauss:
            self.g1ktable_nw = self.ept_nw.g1ktable_ept
            self.g1ktable_w = self.ept.g1ktable_ept - self.ept_nw.g1ktable_ept
            self.g1ktable_w[:,0] = self.kv
            self.g1ktable = self.g1ktable_nw + self.g1ktable_w; self.g1ktable[:,0] = self.kv
        
            self.g3ktable_nw = self.ept_nw.g3ktable_ept
            self.g3ktable_w = self.ept.g3ktable_ept - self.ept_nw.g3ktable_ept
            self.g3ktable_w[:,0] = self.kv
            self.g3ktable = self.g3ktable_nw + self.g3ktable_w; self.g3ktable[:,0] = self.kv
        
            self.k0_nw, self.k2_nw, self.k4_nw = self.ept_nw.k0, self.ept_nw.k2, self.ept_nw.k4
            self.k0_w = self.ept.k0 - self.ept_nw.k0
            self.k2_w = self.ept.k2 - self.ept_nw.k2
            self.k4_w = self.ept.k4 - self.ept_nw.k4
            self.k0 = self.k0_nw + self.k0_w; self.k2 = self.k2_nw + self.k2_w; self.k4 = self.k4_nw + self.k4_w
        ########
        # up to here is original code from velocileptors
        ########
        self.basedir = basedir
        self.name = name
        
        self.Nskip = 40 # 50
        self.Cn = Cn
        if Cn is None: self.Cn = np.array([1,0,0,0])
        # self.xint = np.linspace(-1,1,500)
        self.xint = np.linspace(-1,1,250)
        self.kint = self.ept.kint #this is p
        self.qint = self.ept.qint #this is r
        self.mu_pow = np.arange(0,9,2)
        # self.mu_pow = np.arange(0,5,2)
        
        self.cutoff = 10
        self.plin_p  = loginterp(k, p)(self.kint) * np.exp(-(self.kint/self.cutoff)**2)
        self.plin_p_nw = loginterp(knw, pnw)(self.kint) * np.exp(-(self.kint/self.cutoff)**2)
        self.plin_p_w = self.plin_p - self.plin_p_nw
        

        self.kv = self.ept.kv
        self.plin  = loginterp(k, p)(self.kv)
        self.plin_nw = loginterp(knw, pnw)(self.kv)
        self.plin_w = self.plin - self.plin_nw
        
        self.damp_exp_p = - 0.5 * self.kint**2 * self.sigma_squared_bao
        self.damp_fac_p = np.exp(self.damp_exp)
        
        self.R = R
        
    def W_R(self,k,R=-1):
        # return 1
        if R<0: R = self.R
        # return np.exp(-(k**2*R**2)) #existing tables 4/27, newly made tables 4/30 (test5, real5)
        return np.exp(-(k**2*R**2)/2) #000-003, newly made table 4/27 (test4, real4)
        # Rcut = R
        # return np.exp(-(k**2*Rcut**2)/2)
    
    def Cd(self,k):
        C0, C1, C2, C3 = self.Cn
        return  -C1*self.W_R(k)+C0
    def Cd2(self,k1,k2):
        C0, C1, C2, C3 = self.Cn
        return C2*self.W_R(k1)*self.W_R(k2)-0.5*C1*(self.W_R(k1)+self.W_R(k2))
    def Cd3(self,k1,k2,k3):
        C0, C1, C2, C3 = self.Cn
        return -C3*self.W_R(k1)*self.W_R(k2)*self.W_R(k3)+C2*(self.W_R(k1)*self.W_R(k2)+self.W_R(k2)*self.W_R(k3)+self.W_R(k3)*self.W_R(k1))/3
    
    def CdCdP_at_mu(self,pars,f,mu_obs, apar=1., aperp=1.,bFoG=0):
        kv, pobs = self.rept.compute_redshift_space_power_at_mu(pars,f,mu_obs, apar=apar, aperp=aperp,bFoG=bFoG)
        return kv, pobs*self.Cd(kv)**2#, self.Cd(kv)**2
    
    def plin_IR_at_mu(self,pars,f,mu_obs,apar=1.,aperp=1.,bFoG=0):
        # don't need pars or f but placed there to ease compatibility with other at_mu functions
        plin_nw = self.plin_nw
        plin_w = self.plin_w
        F = apar/aperp
        AP_fac = np.sqrt(1 + mu_obs**2 *(1./F**2 - 1) )
        mu = mu_obs / F / AP_fac
        damp_exp = self.damp_exp * (1 + f*(2+f)*mu**2)
        damp_fac = np.exp(damp_exp)
        return plin_nw + damp_fac * plin_w
        
    def get_multipoles_from_at_mu(self,at_mu_func,pars, f, ngauss=4, apar=1., aperp=1.,bFoG=0):
        #consider changing ngauss for higher order contributions
        
        nus, ws = np.polynomial.legendre.leggauss(2*ngauss)
        nus_calc = nus[0:ngauss]
        
        L0 = np.polynomial.legendre.Legendre((1))(nus)
        L2 = np.polynomial.legendre.Legendre((0,0,1))(nus)
        L4 = np.polynomial.legendre.Legendre((0,0,0,0,1))(nus)
        
        self.pknutable = np.zeros((len(nus),self.nk))
        
        for ii, nu in enumerate(nus_calc):
            if at_mu_func==self.plin_IR_at_mu: 
                self.pknutable[ii,:] = at_mu_func(pars,f,nu,apar=apar,aperp=aperp,bFoG=bFoG)
            else: self.pknutable[ii,:] = at_mu_func(pars,f,nu,apar=apar,aperp=aperp,bFoG=bFoG)[1]
        
        self.pknutable[ngauss:,:] = np.flip(self.pknutable[0:ngauss],axis=0)
        
        self.p0k = 0.5 * np.sum((ws*L0)[:,None]*self.pknutable,axis=0)
        self.p2k = 2.5 * np.sum((ws*L2)[:,None]*self.pknutable,axis=0)
        self.p4k = 4.5 * np.sum((ws*L4)[:,None]*self.pknutable,axis=0)
        
        return self.kv, self.p0k, self.p2k, self.p4k
    
    ##### note: this doesn't include all contributions up to mu^6 in the IR P_L
    # this can be fixed easily if I edit the get_multipoles function, probably
    def plin_IR_multipoles(self,pars,f, ngauss=4, apar=1., aperp=1.,bFoG=0):
        return self.get_multipoles_from_at_mu(self.plin_IR_at_mu,pars,f,ngauss=ngauss,apar=apar,aperp=aperp,bFoG=bFoG)
    def plin_IR_table(self,pars,f,ngauss=4, apar=1., aperp=1.,bFoG=0,basis = 'Legendre'):
        ret = np.zeros((len(self.mu_pow),self.nk))
        kk, ret[0], ret[1], ret[2] = self.plin_IR_multipoles(pars,f, ngauss=ngauss, apar=apar, aperp=aperp,bFoG=bFoG)
        if basis=='Legendre':  return ret
        # else polynomial 
        return self.leg2poly(ret)
    def CdCdP_multipoles(self,pars, f, ngauss=4, apar=1., aperp=1.,bFoG=0):
        return self.get_multipoles_from_at_mu(self.CdCdP_at_mu,pars,f,ngauss=ngauss,apar=apar,aperp=aperp,bFoG=bFoG)
    def M13C_multipoles(self,pars, f, ngauss=4, apar=1., aperp=1.,bFoG=0):
        return self.get_multipoles_from_at_mu(self.M13C_at_mu,pars,f,ngauss=ngauss,apar=apar,aperp=aperp,bFoG=bFoG)
    def M22C_multipoles(self,pars, f, ngauss=4, apar=1., aperp=1.,bFoG=0):
        return self.get_multipoles_from_at_mu(self.M22C_at_mu,pars,f,ngauss=ngauss,apar=apar,aperp=aperp,bFoG=bFoG)
    def M13B_multipoles(self,pars, f, ngauss=4, apar=1., aperp=1.,bFoG=0):
        return self.get_multipoles_from_at_mu(self.M13B_at_mu,pars,f,ngauss=ngauss,apar=apar,aperp=aperp,bFoG=bFoG)
    def M22B_multipoles(self,pars, f, ngauss=4, apar=1., aperp=1.,bFoG=0):
        return self.get_multipoles_from_at_mu(self.M22B_at_mu,pars,f,ngauss=ngauss,apar=apar,aperp=aperp,bFoG=bFoG)
    
    def compute_tables(self,pars,f,overwrite = False):
        #no need for A terms as tables are already in self.rept
        self.plin_IR_leg = self.plin_IR_table(None,f,basis='Legendre')
        self.plin_IR_poly = self.plin_IR_table(None,f,basis='Polynomial')
        return 
    
    def P_at_mu(self,pars,f,mu_obs, apar=1., aperp=1.,bFoG=0):
        kv, pobs = self.rept.compute_redshift_space_power_at_mu(pars,f,mu_obs, apar=apar, aperp=aperp,bFoG=bFoG)
        return kv, pobs
    def WP_at_mu(self,pars,f,mu_obs, apar=1., aperp=1.,bFoG=0):
        kv, pobs = self.rept.compute_redshift_space_power_at_mu(pars,f,mu_obs, apar=apar, aperp=aperp,bFoG=bFoG)
        return kv, pobs*self.W_R(kv)
    
    def W2P_at_mu(self,pars,f,mu_obs, apar=1., aperp=1.,bFoG=0):
        kv, pobs = self.rept.compute_redshift_space_power_at_mu(pars,f,mu_obs, apar=apar, aperp=aperp,bFoG=bFoG)
        return kv, pobs*self.W_R(kv)**2
    
    def compute_CdCdP_table(self,pars,f):
        table = self.leg2poly(np.array([self.get_multipoles_from_at_mu(self.P_at_mu,pars,f)[i] for i in range(1,4)]+[np.zeros(self.kv.shape) for i in range(len(self.mu_pow)-3)]))
        Wtable = self.leg2poly(np.array([self.get_multipoles_from_at_mu(self.WP_at_mu,pars,f)[i] for i in range(1,4)]+[np.zeros(self.kv.shape) for i in range(len(self.mu_pow)-3)]))
        W2table = self.leg2poly(np.array([self.get_multipoles_from_at_mu(self.W2P_at_mu,pars,f)[i] for i in range(1,4)]+[np.zeros(self.kv.shape) for i in range(len(self.mu_pow)-3)]))
        
        # C_0^2, C_0 C_1, C_0 C_2, C_0 C_3, C_1^2, C_1 C_2, C_1 C_3, C_2^2, C_2 C_3
        final_table = np.zeros((9,len(self.mu_pow),self.nk))
        
        #C0^2
        final_table[0] += table
        #C0C1
        final_table[1] += 2*Wtable
        #C1^2
        final_table[4] += W2table
        return final_table

    
    def compute_M13C_table(self,pars,f):
        kv = self.kv       
        mu_pow = self.mu_pow
        plin = self.plin
        plin_p = self.plin_p
        plin_IR_leg = self.plin_IR_leg
        plin_IR_poly = self.plin_IR_poly
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = pars
        kint = self.kint
        xint = self.xint
        Cn = self.Cn #expansion coefficients
        C0, C1, C2, C3 = self.Cn
        W_R_int = self.W_R(kint)
        W_R = self.W_R(kv)
        Cd = self.Cd(kv)
        
        sigma2_R = simps(kint**2*W_R_int*(plin_p+sn),x=kint)/(2*np.pi**2)
        S_R = (b1**2+2*b1*f/3+f**2/5)*sigma2_R
        sigma2_RR = simps(kint**2*W_R_int**2*(plin_p+sn),x=kint)/(2*np.pi**2)
        S_RR = (b1**2+2*b1*f/3+f**2/5)*sigma2_RR
        prefacs = np.array([b1**2,2*b1*f,f**2,0,0])
            
        # C_0^2, C_0 C_1, C_0 C_2, C_0 C_3, C_1^2, C_1 C_2, C_1 C_3, C_2^2, C_2 C_3
        M13C_table = np.zeros((9,len(mu_pow),self.nk))
        # C_0 C_2 
        M13C_table[2] += np.einsum('j,ij->ij',2 *W_R*S_R  + S_RR,plin_IR_poly)
        M13C_table[2,0] += (2 *W_R*sigma2_R+ sigma2_RR)*sn
        # C_1 C_2 
        M13C_table[5] += np.einsum('j,ij->ij', W_R * (2 *W_R*S_R  + S_RR) ,plin_IR_poly)
        M13C_table[5,0] += W_R *(2 *W_R*sigma2_R+ sigma2_RR)*sn
        # C_0 C_3
        M13C_table[3] += np.einsum('j,ij->ij',3 *W_R*S_RR ,plin_IR_poly)
        M13C_table[3,0] += 3 *W_R*sigma2_RR*sn
        # C_1 C_3
        M13C_table[6] += np.einsum('j,ij->ij', W_R * W_R* S_RR ,plin_IR_poly)
        M13C_table[6,0] += W_R*W_R*sigma2_RR*sn
        
        final_table = np.zeros(M13C_table.shape)
        for i in range(len(final_table)):
            for j in range(len(mu_pow)):
                final_table[i,j:] += prefacs[j]*M13C_table[i,:len(mu_pow)-j]
            
        return final_table
    
    def compute_M22C_table(self,pars,f):
        kv = self.kv
        plin = self.plin
        plin_p = self.plin_p
        plin_IR_leg = self.plin_IR_leg
        plin_IR_poly = self.plin_IR_poly
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = pars
        kint = self.kint
        Cn = self.Cn #expansion coefficients
        C0, C1, C2, C3 = self.Cn
        W_R_int = self.W_R(kint)
        W_R = self.W_R(kv)
        mu_pow = self.mu_pow
        P11_k = np.zeros((len(mu_pow),len(self.kint)))
        
        # this includes sn term of linear power spectrum
        P11_k[0]=plin_p*(b1**2+2/3*b1*f+f**2/5) + sn
        P11_k[1]=plin_p*(4/3*b1**2+4/7*f)
        P11_k[2]=plin_p*7/35*f**2
        
        # C_0^2, C_0 C_1, C_0 C_2, C_0 C_3, C_1^2, C_1 C_2, C_1 C_3, C_2^2, C_2 C_3
        M22C_table = np.zeros((9,len(mu_pow),self.nk))
        # C_1 C_1
        M22C_table[4] += 0.5 * self.leg2poly( self.conv(W_R_int*P11_k,W_R_int*P11_k) + self.conv(W_R_int**2*P11_k,P11_k) )
        # C_1 C_2 
        M22C_table[5] += 2 * self.leg2poly( self.conv(W_R_int**2*P11_k,W_R_int*P11_k) )
        # C_2 C_2
        M22C_table[7] += 2 * self.leg2poly( self.conv(W_R_int**2*P11_k,W_R_int**2*P11_k) )
        
        M22C_table *= 2
        
        return M22C_table
    
    def get_zn(self,n,pars,f,ks,mu_pows,ps,xs,stoch = False):
        # n is true n
        # consider just saving these
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = pars
        res = np.zeros(ks.shape)
        denom = (ks**2+ps**2-2*ks*ps*xs)
        
        abb = (ks/ps+ps/ks)*xs
        if not stoch:
            if n==0:
                res[:,0]+= (b1/14*(7*b2+b1*(10-7*abb+4*xs**2)))[:,0]
                res[:,2]+= (b1*f/14*(7*b1+ks**2*(6-7*abb+8*xs**2)/denom))[:,0]
            elif n==1:
                res[:,1]+= (b1*f/14*(-7*b1*(ks/ps+ps/ks)+2*(7*ks**2*xs+7*ps**2*xs-2*ks*ps*(3+4*xs**2))/denom))[:,0]
                res[:,3]+= (-b1*f**2*ks/(2*ps))[:,0]
            elif n==2:
                res[:,0]+= (f/2*(b1**2+b2+2*b1*(5/7-1/2*abb+2*xs**2/7)+b1*ps**2*(6-7*abb+8*xs**2)/(7*denom)))[:,0]
                res[:,2]+= (f**2/14*(21*b1+ks**2*(6-7*abb+8*xs**2)/denom))[:,0]
            elif n==3:
                res[:,1]+= (f**2/14*(b1*(-7*ks/ps-14*ps/ks)+2*(7*ks**2*xs+7*ps**2*xs-2*ks*ps*(3+4*xs**2))/denom))[:,0]
                res[:,3]+= (-f**3*ks/(2*ps))[:,0]
            elif n==4:
                res[:,0]+= (f**2/14*(7*b1+ps**2*(6-7*abb+8*xs**2)/denom))[:,0]
                res[:,2]+= f**3
            elif n==5:
                res[:,1]+= (-f**3*ps/(2*ks))[:,0]
        else:
            c0 = alpha0 / (2*b1)
            c1 = (alpha2/2 - c0 *f)/b1 

            if n==0:
                res[:,0]+= ( (b1+ c0 * ps**2) /14*(7*b2+b1*(10-7*abb+4*xs**2)))[:,0]
                res[:,2]+= ((b1+ c0 * ps**2)*f/14*(7*b1+ks**2*(6-7*abb+8*xs**2)/denom))[:,0]
            elif n==1:
                res[:,1]+= ((b1+ c0 * ps**2)*f/14*(-7*b1*(ks/ps+ps/ks)+2*(7*ks**2*xs+7*ps**2*xs-2*ks*ps*(3+4*xs**2))/denom))[:,0]
                res[:,3]+= (-(b1+ c0 * ps**2)*f**2*ks/(2*ps))[:,0]
            elif n==2:
                res[:,0]+= 1/14*( (f+c1*ps**2) * (7*b2 + b1 * (10 -7*abb + 4 * xs**2 )) + f*(b1+c0*ps**2) * (7*b1 + ps**2  * (6 - 7*abb+8*xs**2)/ denom) )[:,0]
                res[:,2]+= (f**2 * (b1+c0*ps**2) + 1/14*f*(f+c1*ps**2)* (7*b1 + ks**2  * (6 - 7*abb+8*xs**2)/ denom))[:,0]
            elif n==3:
                res[:,1]+= (-f**2*ps*(b1+c0*ps**2)/(2*ks) + (f+c1*ps**2)*(-b1*f*(ks**2 + ps**2)/(2*ks*ps) + f*(7*ks**2*xs+7*ps**2*xs-2*ks*ps*(3+4*xs**2))/(7*denom) ))[:,0]
                res[:,3]+= (-f**2*ks * (f+c1*ps**2)/(2*ps))[:,0]
            elif n==4:
                res[:,0]+= (f*(f+c1*ps**2)/14*(7*b1+ps**2*(6-7*abb+8*xs**2)/denom))[:,0]
                res[:,2]+= f**2*(f+c1*ps**2)[:,0]
            elif n==5:
                res[:,1]+= (-f**2*(f+c1*ps**2)*ps/(2*ks))[:,0]
        return res
    
    def get_zn_22(self,n,pars,f,ks,mu_pows,ps,xs,stoch=False):
        # n is true n
        # consider just saving these
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = pars
        res = np.zeros(ks.shape)
        denom = (ks**2+ps**2-2*ks*ps*xs)
        if not stoch:
            if n==0:
                # Z_1(k-p) Z_1(p) Z_2(p,k-p)
                res[:,0]+= (b1**2/14*(7*b2+b1*ks**2* (7*ks*xs/ps + 3-10*xs**2 )/denom) )[:,0]
                res[:,2]+= (b1*f/14/ps*ks**2* (7*b1**2*ps*denom + 7*b2*ps*denom + b1* (14*ks**3*xs + ks**2*ps*(2-30*xs**2)+3*ks*ps**2*xs*(3+4*xs**2) - ps**3*(1+6*xs**2) ) ) /denom**2)[:,0]
                res[:,4]+= (b1*f**2*ks**4/14*(7*ks*xs/ps - 1 +7*b1-6*xs**2 )/denom**2 )[:,0]

            elif n==1:
                res[:,1]+= (b1*f*ks/14/ps *( -14*b2*ps**2*denom + 7*b1**2*ks*(ks-2*ps*xs)*denom -2*b1*ks**2*ps*(7*ks*xs+ps*(3-10*xs**2) )    )/denom**2  )[:,0]
                res[:,3]+= (b1*f**2*ks**3 /14/ps * (-7*b1*(-2*ks**2+ps**2+4*ks*ps*xs )+2*ps*(ps-7*ks*xs+6*ps*xs**2) )/denom**2   )[:,0]
                res[:,5]+= (b1*f**3*ks**5/(2*ps*denom**2) )[:,0]
            elif n==2:
                res[:,0]+= (b1*f/14/ps*(ks**2+2*ps**2-2*ks*ps*xs) *(7*b2*ps*denom + b1*ks**2*(7*ks*xs +ps*(3-10*xs**2) ) )/denom**2 )[:,0]
                res[:,2]+= (f**2* ks**2 /14/ps * (7*b2*ps*denom + 7*b1**2*ps*(-2*ks**2+ps**2+4*ks*ps*xs) +2*b1*(7*ks**3*xs + 2*ks*ps**2*xs*(4+3*xs**2) - ps**3*(1+6*xs**2) + ks**2 *(ps-15*ps*xs**2)  )   )/denom**2  )[:,0]
                res[:,4]+= (f**3*ks**4/14/ps*(ps+14*b1*ps-7*ks*xs+6*ps*xs**2)/denom**2 )[:,0]
            elif n==3:
                res[:,1]+= (f**2*ks/14/ps * (-14*b2*ps**2*denom + 7*b1**2*ks*(ks-2*ps*xs)*(ks**2+2*ps**2-2*ks*ps*xs) - 2*b1*ks**2*ps*(7*ks*xs + ps*(3-10*xs**2 ) ) )/denom**2 )[:,0]
                res[:,3]+= (f**3*ks**3/7/ps * (7*b1*denom + ps*(ps-7*ks*xs+6*ps*xs**2) )/denom**2 )[:,0]
                res[:,5]+= (f**4 * ks**5/ (2*ps * denom**2) )[:,0]
            elif n==4:
                res[:,0]+= (f**2*ps/14 * (7*b2*ps*denom+b1*ks**2*(7*ks*xs+ps*(3-10*xs**2)) )/denom**2 )[:,0]
                res[:,2]+= (-f**3*ks**2/14*(ps*(ps-7*ks*xs+6*ps*xs**2) + 7*b1*(ps**2 + 3*ks*(ks-2*ps*xs)) )/denom**2   )[:,0]
                res[:,4]+= (-3*f**4*ks**4/(2*denom**2) )[:,0]
            elif n==5:
                res[:,1]+= (b1*f**3*ks**2*ps*(ks-2*ps*xs)/(2*denom**2) )[:,0]
                res[:,3]+= (3*f**4*ks**3*ps/(2*denom**2))[:,0]
            elif n==6:
                res[:,0]+= (-f**4*ks**2*ps**2/(2*denom**2))[:,0]
        else:
            c0 = alpha0 / (2*b1) # stoch
            c1 = (alpha2/2 - c0 *f)/b1
            # c0 = 0 # stoch2
            b1_p = (b1+c0*ps**2)
            b1_d = (b1+c0*denom)
            f_p = (f+c1*ps**2)
            f_d = (f+c1*denom)
            if n==0:
                # Z_1(k-p) Z_1(p) Z_2(p,k-p)
                res[:,0]+= (b1_p*b1_d /14*(7*b2+b1*ks**2* (7*ks*xs/ps + 3-10*xs**2 )/denom) )[:,0]
                res[:,2]+= (b1_p /14/ps*ks**2*(f*b1_d*(7*ks*xs+ps*(-1+7*b1-6*xs**2)) + f_d*(7*b2*ps*denom+b1*ks**2*(7*ks*xs+ps*(3-10*xs**2)))/denom    )/denom)[:,0]
                res[:,4]+= (b1_p*f*f_d*ks**4/14*(7*ks*xs/ps - 1 +7*b1-6*xs**2 )/denom**2 )[:,0]
            elif n==1:
                res[:,1]+= (b1_p*ks/14*(7*b1*f*ks*(ks-2*ps*xs)*denom*b1_d/ps - 2*f_d*(7*b2*ps*denom+b1*ks**2*(7*ks*xs+ps*(3-10*xs**2)) )    )/denom**2)[:,0]
                res[:,3]+= (b1_p*f*ks**3 /14/ps*(7*b1*f_d*(ks**2-2*ps**2-2*ks*ps*xs)+7*b1_d*f*denom+2*f_d*ps*(ps-7*ks*xs+6*ps*xs**2) )/denom**2    )[:,0]
                res[:,5]+= (b1_p*f**2*f_d*ks**5/(2*ps*denom**2) )[:,0]
            elif n==2:
                res[:,0]+= ((b1_p*f_d*ps**2 + b1_d*f_p*denom)/14/ps*(7*b2*ps*denom + b1*ks**2*(7*ks*xs +ps*(3-10*xs**2) ) )/denom**2 )[:,0]
                res[:,2]+= (ks**2 /14/ps *(-b1_d*f*denom*(7*b1_p*f*ps+f_p*(ps-7*b1*ps-7*ks*xs+6*ps*xs**2) )+f_d*(7*b2*f_p*ps*denom+7*b1*b1_p*f*ps*(-2*ks**2+ps**2+4*ks*ps*xs)-b1_p*f*ps**2*(ps-7*ks*xs+6*ps*xs**2)+b1*f_p*ks**2*(7*ks*xs+ps*(3-10*xs**2)) )      )/denom**2    )[:,0]
                res[:,4]+= (-f*f_d*ks**4/14/ps*(21*b1_p*f*ps+f_p*(ps-7*b1*ps-7*ks*xs+6*ps*xs**2  )  )/denom**2 )[:,0]
            elif n==3:
                res[:,1]+= (ks*(-14*b2*f_d*f_p*ps**2*denom+b1*ks*(7*b1_d*f*f_p*(ks**3-4*ks**2*ps*xs-2*ps**3*xs+ks*ps**2*(1+4*xs**2))+f_d*ps*(7*b1_d*f*ps*(ks-ps*xs)-2*f_p*ks*(3*ps+7*ks*xs-10*ps*xs**2) ) ) )/(14*ps*denom**2)  )[:,0]
                res[:,3]+= (f*ks**3/14/ps * (7*b1*f_d*f_p*denom + 7*b1_d*f*f_p*denom+f_d*ps*(21*b1_p*f*ps+f*f_p*(ps-7*ks*xs+6*ps*xs**2)) )/denom**2 )[:,0]
                res[:,5]+= (f**2*f_d*f_p * ks**5/ (2*ps * denom**2) )[:,0]
            elif n==4:
                res[:,0]+= (f_d*f_p*ps/14 * (7*b2*ps*denom+b1*ks**2*(7*ks*xs+ps*(3-10*xs**2)) )/denom**2 )[:,0]
                res[:,2]+= (-f*ks**2/14*(7*b1*f_d*f_p*(2*ks**2-ps**2-4*ks*ps*xs) +7*b1_d*f*f_p*denom+f_d*ps*(7*b1_p*f*ps+f_p*(ps-7*ks*xs+6*ps*xs**2)) )/denom**2   )[:,0]
                res[:,4]+= (-3*f**2*f_d*f_p*ks**4/(2*denom**2) )[:,0]
            elif n==5:
                res[:,1]+= (b1*f*f_d*f_p*ks**2*ps*(ks-2*ps*xs)/(2*denom**2) )[:,0]
                res[:,3]+= (3*f**2*f_d*f_p*ks**3*ps/(2*denom**2))[:,0]
            elif n==6:
                res[:,0]+= (-f**2*f_d*f_p*ks**2*ps**2/(2*denom**2))[:,0]
        res[np.abs(denom)<1e-5] = 0
        return res
            
    def get_Gnm(self,n,m,x=None):
        if x is None: x = self.xint
        res = np.zeros(x.shape)
        n,m = round(n),round(m)
        for l in range(0,n+1):
            res+= (1+(-1)**(l+n))*(2*l+1)*binom(l,m)*binom((l+m-1)/2,l)*2**(2*l)*gamma(n+1)\
                *gamma((n+l)/2+2)\
                *legendre(l)(x)/(gamma((n-l)/2+1)*gamma(n+l+3))
        return res
    
    def compute_integral13_table(self,pars,f,unity=False,stoch=False):
        # int_x W_R(k-p) Z_1(p) Z_2(k,-p) P_L(p) if unity is False
        # if unity is True int_x Z_1(p) Z_2(k,-p) P_L(p)
        # return array of mu_pow, k, p
        # print('called')
        x = self.xint
        p = self.kint
        kv = self.kv+0
        mu_pow = self.mu_pow
        plin_p = self.plin_p
        plin_IR_leg = self.plin_IR_leg
        plin_IR_poly = self.plin_IR_poly
        
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = pars
        Nskip = self.Nskip
        
        final_array = np.zeros((len(mu_pow),self.nk,len(p))) #powers of mu

        for ii in range((len(kv)-1)//Nskip+1):
            print(ii*Nskip,(ii+1)*Nskip)
            k = kv[ii*Nskip:(ii+1)*Nskip]
            ks, mu_pows, ps, xs = np.meshgrid(k,np.arange(np.max(mu_pow)+1), p,x,indexing='ij')
            #ks = k
            
            if unity: S_int = 1
            else: S_int = self.W_R(np.sqrt(ks**2+ps**2-2*ks*ps*xs))
            for n in range(np.max(mu_pow)+1):
                zn = self.get_zn(n,pars,f,ks,mu_pows,ps,xs,stoch=stoch) 
                if np.max(np.abs(zn))<1e-15: 
                    # print(n,'zn=0')
                    del zn
                    continue
                for m in range(n+1):
                    ### sum over even and odd here, since only the end product is constrained by symmetry
                    Gnm = self.get_Gnm(n,m)
                    if np.max(np.abs(Gnm))<1e-15: 
                        # print(n,m,'Gnm=0')
                        del Gnm
                        continue
                    integral_px = np.einsum('ijk,k->ijk',simps(x=xs, y=np.einsum('ijkl,l->ijkl',0.5*S_int*zn,Gnm),axis=3),plin_p) #complete integral of x then multiply w/ P
                    for i in range(np.max(mu_pow)+1):
                        if m+i>np.max(mu_pow) or (m+i)%2!=0: continue
                        final_array[round((m+i)/2),ii*Nskip:(ii+1)*Nskip,:]+=integral_px[:,i,:]
                    del integral_px
                del zn
            del ks; del mu_pows; del ps; del xs; del S_int
            
        return final_array
        
    def compute_M13B_table(self,pars,f,stoch=False):
        x = self.xint
        p = self.kint
        kv = self.kv+0
        mu_pow = self.mu_pow
        plin_p = self.plin_p
        plin_IR_leg = self.plin_IR_leg
        plin_IR_poly = self.plin_IR_poly
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = pars
        
        name=self.name
        name = name[name.find('/') + 1:]
        name = name[name.find('/') + 1:]
        name = name[4:]
        # load tables
        if stoch:    
            stoch_str = '_stoch'
            c0 = alpha0 / (2*b1)
            c1 = (alpha2/2 - c0 *f)/b1
        else: 
            stoch_str = ''
            c0 = 0
            c1 = 0
        with open(self.basedir+'/output/rsd/z1.100/%s/integral13%s.json'%(name,stoch_str)) as json_file:
            data = json.load(json_file)
            self.integral13 = np.array(data['table'])
        with open(self.basedir+'/output/rsd/z1.100/%s/integral13_unity%s.json'%(name,stoch_str)) as json_file:
            data = json.load(json_file)
            self.integral13_unity = np.array(data['table'])

        # self.integral13 = self.compute_integral13_table(pars,f)
        # self.integral13_unity = self.compute_integral13_table(pars,f,unity=True)
        integral13 = self.integral13
        integral13_unity = self.integral13_unity
        
        mu_pows,ks,  ps = np.meshgrid(mu_pow, kv,p,indexing='ij')
        W_R_p = self.W_R(ps)
        
        # S=W_R(k-p)
        subintegral13_1 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*integral13/(2*np.pi**2),axis=2)
        # S=W_R(p)
        subintegral13_2 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*integral13_unity*W_R_p/(2*np.pi**2),axis=2) 
        # S=W_R(k-p) W_R(p)
        subintegral13_3 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*integral13*W_R_p/(2*np.pi**2),axis=2) 
        
        
        W_R = self.W_R(kv)
        # C_0^2, C_0 C_1, C_0 C_2, C_0 C_3, C_1^2, C_1 C_2, C_1 C_3, C_2^2, C_2 C_3
        M13B_table = np.zeros((9,len(mu_pow),self.nk))
        # C_0 C_1
        C0C1_contr = [np.polymul((subintegral13_1[:,i]+subintegral13_2[:,i]),plin_IR_poly[:,i])[:len(mu_pow)] for i in range(len(kv))]
        C0C1_contr = np.array(C0C1_contr).T
        M13B_table[1] += (b1 + c0*kv**2 )*C0C1_contr
        for i in range(len(mu_pow)-1): 
            M13B_table[1,i+1]+=C0C1_contr[i]*(f+c1*kv**2)
        M13B_table[1] *= 2
        
        # C_1^2 
        C1C1_contr = [np.polymul((subintegral13_1[:,i]+subintegral13_2[:,i]),plin_IR_poly[:,i])[:len(mu_pow)] for i in range(len(kv))]
        C1C1_contr = np.array(C1C1_contr).T
        C1C1_contr = np.einsum('j,ij->ij',W_R,C1C1_contr)
        M13B_table[4] += (b1 + c0*kv**2 )*C1C1_contr
        for i in range(len(mu_pow)-1): 
            M13B_table[4,i+1]+=C1C1_contr[i]*(f+c1*kv**2)
        M13B_table[4] *= 2

        # C_0 C_2 
        C0C2_contr = [np.polymul(subintegral13_3[:,i],plin_IR_poly[:,i])[:len(mu_pow)] for i in range(len(kv))]
        C0C2_contr = np.array(C0C2_contr).T
        M13B_table[2] += (b1 + c0*kv**2 )*C0C2_contr
        for i in range(len(mu_pow)-1): 
            M13B_table[2,i+1]+=C0C2_contr[i]*(f+c1*kv**2)
        M13B_table[2] *= 4

        # C_1 C_2 
        C1C2_contr = [np.polymul(subintegral13_3[:,i],plin_IR_poly[:,i])[:len(mu_pow)] for i in range(len(kv))]
        C1C2_contr = np.array(C1C2_contr).T
        C1C2_contr = np.einsum('j,ij->ij',W_R,C1C2_contr)
        M13B_table[5] += (b1 + c0*kv**2 )*C1C2_contr
        for i in range(len(mu_pow)-1): 
            M13B_table[5,i+1]+=C1C2_contr[i]*(f+c1*kv**2)
        M13B_table[5] *= 4
        
        return M13B_table
    
    def compute_integral22_table(self,pars,f,Y,stoch=False):
        kv=self.kv
        p = self.kint
        x = self.xint
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = pars
        Cn = self.Cn #expansion coefficients
        C0, C1, C2, C3 = self.Cn
        # Y_func = interp1d(p,Y,fill_value='extrapolate')
        Y_func = interp1d([0]+p.tolist(),[0]+Y.tolist(),fill_value='extrapolate')
        
        mu_pow = self.mu_pow
        Nskip = self.Nskip
        # Nskip = 50
        final_array = np.zeros((len(mu_pow),self.nk,len(p))) #powers of mu
        # res=np.zeros((len(mu_pow),self.nk)) #powers of mu
        for ii in range((len(kv)-1)//Nskip+1):
            os.system('echo ii %d'%ii)
            k = kv[ii*Nskip:(ii+1)*Nskip]
            ks, mu_pows, ps, xs = np.meshgrid(k,np.arange(np.max(mu_pow)+1), p,x,indexing='ij')
            Yint = Y_func(np.sqrt(ks**2+ps**2-2*ks*ps*xs))
            for n in range(np.max(mu_pow)+1):
                zn = self.get_zn_22(n,pars,f,ks,mu_pows,ps,xs,stoch=stoch) 
                if np.max(np.abs(zn))<1e-15: 
                    # print(n,'zn=0')
                    del zn
                    continue
                for m in range(n+1):
                    Gnm = self.get_Gnm(n,m)
                    if np.max(np.abs(Gnm))<1e-15: 
                        del Gnm
                        continue
                    integrand = np.einsum('ijkl,l,ijkl->ijkl',zn,Gnm,Yint)
                    integral_px = simps(x=xs, y=0.5*integrand,axis=3)#zn*Gnm*Yint*Xint
                    # integral_px = simps(x=ps[:,:,:,0],y=ps[:,:,:,0]**2*integral_px/(2*np.pi**2),axis=2)
                    for i in range(np.max(mu_pow)+1):
                        if m+i>np.max(mu_pow) or (m+i)%2!=0: continue
                        final_array[round((m+i)/2),ii*Nskip:(ii+1)*Nskip,:]+=integral_px[:,i,:]
                    del integral_px
                del zn
            del ks; del mu_pows; del ps; del xs; del Yint
        return final_array
    
    def compute_M22B_table(self,pars,f,stoch=False):
        kv = self.kv
        p=self.kint
        mu_pow = self.mu_pow
        W_R = self.W_R(kv)
        W_R_int = self.W_R(self.kint)
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = pars
        c0 = alpha0/(2*b1)
        c1 = (alpha2-f*alpha0/b1)/(2*b1)
        Cn = self.Cn #expansion coefficients
        C0, C1, C2, C3 = self.Cn
        plin_p= self.plin_p
        plin_p_nw = self.plin_p_nw
        plin_p_w = self.plin_p_w
        plin_IR_leg = self.plin_IR_leg
        plin_IR_poly = self.plin_IR_poly

        name=self.name
        name = name[name.find('/') + 1:]
        name = name[name.find('/') + 1:]
        name = name[4:]
        if stoch:    stoch_str = '_stoch'
        else: stoch_str = ''
        with open(self.basedir+'/output/rsd/z1.100/%s/integral22_W%s.json'%(name,stoch_str)) as json_file:
            data = json.load(json_file)
            self.integral22_W = np.array(data['table'])
        # Y = W_R(k-p) P_L(k-p)
        integral22_W = self.integral22_W
        mu_pows, ks, ps = np.meshgrid(mu_pow,kv, p,indexing='ij')
        # S= P_L(p)        W_R(k-p) P_L(k-p) 
        subintegral22_2 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*integral22_W * plin_p /(2*np.pi**2),axis=2)
        # S= W_R(p) P_L(p) W_R(k-p) P_L(k-p) 
        subintegral22_3 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*integral22_W * W_R_int * plin_p /(2*np.pi**2),axis=2)
        M22B_table = np.zeros((9,len(mu_pow),self.nk))
        # C_0^2, C_0 C_1, C_0 C_2, C_0 C_3, C_1^2, C_1 C_2, C_1 C_3, C_2^2, C_2 C_3
        # C_0 C_1
        C0C1_contr = 2 * (2* subintegral22_2)
        M22B_table[1] += C0C1_contr
        
        # C_1 C_1
        C1C1_contr = 2 * np.einsum('j,ij->ij',W_R,(2* subintegral22_2)) 
        M22B_table[4] += C1C1_contr

        # C_0 C_2
        C0C2_contr = 4 * subintegral22_3
        M22B_table[2] += C0C2_contr
        
        # C_1 C_2
        C1C2_contr = 4 * np.einsum('j,ij->ij',W_R,(subintegral22_3))
        M22B_table[5] += C1C2_contr

        return M22B_table    

    def compute_stoch_dof_table(self,pars,f,stoch=False):
        # bispectrum stochastic piece proportional to new dof Bshot
        kv = self.kv
        p=self.kint
        mu_pow = self.mu_pow
        W_R = self.W_R(kv)
        W_R_int = self.W_R(self.kint)
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = pars
        if stoch:
            c0 = alpha0/(2*b1)
            c1 = (alpha2-f*alpha0/b1)/(2*b1)
        else:
            c0 = 0
            c1 = 0
        Cn = self.Cn #expansion coefficients
        C0, C1, C2, C3 = self.Cn
        plin_p= self.plin_p
        plin_p_nw = self.plin_p_nw
        plin_p_w = self.plin_p_w
        plin_IR_leg = self.plin_IR_leg
        plin_IR_poly = self.plin_IR_poly
        R = self.R
        mu_pows, ks, ps = np.meshgrid(mu_pow,kv, p,indexing='ij')
        
        var_int = ks*ps*R**2
        
        int_st0 = np.zeros((len(mu_pow),self.nk,len(self.kint)))
        int_st2 = np.zeros((len(mu_pow),self.nk,len(self.kint)))
        int_st4 = np.zeros((len(mu_pow),self.nk,len(self.kint)))

        # hyperbolic functions with cutoff
        # sinh(kpR^2) W_R(p) W_R(k)
        # np.exp(-(self.kint/self.cutoff)**2)
        sinh_cut = 0.5*(np.exp(var_int-ps**2*R**2/2-ks**2*R**2/2)-np.exp(-var_int-ps**2*R**2/2-ks**2*R**2/2))
        cosh_cut = 0.5*(np.exp(var_int-ps**2*R**2/2-ks**2*R**2/2)+np.exp(-var_int-ps**2*R**2/2-ks**2*R**2/2))
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n dx/2 for int_stn
        # 2pi factor absorbed (later) by 1/2pi^3
        int_st0[0] += (sinh_cut/var_int)[0]
        
        int_st2[0] += ((var_int*cosh_cut - sinh_cut)/var_int**3)[0]
        int_st2[1] += ((-3*var_int*cosh_cut + (3+var_int**2)*sinh_cut  )/var_int**3)[0]

        int_st4[0] += ((-9*var_int*cosh_cut + 3*(3+var_int**2)*sinh_cut   )/var_int**5)[0]
        int_st4[1] += (6* (var_int*(15+var_int**2)*cosh_cut -3*(5+2*var_int**2)*sinh_cut  )/var_int**5)[0]
        int_st4[2] += ((-5*var_int*(21+2*var_int**2)*cosh_cut + (105+45*var_int**2+var_int**4)*sinh_cut )/var_int**5)[0]
        
        plin_ps = interp1d(self.kint,self.plin_p)(ps)
        
        # W_R(p) W_R(k) included in int_st 
        # int_x e^kpxR^2 G_nm(x) mu_p^n P_L(p) W_R(p) dp/2pi**2 dx/2 for int_stn
        int_P_0 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st0 )*plin_ps /(2*np.pi**2),axis=2)
        int_P_2 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st2 )*plin_ps /(2*np.pi**2),axis=2)
        int_P_4 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st4 )*plin_ps /(2*np.pi**2),axis=2)

        # int_x e^kpxR^2 G_nm(x) mu_p^n P_L(p) W_R^2(p) dp/2pi**2 dx/2 for int_stn
        int_WP_0 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st0 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        int_WP_2 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st2 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        int_WP_4 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st4 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n P_L(p) W_R(p) dp/2pi**2 dx/2 for int_stn
        int_Pp2_0 = simps(x=ps[:,:,:],y=ps[:,:,:]**4*(int_st0 )*plin_ps /(2*np.pi**2),axis=2)
        int_Pp2_2 = simps(x=ps[:,:,:],y=ps[:,:,:]**4*(int_st2 )*plin_ps /(2*np.pi**2),axis=2)
        int_Pp2_4 = simps(x=ps[:,:,:],y=ps[:,:,:]**4*(int_st4 )*plin_ps /(2*np.pi**2),axis=2)
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n P_L(p) W_R(p) dp/2pi**2 dx/2 for int_stn
        int_WPp2_0 = simps(x=ps[:,:,:],y=ps[:,:,:]**4*(int_st0 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        int_WPp2_2 = simps(x=ps[:,:,:],y=ps[:,:,:]**4*(int_st2 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        int_WPp2_4 = simps(x=ps[:,:,:],y=ps[:,:,:]**4*(int_st4 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n W_R^2(p) dp/2pi**2 dx/2 for int_stn
        int_W_0 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st0 )*W_R_int /(2*np.pi**2),axis=2)
        int_W_2 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st2 )*W_R_int/(2*np.pi**2),axis=2)
        int_W_4 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st4 )*W_R_int /(2*np.pi**2),axis=2)
        
        int_1_0 = np.zeros((len(mu_pow),self.nk))
        # W_R(p) W_R(k) inclusion for int_st doesn't apply here
        # int_x e^kpxR^2 G_nm(x) mu_p^n W_R(p) dp/2pi**2 dx/2 for int_stn
        int_1_0[0] = simps(x=ps[:,:,:],y=ps[:,:,:]**2 *W_R_int /(2*np.pi**2),axis=2)[0]
                
                
        #C0 C1
        temp_A = [np.polymul((int_1_0[:,i]),plin_IR_poly[:,i])[:len(mu_pow)] for i in range(len(kv))]
        temp_A = np.array(temp_A).T
        C0C1_contr_A = b1*2*temp_A
        C0C1_contr = b1*C0C1_contr_A + c0*kv**2*C0C1_contr_A
        C0C1_contr[1:] = f*C0C1_contr_A[:-1] + c1*kv**2*C0C1_contr_A[:-1]
        temp_B = b1*int_P_0+c0*int_Pp2_0+f*int_P_2+c1*int_Pp2_2
        C0C1_contr_B = b1*(1+W_R)*temp_B
        C0C1_contr += C0C1_contr_B
        
        #C1 C1
        C1C1_contr = C0C1_contr*W_R
        
        #C0 C2
        temp_A = [np.polymul((int_W_0[:,i]),plin_IR_poly[:,i])[:len(mu_pow)] for i in range(len(kv))]
        temp_A = np.array(temp_A).T
        C0C2_contr_A = b1*2*temp_A
        C0C2_contr = b1*C0C2_contr_A + c0*kv**2*C0C2_contr_A
        C0C2_contr[1:] = f*C0C2_contr_A[:-1] + c1*kv**2*C0C2_contr_A[:-1]
        temp_B = b1*int_WP_0+c0*int_WPp2_0+f*int_WP_2+c1*int_WPp2_2
        C0C2_contr_B = b1*2*temp_B
        C0C2_contr += C0C2_contr_B

        #C1 C2
        C1C2_contr = C0C2_contr*W_R

        final_array = np.zeros((9,len(mu_pow),self.nk))
        # C_0^2, C_0 C_1, C_0 C_2, C_0 C_3, C_1^2, C_1 C_2, C_1 C_3, C_2^2, C_2 C_3
        final_array[1]+=C0C1_contr
        final_array[4]+=C1C1_contr
        final_array[2]+=C0C2_contr
        final_array[5]+=C1C2_contr
                
        final_array *= 4 # contribution same between M22 and M13, so final is M22+2 M13 = 4 M13
        
        return final_array
    
    def compute_stoch_BSN_table(self,pars,f):
        # bispectrum stochastic piece not proportional to new dof
        kv = self.kv
        p=self.kint
        mu_pow = self.mu_pow
        W_R = self.W_R(kv)
        W_R_int = self.W_R(self.kint)
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = pars
        # c0 = alpha0/(2*b1)
        # c1 = (alpha2-f*alpha0/b1)/(2*b1)
        Cn = self.Cn #expansion coefficients
        C0, C1, C2, C3 = self.Cn
        plin_p= self.plin_p
        plin_p_nw = self.plin_p_nw
        plin_p_w = self.plin_p_w
        plin_IR_leg = self.plin_IR_leg
        plin_IR_poly = self.plin_IR_poly
        R = self.R
        mu_pows, ks, ps = np.meshgrid(mu_pow,kv, p,indexing='ij')
        var_int = ks*ps*R**2

        int_st0 = np.zeros((len(mu_pow),self.nk,len(self.kint)))
        int_st2 = np.zeros((len(mu_pow),self.nk,len(self.kint)))
        int_st4 = np.zeros((len(mu_pow),self.nk,len(self.kint)))
        
        # hyperbolic functions with smoothing
        # sinh(kpR^2) W_R(p) W_R(k)
        sinh_cut = 0.5*(np.exp(var_int-ps**2*R**2/2-ks**2*R**2/2)-np.exp(-var_int-ps**2*R**2/2-ks**2*R**2/2))
        cosh_cut = 0.5*(np.exp(var_int-ps**2*R**2/2-ks**2*R**2/2)+np.exp(-var_int-ps**2*R**2/2-ks**2*R**2/2))
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n dx/2 for int_stn
        # 2pi factor absorbed (later) by 1/2pi^3
        int_st0[0] += (sinh_cut/var_int)[0]
        
        int_st2[0] += ((var_int*cosh_cut - sinh_cut)/var_int**3)[0]
        int_st2[1] += ((-3*var_int*cosh_cut + (3+var_int**2)*sinh_cut  )/var_int**3)[0]

        int_st4[0] += ((-9*var_int*cosh_cut + 3*(3+var_int**2)*sinh_cut   )/var_int**5)[0]
        int_st4[1] += (6* (var_int*(15+var_int**2)*cosh_cut -3*(5+2*var_int**2)*sinh_cut  )/var_int**5)[0]
        int_st4[2] += ((-5*var_int*(21+2*var_int**2)*cosh_cut + (105+45*var_int**2+var_int**4)*sinh_cut )/var_int**5)[0]
        
        plin_ps = interp1d(self.kint,self.plin_p)(ps)
        
        # W_R(p) W_R(k) included in int_st 
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n W_R^2(p) dp/2pi**2 dx/2 for int_stn
        int_W_0 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st0 )*W_R_int /(2*np.pi**2),axis=2)
        int_W_2 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st2 )*W_R_int/(2*np.pi**2),axis=2)
        int_W_4 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st4 )*W_R_int /(2*np.pi**2),axis=2)
        
        int_1_0 = np.zeros((len(mu_pow),self.nk))
        # W_R(p) W_R(k) inclusion for int_st doesn't apply here
        # int_x e^kpxR^2 G_nm(x) mu_p^n W_R(p) dp/2pi**2 dx/2 for int_stn
        int_1_0[0] = simps(x=ps[:,:,:],y=ps[:,:,:]**2 *W_R_int /(2*np.pi**2),axis=2)[0]
        
        #C0 C1
        # C0C1_contr_C = sn**2*2*int_1_0
        C0C1_contr_C = int_1_0/np.max(np.abs(int_1_0)) #make it easier to run cobaya
        C0C1_contr = C0C1_contr_C

        #C1 C1
        C1C1_contr = C0C1_contr*W_R
        
        #C0 C2        
        # C0C2_contr_C = sn**2*2*int_W_0
        C0C2_contr_C = int_W_0/np.max(np.abs(int_1_0)) #make it easier to run cobaya
        C0C2_contr = C0C2_contr_C

        #C1 C2
        C1C2_contr = C0C2_contr*W_R

        final_array = np.zeros((9,len(mu_pow),self.nk))
        # C_0^2, C_0 C_1, C_0 C_2, C_0 C_3, C_1^2, C_1 C_2, C_1 C_3, C_2^2, C_2 C_3
        final_array[1]+=C0C1_contr
        final_array[4]+=C1C1_contr
        final_array[2]+=C0C2_contr
        final_array[5]+=C1C2_contr
        
        final_array *= 4 # contribution same between M22 and M13, so final is M22+2 M13 = 4 M13
        
        return final_array

    def compute_stoch_SN_table(self,pars,f,stoch = False):
        # bispectrum stochastic piece not proportional to new dof (Bshot), but to SN
        kv = self.kv
        p=self.kint
        mu_pow = self.mu_pow
        W_R = self.W_R(kv)
        W_R_int = self.W_R(self.kint)
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4 = pars
        if stoch:
            c0 = alpha0/(2*b1)
            c1 = (alpha2-f*alpha0/b1)/(2*b1)
        else: 
            c0=0; c1=0
        Cn = self.Cn #expansion coefficients
        C0, C1, C2, C3 = self.Cn
        plin_p= self.plin_p
        plin_p_nw = self.plin_p_nw
        plin_p_w = self.plin_p_w
        plin_IR_leg = self.plin_IR_leg
        plin_IR_poly = self.plin_IR_poly
        R = self.R
        mu_pows, ks, ps = np.meshgrid(mu_pow,kv, p,indexing='ij')
        var_int = ks*ps*R**2

        int_st0 = np.zeros((len(mu_pow),self.nk,len(self.kint)))
        int_st2 = np.zeros((len(mu_pow),self.nk,len(self.kint)))
        int_st4 = np.zeros((len(mu_pow),self.nk,len(self.kint)))
        
        # hyperbolic functions with cutoff
        # sinh(kpR^2) W_R(p) W_R(k)
        # np.exp(-(self.kint/self.cutoff)**2)
        sinh_cut = 0.5*(np.exp(var_int-ps**2*R**2/2-ks**2*R**2/2)-np.exp(-var_int-ps**2*R**2/2-ks**2*R**2/2))
        cosh_cut = 0.5*(np.exp(var_int-ps**2*R**2/2-ks**2*R**2/2)+np.exp(-var_int-ps**2*R**2/2-ks**2*R**2/2))
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n dx/2 for int_stn
        # 2pi factor absorbed (later) by 1/2pi^3
        int_st0[0] += (sinh_cut/var_int)[0]
        
        int_st2[0] += ((var_int*cosh_cut - sinh_cut)/var_int**3)[0]
        int_st2[1] += ((-3*var_int*cosh_cut + (3+var_int**2)*sinh_cut  )/var_int**3)[0]

        int_st4[0] += ((-9*var_int*cosh_cut + 3*(3+var_int**2)*sinh_cut   )/var_int**5)[0]
        int_st4[1] += (6* (var_int*(15+var_int**2)*cosh_cut -3*(5+2*var_int**2)*sinh_cut  )/var_int**5)[0]
        int_st4[2] += ((-5*var_int*(21+2*var_int**2)*cosh_cut + (105+45*var_int**2+var_int**4)*sinh_cut )/var_int**5)[0]
        
        plin_ps = interp1d(self.kint,self.plin_p)(ps)
        plin_ps = self.plin_p
        
        # W_R(p) W_R(k) included in int_st 
        # int_x e^kpxR^2 G_nm(x) mu_p^n P_L(p) W_R(p) dp/2pi**2 dx/2 for int_stn
        int_P_0 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st0 )*plin_ps /(2*np.pi**2),axis=2)
        int_P_2 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st2 )*plin_ps /(2*np.pi**2),axis=2)
        int_P_4 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st4 )*plin_ps /(2*np.pi**2),axis=2)

        # int_x e^kpxR^2 G_nm(x) mu_p^n P_L(p) W_R^2(p) dp/2pi**2 dx/2 for int_stn
        int_WP_0 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st0 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        int_WP_2 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st2 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        int_WP_4 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st4 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n P_L(p) W_R(p) dp/2pi**2 dx/2 for int_stn
        int_Pp2_0 = simps(x=ps[:,:,:],y=ps[:,:,:]**4*(int_st0 )*plin_ps /(2*np.pi**2),axis=2)
        int_Pp2_2 = simps(x=ps[:,:,:],y=ps[:,:,:]**4*(int_st2 )*plin_ps /(2*np.pi**2),axis=2)
        int_Pp2_4 = simps(x=ps[:,:,:],y=ps[:,:,:]**4*(int_st4 )*plin_ps /(2*np.pi**2),axis=2)
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n P_L(p) W_R(p) dp/2pi**2 dx/2 for int_stn
        int_WPp2_0 = simps(x=ps[:,:,:],y=ps[:,:,:]**4*(int_st0 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        int_WPp2_2 = simps(x=ps[:,:,:],y=ps[:,:,:]**4*(int_st2 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        int_WPp2_4 = simps(x=ps[:,:,:],y=ps[:,:,:]**4*(int_st4 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n W_R^2(p) dp/2pi**2 dx/2 for int_stn
        int_W_0 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st0 )*W_R_int /(2*np.pi**2),axis=2)
        int_W_2 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st2 )*W_R_int/(2*np.pi**2),axis=2)
        int_W_4 = simps(x=ps[:,:,:],y=ps[:,:,:]**2*(int_st4 )*W_R_int /(2*np.pi**2),axis=2)
        
        int_1_0 = np.zeros((len(mu_pow),self.nk))
        # W_R(p) W_R(k) inclusion for int_st doesn't apply here
        # int_x e^kpxR^2 G_nm(x) mu_p^n W_R(p) dp/2pi**2 dx/2 for int_stn
        int_1_0[0] = simps(x=ps[:,:,:],y=ps[:,:,:]**2 *W_R_int /(2*np.pi**2),axis=2)[0]
        #C0 C1
        C0C1_contr_A = np.zeros(int_1_0.shape)
        temp_A = [np.polymul((int_1_0[:,i]),plin_IR_poly[:,i])[:len(mu_pow)] for i in range(len(kv))]
        temp_A = np.array(temp_A).T
        C0C1_contr_A[1:] = (2*sn*f*2*temp_A)[:-1]
        C0C1_contr = b1*C0C1_contr_A + c0*kv**2*C0C1_contr_A
        C0C1_contr[1:] = f*C0C1_contr_A[:-1] + c1*kv**2*C0C1_contr_A[:-1]
        temp_B = (b1*int_P_2+c0*int_Pp2_2+f*int_P_4+c1*int_Pp2_4)
        C0C1_contr_B = (2*sn*f*(1+W_R)*temp_B)
        C0C1_contr += C0C1_contr_B

        #C1 C1
        C1C1_contr = C0C1_contr*W_R
        
        #C0 C2
        C0C2_contr_A = np.zeros(int_1_0.shape)
        temp_A = [np.polymul((int_W_0[:,i]),plin_IR_poly[:,i])[:len(mu_pow)] for i in range(len(kv))]
        temp_A = np.array(temp_A).T
        C0C2_contr_A[1:] = (2*sn*f*2*temp_A)[:-1]
        C0C2_contr = b1*C0C2_contr_A + c0*kv**2*C0C2_contr_A
        C0C2_contr[1:] = f*C0C2_contr_A[:-1] + c1*kv**2*C0C2_contr_A[:-1]
        temp_B = b1*int_WP_2+c0*int_WPp2_2+f*int_WP_4+c1*int_WPp2_4
        C0C2_contr_B = (2*sn*f*2*temp_B)
        C0C2_contr += C0C2_contr_B

        #C1 C2
        C1C2_contr = C0C2_contr*W_R

        final_array = np.zeros((9,len(mu_pow),self.nk))
        # C_0^2, C_0 C_1, C_0 C_2, C_0 C_3, C_1^2, C_1 C_2, C_1 C_3, C_2^2, C_2 C_3
        final_array[1]+=C0C1_contr
        final_array[4]+=C1C1_contr
        final_array[2]+=C0C2_contr
        final_array[5]+=C1C2_contr
        
        final_array *= 4 # contribution same between M22 and M13, so final is M22+2 M13 = 4 M13
        
        return final_array
        
    def conv(self,X,Y):
        ell = self.mu_pow 
        ell_max = np.max(ell) 
        
        sph_kr = SphericalBesselTransform(self.kint, L=ell_max+1,ncol=2,fourier=True)
        sph_rk = SphericalBesselTransform(self.qint, L=ell_max*2+1,ncol=1)
        #assume legendre basis for X, Y
        kv=self.kv
        X_ell_k = X+0
        Y_ell_k = Y+0
        X_ell_r = np.zeros((len(X_ell_k),len(self.qint)))
        Y_ell_r = np.zeros((len(X_ell_k),len(self.qint)))
        XY_L_k = np.zeros((2*len(X_ell_k),len(self.qint)))
        XY_L_r = np.zeros((2*len(X_ell_k),len(self.qint)))
        for i in ell: #only enum over even
            rtemps, XYtemps = sph_kr.sph(i,[X_ell_k[round(i/2)],Y_ell_k[round(i/2)]])
            XYtemps*=(-1)**(round(i/2))#/(2*np.pi**2)
            X_ell_r[round(i/2)] = interp1d(rtemps, XYtemps[0],fill_value='extrapolate')(self.qint)
            Y_ell_r[round(i/2)] = interp1d(rtemps, XYtemps[1],fill_value='extrapolate')(self.qint)
        
        for L in ell:
            for i in ell:
                for j in ell:
                    XY_L_r[round(L/2)] += (2*L+1)*X_ell_r[round(i/2)]*Y_ell_r[round(j/2)]*self.my_wigner3j(i,j,L)**2
            ktemps, XY_L_k[round(L/2)] = sph_rk.sph(L,XY_L_r[round(L/2)])
            
            XY_L_k[round(L/2)]*=(-1)**(round(L/2))*4*np.pi
        res = np.zeros((len(X_ell_k),len(self.kv)))
        for i in range(len(ell)):
            res[i] = interp1d(ktemps,XY_L_k[i],fill_value='extrapolate')(self.kv)
        return res
            
    def poly2leg(self,inarr):
        # inarr = (k,mu_pow), mu_pows are only even
        res = np.zeros(inarr.shape)

        for i in range(len(inarr[0])):
            coeffs = []
            for j in range(len(self.mu_pow)):
                coeffs.append(inarr[j][i])
                if j!=len(inarr)-1: coeffs.append(0)
            # print(coeffs)
            new_coeffs = np.polynomial.legendre.poly2leg(coeffs)#.coefs
            new_coeffs = new_coeffs.tolist()
            while len(new_coeffs)<len(coeffs): new_coeffs.append(0)
            new_coeffs = np.array(new_coeffs)
            idxs = [2*j for j in range(len(self.mu_pow))]
            # print(coeffs,new_coeffs,idxs)
            res[:,i] = new_coeffs[idxs]
        return res   
        
    def leg2poly(self,inarr):
        res = np.zeros(inarr.shape)        
        for i in range(len(inarr[0])):
            coeffs = []
            # for j in range(len(inarr)):
            for j in range(len(self.mu_pow)):
                coeffs.append(inarr[j][i])
                if j!=len(inarr)-1: coeffs.append(0)
            new_coeffs = np.polynomial.legendre.leg2poly(coeffs)
            new_coeffs = new_coeffs.tolist()
            while len(new_coeffs)<len(coeffs): new_coeffs.append(0)
            new_coeffs = np.array(new_coeffs)
            idxs = [2*j for j in range(len(self.mu_pow))]
            res[:,i] = new_coeffs[idxs]
        return res   
            
    def my_wigner3j(self,j1,j2,j):
        #assume m1, m2, m=0
        j1,j2,j = round(j1),round(j2),round(j)
        if j>j1+j2 or j<np.abs(j1-j2): return 0
        J = j1+j2+j
        if J%2==1: return 0
        g = round(J/2)
        return (-1)**g*np.sqrt(math.factorial(2*g-2*j1)*math.factorial(2*g-2*j2)*math.factorial(2*g-2*j)/math.factorial(2*g+1))*\
            math.factorial(g)/(math.factorial(g-j1)*math.factorial(g-j2)*math.factorial(g-j))
    
    
