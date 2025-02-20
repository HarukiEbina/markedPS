import numpy as np

from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from scipy.special import factorial2,legendre,binom,gamma
from scipy.integrate import simpson as simps
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
    Class to compute the 1-loop marked power spectrum in EPT approach
    
    Based on the velocileptors EPT (https://github.com/sfschen/velocileptors/tree/master/velocileptors/EPT)
    
    Default output with tables is (mu,k) array with mu_deg = 0,2,4

    FFTLog is not implemented for M22B and M13B (the "three-point" terms). This is the current limiting factor in computation time
    
    stoch functionality is deprecated
    '''
    
    def __init__(self, k, p, pnw=None, *args, rbao = 110, kmin = 1e-2, kmax = 0.5, nk = 100, sbao=None,Cn=None,\
                 name='toy',basedir = '.',R=15, **kw):
        
        self.nk, self.kmin, self.kmax = nk, kmin, kmax
        self.rbao = rbao
        
        # call usual power spectrum code (resummed)
        self.rept = REPT(k, p, pnw=pnw, rbao = rbao, kmin = kmin, kmax = kmax, nk = nk, sbao=sbao, **kw)
        self.ept = self.rept.ept
        
        if pnw is None:
            knw = self.ept.kint
            Nfilter =  np.ceil(np.log(7) /  np.log(knw[-1]/knw[-2])) // 2 * 2 + 1 # filter length ~ log span of one oscillation from k = 0.01
            pnw = savgol_filter(self.ept.pint, int(Nfilter), 4)
        else:
            knw, pnw = k, pnw
            
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
            
        self.damp_exp = - 0.5 * self.kv**2 * self.sigma_squared_bao
        self.damp_fac = np.exp(self.damp_exp)
        
        self.basedir = basedir
        self.name = name
        
        self.Nskip = self.nk
        self.Cn = Cn
        if Cn is None: self.Cn = np.array([1,0,0,0])
        self.xint = np.linspace(-1,1,250)
        self.kint = self.ept.kint #this is p
        self.qint = self.ept.qint #this is r
        self.mu_pow = np.arange(0,5,2)
        
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
        if R<0: R = self.R
        return np.exp(-(k**2*R**2)/2) 
    
    def Cd(self,k,Cn=None):
        if Cn is None: C0, C1, C2, C3 = self.Cn
        else: C0, C1, C2, C3 = Cn
        return  C1*self.W_R(k)+C0
    def Cd2(self,k1,k2,Cn=None):
        if Cn is None: C0, C1, C2, C3 = self.Cn
        else: C0, C1, C2, C3 = Cn
        return C2*self.W_R(k1)*self.W_R(k2)+0.5*C1*(self.W_R(k1)+self.W_R(k2))
    def Cd3(self,k1,k2,k3,Cn=None):
        if Cn is None: C0, C1, C2, C3 = self.Cn
        else: C0, C1, C2, C3 = Cn
        return C3*self.W_R(k1)*self.W_R(k2)*self.W_R(k3)+C2*(self.W_R(k1)*self.W_R(k2)+self.W_R(k2)*self.W_R(k3)+self.W_R(k3)*self.W_R(k1))/3
        
    def plin_IR_at_mu(self,pars,f,mu_obs,apar=1.,aperp=1.,bFoG=0):
        # pars placed there to ease compatibility with other at_mu functions
        plin_nw = self.plin_nw
        plin_w = self.plin_w
        F = apar/aperp
        AP_fac = np.sqrt(1 + mu_obs**2 *(1./F**2 - 1) )
        mu = mu_obs / F / AP_fac
        damp_exp = self.damp_exp * (1 + f*(2+f)*mu**2)
        damp_fac = np.exp(damp_exp)
        return plin_nw + damp_fac * plin_w
        
    def get_multipoles_from_at_mu(self,at_mu_func,pars, f, ngauss=4, apar=1., aperp=1.,bFoG=0):        
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
    
    def plin_IR_multipoles(self,f, ngauss=4, apar=1., aperp=1.,bFoG=0):
        return self.get_multipoles_from_at_mu(self.plin_IR_at_mu,None,f,ngauss=ngauss,apar=apar,aperp=aperp,bFoG=bFoG)
    def plin_IR_table(self,f,ngauss=4, apar=1., aperp=1.,bFoG=0,basis = 'Legendre'):
        ret = np.zeros((len(self.mu_pow),self.nk))
        kk, ret[0], ret[1], ret[2] = self.plin_IR_multipoles(f, ngauss=ngauss, apar=apar, aperp=aperp,bFoG=bFoG)
        if basis=='Legendre':  return ret
        # else polynomial 
        return self.leg2poly(ret)
    
    def compute_tables(self,f,write=False):
        # compute IR-resumed plin in legendre and polynomial bases
        # in mu-k grid
        self.plin_IR_leg = self.plin_IR_table(f,basis='Legendre')
        self.plin_IR_poly = self.plin_IR_table(f,basis='Polynomial')

        # compute other tables in bias-mark-mu-k grid
        self.M13B_table = self.compute_M13B_table(f,write=write)
        self.M13C_table = self.compute_M13C_table(f)
        self.M22B_table = self.compute_M22B_table(f,write=write)
        self.M22C_table = self.compute_M22C_table(f)

        self.SN_table = self.compute_stoch_SN_table(f) # bispectrum terms proportional to sn
        self.B0_table = self.compute_stoch_BSN_table(f) # bispectrum terms proportional to b0
        self.Bshot_table = self.compute_stoch_dof_table(f) # bispectrum terms proportional to bshot
        
        return 

    def get_bias_vec(self,pars):
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4,bshot, b0, nm0, nm2, db, df = pars
        bias_monomial = [1, b1, b1**2, b2, b1*b2, b2**2, bs, b1*bs, b2*bs, bs**2, b3, b1*b3, b1**3, b1**4, b1**2* b2, b1**2* bs]
        bias_monomial += [sn, sn2, sn4, alpha0, alpha2, alpha4]
        bias_monomial += [b1 *sn, b1**2* sn, b1**3* sn, b1**4 *sn, sn**2, b1* sn**2, b1**2* sn**2, b1**3* sn**2, b1**4* sn**2 ]
        bias_monomial += [ b1* bshot, b1**2 * bshot]#, b0, nm0, nm2]
        bias_monomial = np.array(bias_monomial)
        return bias_monomial
    
    def combine_biases(self,pars,table):
        bias_monomial = self.get_bias_vec(pars)
        table_len = len(table)
        try: return np.einsum('m,mijk->ijk',bias_monomial[:table_len],table)
        except: return np.einsum('m,mjk->jk',bias_monomial[:table_len],table)        
    
    def get_Cn_vec(self,Cn,Cn2=None):
        C0, C1, C2, C3 = Cn
        if Cn2 is None: C0_2, C1_2, C2_2, C3_2 = Cn
        else: C0_2, C1_2, C2_2, C3_2 = Cn2
        return np.array([C0*C0_2, (C0*C1_2+C0_2*C1)/2, (C0* C2_2 +C0_2* C2 )/2, (C0* C3_2 +C0_2* C3 )/2,
                C1*C1_2, (C1*C2_2+C1_2*C2)/2, (C1*C3_2+C1_2*C3)/2, C2*C2_2, (C2*C3_2+C2_2*C3)/2])
        # return [C0**2, C0* C1, C0 *C2, C0 *C3, C1**2, C1 *C2, C1* C3, C2**2, C2*C3]
        
    def combine_mark_params(self,Cn,table,change=False,Cn2=None):
        if change: 
            Cn_new = [Cn[i]*(-1)**i for i in range(len(Cn))] 
        else: Cn_new = Cn    
        if Cn2 is not None: 
            if change: Cn2_new = [Cn2[i]*(-1)**i for i in range(len(Cn2))]
            else: Cn2_new = Cn2
        else: Cn2_new = None
        Cn_vec = self.get_Cn_vec(Cn_new,Cn2=Cn2_new)
        try: return np.einsum('i,ijk->jk',Cn_vec,table)
        except: return np.einsum('i,mijk->mjk',Cn_vec,table)

    def compute_table_power(self,pars,table,Cn,Cn2=None):
        tmp = self.combine_biases(pars,table)
        return self.combine_mark_params(Cn,tmp,Cn2=Cn2)
    
    def compute_power(self,pars,f,Cn=None,Cn2=None,basis='Legendre'):
        # return the total redshift space power in polynomial basis
        if Cn is None: Cn = self.Cn
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4,bshot, b0, nm0, nm2,db, df = pars
        # higher point terms 
        M13B = self.compute_table_power(pars,self.M13B_table,Cn,Cn2=Cn2)
        M13C = self.compute_table_power(pars,self.M13C_table,Cn,Cn2=Cn2)
        M22B = self.compute_table_power(pars,self.M22B_table,Cn,Cn2=Cn2)
        M22C = self.compute_table_power(pars,self.M22C_table,Cn,Cn2=Cn2)
        
        SN = self.compute_table_power(pars,self.SN_table,Cn,Cn2=Cn2)
        # BSN = self.compute_table_power(pars,f,self.B0_table,Cn,Cn2=Cn2)
        BSN = self.combine_mark_params(Cn,b0* self.B0_table,Cn2=Cn2)
        dof = self.compute_table_power(pars,self.Bshot_table,Cn,Cn2=Cn2)

        # power spectrum terms 
        MA = self.compute_CdCdP_table(pars,f)
        MA = self.combine_mark_params(Cn,MA,Cn2=Cn2)

        stochastic = self.get_Nm_contr(pars,Cn=Cn,Cn2=Cn2)
        Cd = self.Cd(self.kv,Cn=Cn)
        Cd2 = self.Cd(self.kv,Cn=Cn2)
        lowk = np.array([db*self.plin*Cd*Cd2,df*self.plin*Cd*Cd2, np.zeros(self.kv.shape)])
    
        ret = MA + 2*(M13B+M13C) + M22B + M22C + SN + BSN + dof + lowk
        # ret = MA + 2*(M13B+M13C) + M22B + M22C + SN + BSN + dof + self.leg2poly(lowk)

        ret += stochastic
        
        if basis=='Polynomial':  return ret
        return self.poly2leg(ret)
        
    def P_at_mu(self,pars,f,mu_obs, apar=1., aperp=1.,bFoG=0):
        
        kv, pobs = self.rept.compute_redshift_space_power_at_mu(pars[:11],f,mu_obs, apar=apar, aperp=aperp,bFoG=bFoG)
        return kv, pobs
        
    def WP_at_mu(self,pars,f,mu_obs, apar=1., aperp=1.,bFoG=0):
        kv, pobs = self.rept.compute_redshift_space_power_at_mu(pars[:11],f,mu_obs, apar=apar, aperp=aperp,bFoG=bFoG)
        return kv, pobs*self.W_R(kv)
    
    def W2P_at_mu(self,pars,f,mu_obs, apar=1., aperp=1.,bFoG=0):
        kv, pobs = self.rept.compute_redshift_space_power_at_mu(pars[:11],f,mu_obs, apar=apar, aperp=aperp,bFoG=bFoG)
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

    def multsn(self,ar):
        res = np.zeros(ar.shape)
        res[16] += ar[0] # sn 
        res[22] += ar[1] # b1 sn 
        res[23] += ar[2] # b1^2 sn 
        res[24] += ar[12] # b1^3 sn 
        res[25] += ar[13] # b1^4 sn 
        res[26] += ar[16] # sn^2
        res[27] += ar[22] # b1 sn^2
        res[28] += ar[23] # b1^2 sn^2
        res[29] += ar[24] # b1^3 sn^2
        res[30] += ar[25] # b1^4 sn^2
        return res
        
    def multb1(self,ar):
        # 1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1 b3
        # b1^3, b1^4, b1^2 b2, b1^2 bs
        # sn, sn2, sn4, alpha0, alpha2, alpha4
        # b1 sn, b1^2 sn, b1^3 sn, b1^4 sn, 
        # sn^2, b1 sn^2, b1^2 sn^2, b1^3 sn^2, b1^4 sn^2 
        res = np.zeros(ar.shape)

        res[1] += ar[0] # b1
        res[2] += ar[1] # b1^2 
        res[4] += ar[3] # b1b2
        res[7] += ar[6] # b1bs
        res[11] += ar[10] # b1b3
        res[12] += ar[2] # b1^3
        res[13] += ar[12] # b1^4
        res[14] += ar[4] # b1^2 b2
        res[15] += ar[7] # b1^2 bs
        if len(res)>16:
            res[22] += ar[16] # b1 sn
            res[23] += ar[22] # b1^2 sn
            res[24] += ar[23] # b1^3 sn
            res[25] += ar[24] # b1^4 sn
            res[27] += ar[26] # b1 sn^2
            res[28] += ar[27] # b1^2 sn^2
            res[29] += ar[28] # b1^3 sn^2
            res[30] += ar[29] # b1^4 sn^2

            res[32] += ar[31] # b1^2 bshot
        return res
        
    def multb12(self,ar):
        # 1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1 b3
        # b1^3, b1^4, b1^2 b2, b1^2 bs
        # sn, sn2, sn4, alpha0, alpha2, alpha4
        # b1 sn, b1^2 sn, b1^3 sn, b1^4 sn, 
        # sn^2, b1 sn^2, b1^2 sn^2, b1^3 sn^2, b1^4 sn^2 
        res = np.zeros(ar.shape)

        res[2] += ar[0] # b1^2 
        res[12] += ar[1] # b1^3
        res[13] += ar[2] # b1^4
        res[14] += ar[3] # b1^2 b2
        res[15] += ar[6] # b1^2 bs
        res[23] += ar[16] # b1^2 sn
        res[24] += ar[22] # b1^3 sn
        res[25] += ar[23] # b1^4 sn
        res[28] += ar[26] # b1^2 sn^2
        res[29] += ar[27] # b1^3 sn^2
        res[30] += ar[28] # b1^4 sn^2
        
        return res
        
    def compute_M13C_table(self,f):
        kv = self.kv       
        mu_pow = self.mu_pow
        plin = self.plin
        plin_p = self.plin_p
        plin_IR_leg = self.plin_IR_leg
        plin_IR_poly = self.plin_IR_poly
        kint = self.kint
        xint = self.xint
        Cn = self.Cn #expansion coefficients
        C0, C1, C2, C3 = self.Cn
        W_R_int = self.W_R(kint)
        W_R = self.W_R(kv)
        Cd = self.Cd(kv)

        sigma2_R = np.zeros(33)
        sigma2_R[0] += simps(kint**2*W_R_int*(plin_p),x=kint)/(2*np.pi**2) # 1
        sigma2_R[16] += simps(kint**2*W_R_int,x=kint)/(2*np.pi**2) # sn

        sigma2_RR = np.zeros(33)
        sigma2_RR[0] += simps(kint**2*W_R_int**2*(plin_p),x=kint)/(2*np.pi**2) # 1
        sigma2_RR[16] += simps(kint**2*W_R_int**2,x=kint)/(2*np.pi**2) # sn

        S_R = np.zeros(33)
        S_R[0] += f**2/5* simps(kint**2*W_R_int*(plin_p),x=kint)/(2*np.pi**2) # 1
        S_R[1] += 2*f/3* simps(kint**2*W_R_int*(plin_p),x=kint)/(2*np.pi**2) # b1
        S_R[2] += simps(kint**2*W_R_int*(plin_p),x=kint)/(2*np.pi**2) # b1^2
        S_R[16] += f**2/5* simps(kint**2*W_R_int,x=kint)/(2*np.pi**2) # sn
        S_R[22] += 2*f/3* simps(kint**2*W_R_int,x=kint)/(2*np.pi**2) # b1 sn
        S_R[23] += simps(kint**2*W_R_int,x=kint)/(2*np.pi**2) # b1^2 sn
        
        S_RR = np.zeros(33)
        S_RR[0] += f**2/5* simps(kint**2*W_R_int**2*(plin_p),x=kint)/(2*np.pi**2) # 1
        S_RR[1] += 2*f/3* simps(kint**2*W_R_int**2*(plin_p),x=kint)/(2*np.pi**2) # b1
        S_RR[2] += simps(kint**2*W_R_int**2*(plin_p),x=kint)/(2*np.pi**2) # b1^2
        S_RR[16] += f**2/5* simps(kint**2*W_R_int**2,x=kint)/(2*np.pi**2) # sn
        S_RR[22] += 2*f/3* simps(kint**2*W_R_int**2,x=kint)/(2*np.pi**2) # b1 sn
        S_RR[23] += simps(kint**2*W_R_int**2,x=kint)/(2*np.pi**2) # b1^2 sn

        # 1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1 b3
        # b1^3, b1^4, b1^2 b2, b1^2 bs
        # sn, sn2, sn4, alpha0, alpha2, alpha4
        # b1 sn, b1^2 sn, b1^3 sn, b1^4 sn, 
        # sn^2, b1 sn^2, b1^2 sn^2, b1^3 sn^2, b1^4 sn^2 
        
        # C_0^2, C_0 C_1, C_0 C_2, C_0 C_3, C_1^2, C_1 C_2, C_1 C_3, C_2^2, C_2 C_3
        M13C_table = np.zeros((33,9,len(mu_pow),self.nk))
        # C_0 C_2 
        M13C_table[:,2] += np.einsum('mk,jk->mjk', np.einsum('m,k->mk',2*S_R,W_R)+np.einsum('m,k->mk',S_RR,np.ones(W_R.shape)) ,plin_IR_poly)
        M13C_table[:,2,0] += self.multsn(np.einsum('m,k->mk',2*sigma2_R,W_R)+np.einsum('m,k->mk',sigma2_RR,np.ones(W_R.shape)) )
        
        # C_1 C_2 
        M13C_table[:,5] += np.einsum('k,mk,jk->mjk',W_R,  np.einsum('k,m->mk',2*W_R,S_R)+np.einsum('m,k->mk',S_RR,np.ones(W_R.shape)) ,plin_IR_poly)
        M13C_table[:,5,0] += self.multsn( np.einsum('m,k->mk',2*sigma2_R,W_R**2)+np.einsum('m,k->mk',sigma2_RR,W_R) )
        
        # C_0 C_3
        M13C_table[:,3] += np.einsum('k,m,jk->mjk',3 *W_R, S_RR,plin_IR_poly)
        M13C_table[:,3,0] += self.multsn(np.einsum('m,k->mk',3*sigma2_RR,W_R) )

        # C_1 C_3
        M13C_table[:,6] += np.einsum('k,m,jk->mjk',W_R**2, S_RR  ,plin_IR_poly)
        M13C_table[:,6,0] += self.multsn( np.einsum('m,k->mk',sigma2_RR,W_R**2) )
        
        final_table = np.zeros(M13C_table.shape)
        final_table += self.multb12(M13C_table)
        final_table[:,:,1:] += 2*f*self.multb1(M13C_table)[:,:,:-1]
        final_table[:,:,2:] += f**2*M13C_table[:,:,:-2]
            
        return final_table
    
    def compute_M22C_table(self,f):
        kv = self.kv
        plin = self.plin
        plin_p = self.plin_p
        plin_IR_leg = self.plin_IR_leg
        plin_IR_poly = self.plin_IR_poly
        kint = self.kint
        Cn = self.Cn #expansion coefficients
        C0, C1, C2, C3 = self.Cn
        W_R_int = self.W_R(kint)
        W_R = self.W_R(kv)
        mu_pow = self.mu_pow
        P11_k = np.zeros((33,len(mu_pow),len(self.kint))) # no index for Cn
        
        # this includes sn term of linear power spectrum
        P11_k[0,0] += plin_p* f**2/5 # 1, mu^0
        P11_k[1,0] += plin_p* 2/3*f # b1
        P11_k[2,0] += plin_p # b1^2
        P11_k[16,0] += 1 # sn

        P11_k[0,1] += plin_p * 4/7*f # 1, mu^2
        P11_k[2,1] += plin_p * 4/3 # b1^2 

        P11_k[0,2] += plin_p *7/35*f**2 # 1, mu^4   
                
        # 1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1 b3
        # b1^3, b1^4, b1^2 b2, b1^2 bs
        # sn, sn2, sn4, alpha0, alpha2, alpha4
        # b1 sn, b1^2 sn, b1^3 sn, b1^4 sn, 
        # sn^2, b1 sn^2, b1^2 sn^2, b1^3 sn^2, b1^4 sn^2 
        
        # C_0^2, C_0 C_1, C_0 C_2, C_0 C_3, C_1^2, C_1 C_2, C_1 C_3, C_2^2, C_2 C_3
        M22C_table = np.zeros((33,9,len(mu_pow),self.nk))

        def loop_conv(ar1, ar2):
            # convolve all 33 bias terms and change them to polynomial basis
            # assume that ar1 and ar2 are np arrays of the same shape (33,len(mu_pow), len(self.kint))
            assert ar1.shape == ar2.shape
            
            res1 = np.zeros((33,len(mu_pow), len(self.kv)))
            # ar2 is 1
            for i in range(len(ar1)):
                res1[i] += self.conv(ar1[i],ar2[0])
                
            res2 = np.zeros((33,len(mu_pow), len(self.kv)))
            # ar2 is b1
            for i in range(len(ar1)):
                res2[i] += (self.conv(ar1[i],ar2[1]))

            res3 = np.zeros((33,len(mu_pow), len(self.kv)))
            # ar2 is b1^2
            for i in range(len(ar1)):
                res3[i] += (self.conv(ar1[i],ar2[2]))

            res4 = np.zeros((33,len(mu_pow), len(self.kv)))
            # ar2 is sn
            for i in range(len(ar1)):
                res4[i] += (self.conv(ar1[i],ar2[16]))

            res = res1 + self.multb1(res2) + self.multb12(res3) + self.multsn(res4)
            for i in range(len(ar1)):
                res[i] = self.leg2poly(res[i])
            return res
        
        # C_1 C_1
        M22C_table[:,4] += 0.5 * loop_conv( np.einsum('k,ijk->ijk',W_R_int,P11_k),np.einsum('k,ijk->ijk',W_R_int,P11_k) )
        M22C_table[:,4] += 0.5 * loop_conv( np.einsum('k,ijk->ijk',W_R_int**2,P11_k),P11_k )
        
        # C_1 C_2 
        # M22C_table[:,5] += 2 * loop_conv( np.einsum('k,ijk->ijk',W_R_int**2,P11_k),np.einsum('k,ijk->ijk',W_R_int,P11_k) )
        M22C_table[:,5] += 2 * loop_conv( W_R_int**2*P11_k,W_R_int*P11_k)
        # C_2 C_2
        M22C_table[:,7] += 2 * loop_conv( np.einsum('k,ijk->ijk',W_R_int**2,P11_k),np.einsum('k,ijk->ijk',W_R_int**2,P11_k) )
        
        M22C_table *= 2
        
        return M22C_table
    
    def get_zn(self,n,f,ks,mu_pows,ps,xs,stoch = False):
        # n is true n
        # 1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1 b3
        # b1^3, b1^4, b1^2 b2, b1^2 bs
        
        # make nested list for result. shape(12,mu_pow)
        res = [[0 for _ in range(len(mu_pows))] for _ in range(5)]
        
        denom = (ks**2+ps**2-2*ks*ps*xs)
        abb = (ks/ps+ps/ks)*xs
        
        if not stoch:
            if n==0:
                res[2][0] += 1/14*(10-7*abb+4*xs**2) # b1^2
                res[4][0] += 1/2 # b1 b2

                res[1][2] += f/14*(ks**2*(6-7*abb+8*xs**2)/denom) # b1
                res[2][2] += f/2 # b1^2
            elif n==1:
                res[1][1] += f/7*(7*ks**2*xs+7*ps**2*xs-2*ks*ps*(3+4*xs**2))/denom # b1
                res[2][1] += -f/2*(ks/ps+ps/ks) # b1^2

                res[1][3] += -f**2*ks/(2*ps) # b1
            elif n==2:
                res[1][0] += f/2*(2*(5/7-1/2*abb+2*xs**2/7)+ps**2*(6-7*abb+8*xs**2)/(7*denom)) # b1
                res[2][0] += f/2 # b1^2
                res[3][0] += f/2 # b2

                res[0][2] += f**2/14*ks**2*(6-7*abb+8*xs**2)/denom # 1
                res[1][2] += f**2/14*21 # b1
            elif n==3:
                res[0][1] += f**2/7*(7*ks**2*xs+7*ps**2*xs-2*ks*ps*(3+4*xs**2))/denom # 1
                res[1][1] += f**2/14*(-7*ks/ps-14*ps/ks) # b1

                res[0][3] += (-f**3*ks/(2*ps)) # 1
            elif n==4:
                res[0][0] += f**2/14*ps**2*(6-7*abb+8*xs**2)/denom # 1
                res[1][0] += f**2/2 # b1
                
                res[0][2] += f**3 # 1
                
            elif n==5:
                res[0][1]+= (-f**3*ps/(2*ks)) # 1
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
    
    def get_zn_22(self,n,f,ks,mu_pows,ps,xs,stoch=False):
        # n is true n
        
        # 1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1 b3
        # b1^3, b1^4, b1^2 b2, b1^2 bs
        # make nested list for result. shape(16,mu_pow)
        res = [[0 for _ in range(len(mu_pows))] for _ in range(16)]
        denom = (ks**2+ps**2-2*ks*ps*xs)
        if not stoch:
            if n==0:
                # Z_1(k-p) Z_1(p) Z_2(p,k-p)
                res[12][0] = 1/14*ks**2* (7*ks*xs/ps + 3-10*xs**2 )/denom # b1^3 
                res[14][0] = 1/2 # b1^2 b2

                res[2][2] = f/14/ps*ks**2*(14*ks**3*xs + ks**2*ps*(2-30*xs**2)+3*ks*ps**2*xs*(3+4*xs**2) - ps**3*(1+6*xs**2))/denom**2 # b1^2 
                res[4][2] = f/14/ps*ks**2* 7*ps /denom # b1 b2
                res[12][2] = f/14/ps*ks**2* 7*ps /denom # b1^3

                res[1][4] = f**2*ks**4/14*(7*ks*xs/ps -1 - 6*xs**2 )/denom**2 # b1
                res[2][4] = f**2*ks**4/2/denom**2 # b1^2
            elif n==1:
                res[2][1] += f*ks/14/ps *-2*ks**2*ps*(7*ks*xs+ps*(3-10*xs**2) )/denom**2 # b1^2 
                res[4][1] += f*ks/14/ps *-14*ps**2/denom # b1 b2
                res[12][1] += f*ks/14/ps *7*ks*(ks-2*ps*xs)/denom # b1^3

                res[1][3] += f**2*ks**3/14/ps *2*ps*(ps-7*ks*xs+6*ps*xs**2)/denom**2 # b1
                res[2][3] += f**2*ks**3 /14/ps *-7*(-2*ks**2+ps**2+4*ks*ps*xs )/denom**2 # b1^2

                # res[0,:,5] += f**3*ks**5/(2*ps*denom**2) # b1
            elif n==2:
                res[2][0] += f/14/ps*(ks**2+2*ps**2-2*ks*ps*xs) *ks**2*(7*ks*xs +ps*(3-10*xs**2) )/denom**2 # b1^2 
                res[4][0] += f/14/ps*(ks**2+2*ps**2-2*ks*ps*xs) *7*ps/denom # b1 b2 

                res[1][2] += f**2* ks**2 /14/ps *2*(7*ks**3*xs + 2*ks*ps**2*xs*(4+3*xs**2) - ps**3*(1+6*xs**2) + ks**2 *(ps-15*ps*xs**2)) /denom**2 # b1
                res[2][2] += f**2* ks**2 /14/ps *7*ps*(-2*ks**2+ps**2+4*ks*ps*xs)/denom**2 # b1^2
                res[3][2] += f**2* ks**2 /14/ps *7*ps/denom # b2

                res[0][4] += f**3*ks**4/14/ps*(ps-7*ks*xs+6*ps*xs**2)/denom**2 # 1
                res[1][4] += f**3*ks**4/denom**2 # b1
            elif n==3:
                res[1][1] += f**2*ks/14/ps *(-2)*ks**2*ps*(7*ks*xs + ps*(3-10*xs**2 ) )/denom**2 # b1
                res[2][1] += f**2*ks/14/ps *7*ks*(ks-2*ps*xs)*(ks**2+2*ps**2-2*ks*ps*xs)/denom**2 # b1^2 
                res[3][1] += f**2*ks/14/ps *(-14)*ps**2/denom # b2

                res[0][3] += f**3*ks**3/7/ps * ps*(ps-7*ks*xs+6*ps*xs**2)/denom**2 # 1
                res[1][3] += f**3*ks**3/7/ps * 7/denom # b1

                # res[0,:,5] += f**4 * ks**5/ (2*ps * denom**2) # 1
                
            elif n==4:
                res[1][0] += f**2*ps/14 * ks**2*(7*ks*xs+ps*(3-10*xs**2))/denom**2 # b1
                res[3][0] += f**2*ps/14 * 7*ps/denom # b2

                res[0][2] += -f**3*ks**2/14*ps*(ps-7*ks*xs+6*ps*xs**2)/denom**2 # 1
                res[1][2] += -f**3*ks**2/14*7*(ps**2 + 3*ks*(ks-2*ps*xs))/denom**2 # b1

                res[0][4] += -3*f**4*ks**4/(2*denom**2) # 1
                
            elif n==5:
                res[1][1] += f**3*ks**2*ps*(ks-2*ps*xs)/(2*denom**2) # b1
                
                res[0][3] += (3*f**4*ks**3*ps/(2*denom**2)) # 1

            elif n==6:
                res[0][0] += (-f**4*ks**2*ps**2/(2*denom**2)) # 1
                
                # res[:,0]+= (-f**4*ks**2*ps**2/(2*denom**2))
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
    
    def compute_integral13_table(self,f,unity=False,stoch=False):
        x = self.xint
        p = self.kint
        kv = self.kv
        mu_pow = self.mu_pow
        plin_p = self.plin_p
        plin_IR_leg = self.plin_IR_leg
        plin_IR_poly = self.plin_IR_poly
        
        Nskip = self.Nskip
        
        final_array = np.zeros((5,len(mu_pow),self.nk,len(p))) #powers of mu

        mu_pows = np.arange(np.max(mu_pow)+1)

        for ii in range((len(kv)-1)//Nskip+1):
            k = kv[ii*Nskip:(ii+1)*Nskip]
            ks, ps, xs = np.meshgrid(k,p,x,indexing='ij',copy=False)
            
            if unity: S_int = 1
            else: S_int = self.W_R(np.sqrt(ks**2+ps**2-2*ks*ps*xs))
            for n in range(np.max(mu_pow)+1):
                zn = self.get_zn(n,f,ks,mu_pows,ps,xs,stoch=stoch) 
                for m in range(n+1):
                    # sum over even and odd here, since only the end product is constrained by symmetry
                    Gnm = self.get_Gnm(n,m)
                    if np.max(np.abs(Gnm))<1e-15: 
                        del Gnm
                        continue
                    #complete integral of x then multiply w/ P
                    integral_px = np.zeros((len(zn),len(zn[0]),len(k),len(p)))
                    for i1 in range(len(zn)):
                        for j1 in range(len(zn[0])):
                            if isinstance(zn[i1][j1], np.ndarray):
                                if unity: tmp = .5*S_int*np.einsum('jkl,l->jkl',zn[i1][j1],Gnm)
                                else: tmp = .5*np.einsum('ikl,ikl,l->ikl',zn[i1][j1],S_int,Gnm)
                                integral_px[i1,j1] += np.einsum('jk,k->jk',simps(x=x,y=tmp,axis=2),plin_p)
                            elif zn[i1][j1]!=0:
                                if unity: 
                                    tmp = .5*S_int*zn[i1][j1]*Gnm # shape x
                                    tmp = simps(x=x,y=tmp) # scalar
                                    integral_px[i1,j1] += tmp*np.einsum('jk,k->jk',np.ones(integral_px[i1,j1].shape),plin_p)
                                else: 
                                    tmp = .5*zn[i1][j1]*np.einsum('ikl,l->ikl',S_int,Gnm)
                                    integral_px[i1,j1] += np.einsum('jk,k->jk',simps(x=x,y=tmp,axis=2),plin_p)
                    
                    for i in range(np.max(mu_pow)+1):
                        if m+i>np.max(mu_pow) or (m+i)%2!=0: continue
                        final_array[:5,round((m+i)/2),ii*Nskip:(ii+1)*Nskip,:]+=integral_px[:,i,:,:] # bias length is different 
                    del integral_px
                del zn
            del ks; del ps; del xs; del S_int
            
        return final_array
        
    def compute_M13B_table(self,f,stoch=False,write=False):
        x = self.xint
        p = self.kint
        kv = self.kv+0
        mu_pow = self.mu_pow
        plin_p = self.plin_p
        plin_IR_leg = self.plin_IR_leg
        plin_IR_poly = self.plin_IR_poly
        # 1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1 b3
        # b1^3, b1^4, b1^2 b2, b1^2 bs
        
        name=self.name
        if stoch:    
            # deprecated
            print('stoch functionality is deprecated')
            stoch_str = '_stoch'
            c0 = alpha0 / (2*b1)
            c1 = (alpha2/2 - c0 *f)/b1
        else: 
            stoch_str = ''
            c0 = 0
            c1 = 0
        # load tables
        tmp = os.path.join(self.basedir,name)
        if not os.path.exists(tmp) and write: os.makedirs(tmp)
        
        tmp = os.path.join(self.basedir, name, "integral13%s.json"%stoch_str)
        if os.path.exists(tmp):
            with open(tmp) as json_file:
                data = json.load(json_file)
                # self.integral13 = np.array(data['table'])
                self.integral13 = np.array(data)
        else: 
            print('no integral13... computing')
            self.integral13 = self.compute_integral13_table(f,stoch=stoch)
            if write:
                out_file = open(tmp, "w")
                json.dump(self.integral13.tolist(), out_file, indent = 6)
                out_file.close()

        tmp = os.path.join(self.basedir, name, "integral13_unity%s.json"%stoch_str)
        if os.path.exists(tmp):
            with open(tmp) as json_file:
                data = json.load(json_file)
                # self.integral13_unity = np.array(data['table'])
                self.integral13_unity = np.array(data)
        else: 
            print('no integral13_unity... computing')
            self.integral13_unity = self.compute_integral13_table(f,unity=True,stoch=stoch)
            if write:
                out_file = open(tmp, "w")
                json.dump(self.integral13_unity.tolist(), out_file, indent = 6)
                out_file.close()
    
        integral13 = self.integral13
        integral13_unity = self.integral13_unity
        
        W_R_p = self.W_R(p)
        
        # S=W_R(k-p)
        subintegral13_1 = simps(x=p,y= np.einsum('l,ijkl->ijkl',p**2,integral13/(2*np.pi**2)) ,axis=3)
        tmp = np.zeros((33,len(subintegral13_1[0]),len(subintegral13_1[0,0])))
        tmp[:len(subintegral13_1)] += subintegral13_1
        subintegral13_1 = tmp + 0.
        # S=W_R(p)
        subintegral13_2 = simps(x=p,y= np.einsum('l,ijkl->ijkl',p**2*W_R_p,integral13_unity/(2*np.pi**2)) ,axis=3)
        tmp = np.zeros((33,len(subintegral13_2[0]),len(subintegral13_2[0,0])))
        tmp[:len(subintegral13_2)] += subintegral13_2
        subintegral13_2 = tmp + 0.
        # S=W_R(k-p) W_R(p)
        subintegral13_3 = simps(x=p,y= np.einsum('l,ijkl->ijkl',p**2*W_R_p,integral13/(2*np.pi**2)) ,axis=3)
        tmp = np.zeros((33,len(subintegral13_3[0]),len(subintegral13_3[0,0])))
        tmp[:len(subintegral13_3)] += subintegral13_3
        subintegral13_3 = tmp + 0.
        
        # plin_IR_poly.shape = len(mu_pow),nk
        W_R = self.W_R(kv)
        # 1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1 b3
        # b1^3, b1^4, b1^2 b2, b1^2 bs

        # C_0^2, C_0 C_1, C_0 C_2, C_0 C_3, C_1^2, C_1 C_2, C_1 C_3, C_2^2, C_2 C_3
        M13B_table = np.zeros((33,9,len(mu_pow),self.nk))

        def polymul(ar1):
            res = np.zeros((33,len(mu_pow),self.nk))
            assert res.shape == ar1.shape
            # assume that ar2 is plin_IR (len(mu_pow),nk)
            # assert res.shape == ar2.shape
            
            for j in range(len(res)):
                for i in range(len(res[0,0])):
                    res[j,:,i] += np.polymul(ar1[j,:,i],plin_IR_poly[:,i])[:len(mu_pow)]
            return res
                    
        # C_0 C_1
        C0C1_contr = polymul(subintegral13_1 + subintegral13_2)
        
        M13B_table[:,1] += self.multb1(C0C1_contr)
        for i in range(len(mu_pow)-1): 
            M13B_table[:,1,i+1]+=C0C1_contr[:,i]*f
        M13B_table[:,1] *= 2
        
        # C_1^2 
        C1C1_contr = polymul(subintegral13_1+subintegral13_2)         
        C1C1_contr = np.einsum('k,ijk->ijk',W_R,C1C1_contr)

        M13B_table[:,4] += self.multb1(C1C1_contr)
        for i in range(len(mu_pow)-1): 
            M13B_table[:,4,i+1]+=C1C1_contr[:,i]*f
        M13B_table[:,4] *= 2

        # C_0 C_2 
        C0C2_contr = polymul(subintegral13_3)

        M13B_table[:,2] += self.multb1(C0C2_contr)
        for i in range(len(mu_pow)-1): 
            M13B_table[:,2,i+1]+=C0C2_contr[:,i]*f
        
        M13B_table[:,2] *= 4

        # C_1 C_2 
        C1C2_contr = polymul(subintegral13_3)
        C1C2_contr = np.einsum('k,ijk->ijk',W_R,C1C2_contr)

        M13B_table[:,5] += self.multb1(C1C2_contr)
        for i in range(len(mu_pow)-1): 
            M13B_table[:,5,i+1]+=C1C2_contr[:,i]*f
        
        M13B_table[:,5] *= 4
        
        return M13B_table
    
    def compute_integral22_table(self,f,Y,stoch=False):
        kv=self.kv
        p = self.kint
        x = self.xint
        Cn = self.Cn #expansion coefficients
        C0, C1, C2, C3 = self.Cn
        # Y_func = interp1d(p,Y,fill_value='extrapolate')
        Y_func = interp1d([0]+p.tolist(),[0]+Y.tolist(),fill_value='extrapolate')
        
        mu_pow = self.mu_pow
        Nskip = self.Nskip
        # Nskip = 50
        final_array = np.zeros((16,len(mu_pow),self.nk,len(p))) #powers of mu
        mu_pows = np.arange(np.max(mu_pow)+1)
        for ii in range((len(kv)-1)//Nskip+1):
            k = kv[ii*Nskip:(ii+1)*Nskip]
            ks, ps, xs = np.meshgrid(k, p,x,indexing='ij',copy=False)
            Yint = Y_func(np.sqrt(ks**2+ps**2-2*ks*ps*xs))
            for n in range(np.max(mu_pow)+1):
                zn = self.get_zn_22(n,f,ks,mu_pows,ps,xs,stoch=stoch) 
                for m in range(n+1):
                    Gnm = self.get_Gnm(n,m)
                    if np.max(np.abs(Gnm))<1e-15: 
                        del Gnm
                        continue
                        
                    integral_px = np.zeros((len(zn),len(zn[0]),len(k),len(p)))
                    for i1 in range(len(zn)):
                        for j1 in range(len(zn[0])):
                            if isinstance(zn[i1][j1], np.ndarray):
                                integrand = np.einsum('ikl,l,ikl->ikl',zn[i1][j1],Gnm,Yint)
                                integral_px[i1,j1] += simps(x=x, y=0.5*integrand,axis=2)
                            elif zn[i1][j1]!=0:
                                integrand = zn[i1][j1]*np.einsum('l,ikl->ikl',Gnm,Yint)
                                integral_px[i1,j1] += simps(x=x, y=0.5*integrand,axis=2)
                    
                    for i in range(np.max(mu_pow)+1):
                        if m+i>np.max(mu_pow) or (m+i)%2!=0: continue
                        final_array[:16,round((m+i)/2),ii*Nskip:(ii+1)*Nskip,:]+=integral_px[:,i,:,:]
                    del integral_px; del integrand
                del zn
            del ks; del ps; del xs; del Yint
        return final_array
    
    def compute_M22B_table(self,f,stoch=False,write=False):
        kv = self.kv
        p=self.kint
        mu_pow = self.mu_pow
        W_R = self.W_R(kv)
        W_R_int = self.W_R(self.kint)
        Cn = self.Cn #expansion coefficients
        C0, C1, C2, C3 = self.Cn
        plin_p= self.plin_p
        plin_p_nw = self.plin_p_nw
        plin_p_w = self.plin_p_w
        plin_IR_leg = self.plin_IR_leg
        plin_IR_poly = self.plin_IR_poly

        name=self.name
        if stoch:    stoch_str = '_stoch'
        else: stoch_str = ''
        tmp = os.path.join(self.basedir,name)
        if not os.path.exists(tmp) and write: os.makedirs(tmp)
        tmp = os.path.join(self.basedir, name, "integral22_W%s.json"%stoch_str)
        if os.path.exists(tmp):
            with open(os.path.join(self.basedir, name, "integral22_W%s.json"%stoch_str)) as json_file:
                data = json.load(json_file)
                self.integral22_W = np.array(data)
        else: 
            print('no integral22_W... computing')
            self.integral22_W = self.compute_integral22_table(f,W_R_int*plin_p,stoch=stoch)
            if write:
                out_file = open(tmp, "w")
                json.dump(self.integral22_W.tolist(), out_file, indent = 6)
                out_file.close()
            
        # Y = W_R(k-p) P_L(k-p)
        integral22_W = self.integral22_W

        # S= P_L(p)        W_R(k-p) P_L(k-p) 
        subintegral22_2 = simps(x=p,y= np.einsum('l,ijkl->ijkl',p**2*plin_p,integral22_W/(2*np.pi**2)) ,axis=3)
        tmp = np.zeros((33,len(subintegral22_2[0]),len(subintegral22_2[0,0])))
        tmp[:len(subintegral22_2)] += subintegral22_2
        subintegral22_2 = tmp + 0.

        # S= W_R(p) P_L(p) W_R(k-p) P_L(k-p) 
        subintegral22_3 = simps(x=p,y= np.einsum('l,ijkl->ijkl',p**2*plin_p*W_R_int,integral22_W/(2*np.pi**2)) ,axis=3)
        tmp = np.zeros((33,len(subintegral22_3[0]),len(subintegral22_3[0,0])))
        tmp[:len(subintegral22_3)] += subintegral22_3
        subintegral22_3 = tmp + 0.
        
        # 1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1 b3
        # b1^3, b1^4, b1^2 b2, b1^2 bs

        # C_0^2, C_0 C_1, C_0 C_2, C_0 C_3, C_1^2, C_1 C_2, C_1 C_3, C_2^2, C_2 C_3
        M22B_table = np.zeros((33,9,len(mu_pow),self.nk))

        # C_0 C_1
        C0C1_contr = 2 * (2* subintegral22_2)
        M22B_table[:,1] += C0C1_contr
        
        # C_1 C_1
        C1C1_contr = 2 * np.einsum('j,kij->kij',W_R,(2* subintegral22_2)) 
        M22B_table[:,4] += C1C1_contr

        # C_0 C_2
        C0C2_contr = 4 * subintegral22_3
        M22B_table[:,2] += C0C2_contr
        
        # C_1 C_2
        C1C2_contr = 4 * np.einsum('j,kij->kij',W_R,(subintegral22_3))
        M22B_table[:,5] += C1C2_contr

        return M22B_table    

    def compute_stoch_dof_table(self,f,stoch=False):
        # bispectrum stochastic piece proportional to new dof Bshot
        kv = self.kv
        p=self.kint
        mu_pow = self.mu_pow
        W_R = self.W_R(kv)
        W_R_int = self.W_R(self.kint)
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
        mu_pows, ks, ps = np.meshgrid(mu_pow,kv, p,indexing='ij',copy=False)
        
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
        int_P_0 = simps(x=p,y=ps[:,:,:]**2*(int_st0 )*plin_ps /(2*np.pi**2),axis=2)
        int_P_2 = simps(x=p,y=ps[:,:,:]**2*(int_st2 )*plin_ps /(2*np.pi**2),axis=2)
        # int_P_4 = simps(x=p,y=ps[:,:,:]**2*(int_st4 )*plin_ps /(2*np.pi**2),axis=2)

        # int_x e^kpxR^2 G_nm(x) mu_p^n P_L(p) W_R^2(p) dp/2pi**2 dx/2 for int_stn
        int_WP_0 = simps(x=p,y=ps[:,:,:]**2*(int_st0 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        int_WP_2 = simps(x=p,y=ps[:,:,:]**2*(int_st2 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        # int_WP_4 = simps(x=p,y=ps[:,:,:]**2*(int_st4 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n P_L(p) W_R(p) dp/2pi**2 dx/2 for int_stn
        # int_Pp2_0 = simps(x=p,y=ps[:,:,:]**4*(int_st0 )*plin_ps /(2*np.pi**2),axis=2)
        # int_Pp2_2 = simps(x=p,y=ps[:,:,:]**4*(int_st2 )*plin_ps /(2*np.pi**2),axis=2)
        # int_Pp2_4 = simps(x=p,y=ps[:,:,:]**4*(int_st4 )*plin_ps /(2*np.pi**2),axis=2)
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n P_L(p) W_R(p) dp/2pi**2 dx/2 for int_stn
        # int_WPp2_0 = simps(x=p,y=ps[:,:,:]**4*(int_st0 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        # int_WPp2_2 = simps(x=p,y=ps[:,:,:]**4*(int_st2 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        # int_WPp2_4 = simps(x=p,y=ps[:,:,:]**4*(int_st4 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n W_R^2(p) dp/2pi**2 dx/2 for int_stn
        int_W_0 = simps(x=p,y=ps[:,:,:]**2*(int_st0 )*W_R_int /(2*np.pi**2),axis=2)
        # int_W_2 = simps(x=p,y=ps[:,:,:]**2*(int_st2 )*W_R_int/(2*np.pi**2),axis=2)
        # int_W_4 = simps(x=p,y=ps[:,:,:]**2*(int_st4 )*W_R_int /(2*np.pi**2),axis=2)
        
        int_1_0 = np.zeros((len(mu_pow),self.nk))
        # W_R(p) W_R(k) inclusion for int_st doesn't apply here
        # int_x e^kpxR^2 G_nm(x) mu_p^n W_R(p) dp/2pi**2 dx/2 for int_stn
        int_1_0[0] = simps(x=p,y=ps[:,:,:]**2 *W_R_int /(2*np.pi**2),axis=2)[0]
                
        # 1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1 b3
        # b1^3, b1^4, b1^2 b2, b1^2 bs
        # sn, sn2, sn4, alpha0, alpha2, alpha4
        # b1 sn, b1^2 sn, b1^3 sn, b1^4 sn, 
        # sn^2, b1 sn^2, b1^2 sn^2, b1^3 sn^2, b1^4 sn^2 
        # b1 bshot, b1^2 bshot, b0
        # nm0, nm2
        
        #C0 C1
        temp_A = [np.polymul((int_1_0[:,i]),plin_IR_poly[:,i])[:len(mu_pow)] for i in range(len(kv))]
        temp_A = np.array(temp_A).T

        C0C1_contr = np.zeros((33,len(mu_pow),self.nk))
        C0C1_contr[32] += 2*temp_A  # b1^2 bshot
        C0C1_contr[31,1:] = f*2*temp_A[:-1]  # b1 bshot

        C0C1_contr[32] += (1+W_R) * int_P_0 # b1^2 bshot
        C0C1_contr[31] += f* (1+W_R) * int_P_2 # b1 bshot
                
        #C1 C1
        C1C1_contr = C0C1_contr*W_R
        
        #C0 C2
        temp_A = [np.polymul((int_W_0[:,i]),plin_IR_poly[:,i])[:len(mu_pow)] for i in range(len(kv))]
        temp_A = np.array(temp_A).T
        
        C0C2_contr = np.zeros((33,len(mu_pow),self.nk))
        C0C2_contr[32] += 2*temp_A  # b1^2 bshot
        C0C2_contr[31,1:] = f*2*temp_A[:-1]  # b1 bshot

        C0C2_contr[32] += 2 * int_WP_0 # b1^2 bshot
        C0C2_contr[31] += 2 * f * int_WP_2 # b1 bshot
        
        #C1 C2
        C1C2_contr = C0C2_contr*W_R
        
        # C_0^2, C_0 C_1, C_0 C_2, C_0 C_3, C_1^2, C_1 C_2, C_1 C_3, C_2^2, C_2 C_3
        final_array = np.zeros((33,9,len(mu_pow),self.nk))
        final_array[:,1]+=C0C1_contr
        final_array[:,4]+=C1C1_contr
        final_array[:,2]+=C0C2_contr
        final_array[:,5]+=C1C2_contr
                
        final_array *= 4 # contribution same between M22 and M13, so final is M22+2 M13 = 4 M13
        
        return final_array
    
    def compute_stoch_BSN_table(self,f):
        # bispectrum stochastic piece not proportional to new dof
        kv = self.kv
        p=self.kint
        mu_pow = self.mu_pow
        W_R = self.W_R(kv)
        W_R_int = self.W_R(self.kint)
        Cn = self.Cn 
        C0, C1, C2, C3 = self.Cn
        plin_p= self.plin_p
        plin_p_nw = self.plin_p_nw
        plin_p_w = self.plin_p_w
        plin_IR_leg = self.plin_IR_leg
        plin_IR_poly = self.plin_IR_poly
        R = self.R
        mu_pows, ks, ps = np.meshgrid(mu_pow,kv, p,indexing='ij',copy=False)
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
        int_W_0 = simps(x=p,y=ps[:,:,:]**2*(int_st0 )*W_R_int /(2*np.pi**2),axis=2)
        # int_W_2 = simps(x=p,y=ps[:,:,:]**2*(int_st2 )*W_R_int/(2*np.pi**2),axis=2)
        # int_W_4 = simps(x=p,y=ps[:,:,:]**2*(int_st4 )*W_R_int /(2*np.pi**2),axis=2)
        
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

        # 1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1 b3
        # b1^3, b1^4, b1^2 b2, b1^2 bs
        # sn, sn2, sn4, alpha0, alpha2, alpha4
        # b1 sn, b1^2 sn, b1^3 sn, b1^4 sn, 
        # sn^2, b1 sn^2, b1^2 sn^2, b1^3 sn^2, b1^4 sn^2 
        # b1 bshot, b1^2 bshot

        # final_array2 = np.zeros((34,9,len(mu_pow),self.nk))
        # final_array2[33] += final_array # b0
        
        return final_array

    def compute_stoch_SN_table(self,f,stoch = False):
        # bispectrum stochastic piece not proportional to new dof (Bshot), but to SN
        kv = self.kv
        p=self.kint
        mu_pow = self.mu_pow
        W_R = self.W_R(kv)
        W_R_int = self.W_R(self.kint)
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
        mu_pows, ks, ps = np.meshgrid(mu_pow,kv, p,indexing='ij',copy=False)
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
        # int_P_0 = simps(x=p,y=ps[:,:,:]**2*(int_st0 )*plin_ps /(2*np.pi**2),axis=2)
        int_P_2 = simps(x=p,y=ps[:,:,:]**2*(int_st2 )*plin_ps /(2*np.pi**2),axis=2)
        int_P_4 = simps(x=p,y=ps[:,:,:]**2*(int_st4 )*plin_ps /(2*np.pi**2),axis=2)

        # int_x e^kpxR^2 G_nm(x) mu_p^n P_L(p) W_R^2(p) dp/2pi**2 dx/2 for int_stn
        # int_WP_0 = simps(x=p,y=ps[:,:,:]**2*(int_st0 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        int_WP_2 = simps(x=p,y=ps[:,:,:]**2*(int_st2 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        int_WP_4 = simps(x=p,y=ps[:,:,:]**2*(int_st4 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n P_L(p) W_R(p) dp/2pi**2 dx/2 for int_stn
        # int_Pp2_0 = simps(x=p,y=ps[:,:,:]**4*(int_st0 )*plin_ps /(2*np.pi**2),axis=2)
        # int_Pp2_2 = simps(x=p,y=ps[:,:,:]**4*(int_st2 )*plin_ps /(2*np.pi**2),axis=2)
        # int_Pp2_4 = simps(x=p,y=ps[:,:,:]**4*(int_st4 )*plin_ps /(2*np.pi**2),axis=2)
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n P_L(p) W_R(p) dp/2pi**2 dx/2 for int_stn
        # int_WPp2_0 = simps(x=p,y=ps[:,:,:]**4*(int_st0 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        # int_WPp2_2 = simps(x=p,y=ps[:,:,:]**4*(int_st2 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        # int_WPp2_4 = simps(x=p,y=ps[:,:,:]**4*(int_st4 )*plin_ps*W_R_int /(2*np.pi**2),axis=2)
        
        # int_x e^kpxR^2 G_nm(x) mu_p^n W_R^2(p) dp/2pi**2 dx/2 for int_stn
        int_W_0 = simps(x=p,y=ps[:,:,:]**2*(int_st0 )*W_R_int /(2*np.pi**2),axis=2)
        # int_W_2 = simps(x=p,y=ps[:,:,:]**2*(int_st2 )*W_R_int/(2*np.pi**2),axis=2)
        # int_W_4 = simps(x=p,y=ps[:,:,:]**2*(int_st4 )*W_R_int /(2*np.pi**2),axis=2)
        
        int_1_0 = np.zeros((len(mu_pow),self.nk))
        # W_R(p) W_R(k) inclusion for int_st doesn't apply here
        # int_x e^kpxR^2 G_nm(x) mu_p^n W_R(p) dp/2pi**2 dx/2 for int_stn
        int_1_0[0] = simps(x=p,y=ps[:,:,:]**2 *W_R_int /(2*np.pi**2),axis=2)[0]

        # 1, b1, b1^2, b2, b1b2, b2^2, bs, b1bs, b2bs, bs^2, b3, b1 b3
        # b1^3, b1^4, b1^2 b2, b1^2 bs
        # sn, sn2, sn4, alpha0, alpha2, alpha4
        # b1 sn, b1^2 sn, b1^3 sn, b1^4 sn, 
        # sn^2, b1 sn^2, b1^2 sn^2, b1^3 sn^2, b1^4 sn^2 
        # b1 bshot, b1^2 bshot, b0
        # nm0, nm2
        
        #C0 C1
        temp_A = [np.polymul((int_1_0[:,i]),plin_IR_poly[:,i])[:len(mu_pow)] for i in range(len(kv))]
        temp_A = np.array(temp_A).T
        C0C1_contr = np.zeros((33,len(mu_pow),self.nk))

        C0C1_contr[22,1:] += (4*f*temp_A)[:-1] # b1 sn
        C0C1_contr[16,2:] += (4*f**2*temp_A)[:-2] # sn

        C0C1_contr[22] += 2*f*(1+W_R)*int_P_2 # b1 sn
        C0C1_contr[16] += 2*f**2*(1+W_R)*int_P_4 # sn

        #C1 C1
        C1C1_contr = C0C1_contr*W_R
        
        #C0 C2
        temp_A = [np.polymul((int_W_0[:,i]),plin_IR_poly[:,i])[:len(mu_pow)] for i in range(len(kv))]
        temp_A = np.array(temp_A).T
        C0C2_contr = np.zeros((33,len(mu_pow),self.nk))

        C0C2_contr[22,1:] += (4*f*temp_A)[:-1] # b1 sn
        C0C2_contr[16,2:] += (4*f**2*temp_A)[:-2] # sn

        C0C2_contr[22] += 4*f*int_WP_2 # b1 sn
        C0C2_contr[16] += 4*f**2*int_WP_4 # sn
        

        #C1 C2
        C1C2_contr = C0C2_contr*W_R

        # C_0^2, C_0 C_1, C_0 C_2, C_0 C_3, C_1^2, C_1 C_2, C_1 C_3, C_2^2, C_2 C_3
        final_array = np.zeros((33,9,len(mu_pow),self.nk))
        final_array[:,1]+=C0C1_contr
        final_array[:,4]+=C1C1_contr
        final_array[:,2]+=C0C2_contr
        final_array[:,5]+=C1C2_contr
        
        final_array *= 4 # contribution same between M22 and M13, so final is M22+2 M13 = 4 M13
        
        return final_array

    def get_Nm_contr(self,pars,Cn=None,Cn2=None):
        # polynomial basis
        b1, b2, bs, b3, alpha0, alpha2, alpha4, alpha6, sn, sn2, sn4,bshot, b0, nm0, nm2, db, df = pars

        kv = self.kv
        final_array = np.zeros((len(self.mu_pow),self.nk))
        final_array[0] += nm0 * kv**2 *(self.Cd(kv,Cn=Cn)+self.Cd(kv,Cn=Cn2))/2
        final_array[1] += nm2 * kv**2 *(self.Cd(kv,Cn=Cn)+self.Cd(kv,Cn=Cn2))/2
        
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
            new_coeffs = np.polynomial.legendre.poly2leg(coeffs)#.coefs
            new_coeffs = new_coeffs.tolist()
            while len(new_coeffs)<len(coeffs): new_coeffs.append(0)
            new_coeffs = np.array(new_coeffs)
            idxs = [2*j for j in range(len(self.mu_pow))]
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
    
    
