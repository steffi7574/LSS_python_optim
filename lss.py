#!/usr/bin/env python

# V1: Solves the nonlinear LSS problem for Van der Pol's
#     Oscillator. 
#     Newtons method is used to update the state u, the 
#     timescale eta and lagrangemultiplier w in each iteration.
#     The left upper block of the linearized KKT-Matrix (the 
#     Jacobian) is approximated by the identity in order to 
#     make use of the Schurkomplement. 
#     The LSS problems is solved for a range of parameters mu
#     while the costfunction is plotted for each mu. 
#
# V2: - Subroutine for output file changed -> can handle variable 
#        collumns now
#     - Implementing derivatives:
#           - Blackbox J.diff(mu) is correct. 
#           - Break derivative dependencies in Newton-loop
#           - derivatives for iter == 0 is fine.
#             test should be done at iter == 8 (convergence) also !
#       FD and reduced gradient with adjoints works fine! 
#
# V3: Solves the nonlinear LSS problem for Lorenz Attractor with 
#     methods from V1
#
# V4: - Change names of residuals: Switch Ru and Rw !!!
#     - Implementing derivatives as in V2
#       FD and reduced gradient with adjoints match !

import sys
sys.path.append('..')
sys.path.append('../..')
from scipy.integrate import odeint

#import numpy as np
import numpad as np
from numpad import sparse
from numpad.adsparse import spsolve
#from scipy import sparse
#from scipy.sparse.linalg import spsolve

import matplotlib.pyplot as pypl
pypl.matplotlib.interactive(1)

#np.set_printoptions(threshold=np.nan)

import struct
def outputfile(vec, filename):
     #vec = vec.reshape(vec.shape[0],-1)
     ufile=open(filename,'w')
     for col in vec:
        char=''
        for elem in col:
            char=char + '%.40f ' %(elem)
        char += '\n'
        ufile.write(char)
     ufile.close()
     print('File written: ' +filename)

import resource
def using(point=""):
    usage=resource.getrusage(resource.RUSAGE_SELF)
    return '''%s: usertime=%s systime=%s mem=%s mb
           '''%(point,usage[0],usage[1],
                (usage[2]*resource.getpagesize())/1000000.0 )


def printSquare(square):
    # Print out square matrix
    columnHeader = ' '
    for i in range(len(square)):
        columnHeader += '\t' + str(i)
    print(columnHeader)

    i = 0;
    for col in square:
        char = str(i) + '\t'
        for elem in col:
            char = char + str(np.round(elem,1)) + '\t' # print row number and data
        print( char)
        i += 1


def vanderpol(uMid,mu):
# right hand side of the Van-der-Pol-ODE    
    shp = uMid.shape
    uMid = uMid.reshape([-1,2])
    rhs = np.zeros(uMid.shape, uMid.dtype)
    rhs[:,0] = uMid[:,1]
    rhs[:,1] = -uMid[:,0] + mu * (1 - uMid[:,0]**2) * uMid[:,1]
    return rhs.reshape(shp)


def lorenz(uMid, rho):
    shp = uMid.shape
    x, y, z = uMid.reshape([-1, 3]).T
    global sigma,beta
    sigma, beta = 10, 8./3
    dxdt, dydt, dzdt = sigma*(y-x), x*(rho-z)-y, x*y - beta*z
    return np.transpose([dxdt, dydt, dzdt]).reshape(shp)



def costfunction(CASE,u,target):
# return costfunction of ODE 'CASE'
    if CASE == 'vanderpol':
       meanz=(u[:,1]**8).mean(0)
       meanz=meanz**(1./8)
    elif CASE == 'lorenz':
       meanz = (u[:,2]).mean(0) 
    else :
       print('ERROR: Case unknown')
       stop
    J = 1./2.*(meanz-target)**2
    return meanz, J

def ddu(CASE, uc, param):
# partial derivative of right hand side of the ODE and its transpose (L and L*)
# in Lorenz-Case: use global sigma, beta
    global sigma, beta

    N,m = uc.shape[0], uc.shape[1]
    dfduc = np.zeros( (N,m,m) )
    dfdutc = np.zeros( (N,m,m) )

    if CASE == 'vanderpol':
        for k in range(N):
          dfduc[k,0,0] = 0.0
          dfduc[k,0,1] = 1.0
          dfduc[k,1,0] = -1.0 - 2.0 * param * uc[k,0] * uc[k,1]
          dfduc[k,1,1] = param * (1 - uc[k,0]**2)
          dfdutc[k] = dfduc[k].transpose()
    elif CASE == 'lorenz':
        for k in range(N):
          dfduc[k,0,0] = - sigma
          dfduc[k,0,1] = sigma
          dfduc[k,0,2] = 0.0
          dfduc[k,1,0] = param - uc[k,2]
          dfduc[k,1,1] = -1.0
          dfduc[k,1,2] = - uc[k,0]
          dfduc[k,2,0] = uc[k,1]
          dfduc[k,2,1] = uc[k,0]
          dfduc[k,2,2] = - beta
          dfdutc[k] = dfduc[k].transpose()
    else :
        print('ERROR: Case unknown')
        stop
    return [dfduc, dfdutc]


def residuum(CASE,uc,etac,wc,dudtc,dfdutc,dtc,urc,param):
# computes residuum of the KKT conditions
    N,m = uc.shape[0], uc.shape[1]
    uMid = 0.5 * (uc[1:] + uc[:-1])
    if CASE == 'vanderpol':
        g = vanderpol(uMid,param)
    elif CASE == 'lorenz':
        g = lorenz(uMid,param)
    else :
        print('ERROR: Case unknown')
        stop
    Ru = - (1.0 + etac[:,np.newaxis])*dudtc + g
    Reta = - etac - (wc * dudtc).sum(1)
    #SPECIFIC FOR LORENZ!! TODO: generalize!
    wb = np.concatenate(([[.0,.0,.0]], wc, [[.0,.0,.0]]),0) #expend w to boundary w(0)=w(T)=0
    etab = np.concatenate(([0.], etac, [0.]),0) #expend w to boundary w(0)=w(T)=0
    dtb = np.concatenate(([dtc[0]],dtc),0)
    wbMid = 0.5 * (wb[1:] + wb[:-1])
    Rw = -uc + urc + ( (1.0+etab[1:,np.newaxis])*wb[1:] - (1.0 + etab[:-1,np.newaxis])*wb[:-1] ) / dtb[:,np.newaxis]  \
         + (dfdutc*wbMid[:,np.newaxis]).sum(2) 
    return Ru, Reta, Rw  



def sparseSchurmatrix(etac, dudtc, dfduc, dtc):
# computes Schurcomplement matrix of linearized KKT matrix

    N,m = dudtc.shape[0], dudtc.shape[1]

    # lower left blocks
    np.eyedt = ( 1.0 + etac[:, np.newaxis, np.newaxis] ) * np.eye(m,m) / dtc[:,np.newaxis,np.newaxis]
    E = -np.eyedt - 0.5 * dfduc[:-1]
    f = dudtc
    G = +np.eyedt - 0.5 * dfduc[1:]
    def block_ij_to_element_ij(i, j, m):
             i_addition = np.arange(m)[:,np.newaxis] + np.zeros([m,m], int)
             j_addition = np.arange(m)[np.newaxis,:] + np.zeros([m,m], int)
             i = i[:,np.newaxis,np.newaxis] * m + i_addition
             j = j[:,np.newaxis,np.newaxis] * m + j_addition
             return i, j
 
    # construct B * B.T
    diag_data = (E[:,:,np.newaxis,:] * E[:,np.newaxis,:,:]).sum(3) \
              + f[:,:,np.newaxis] * f[:,np.newaxis,:]  \
              + (G[:,:,np.newaxis,:] * G[:,np.newaxis,:,:]).sum(3)

    upper_data = (G[:-1,:,np.newaxis,:] * E[1:,np.newaxis,:,:]).sum(3)
    lower_data = upper_data.transpose([0,2,1])
    
    diag_i = np.arange(diag_data.shape[0])
    diag_j = diag_i
    upper_i = np.arange(diag_data.shape[0] - 1)
    upper_j = upper_i + 1
    lower_i, lower_j = upper_j, upper_i
 
    diag_i, diag_j = block_ij_to_element_ij(diag_i, diag_j, m)
    upper_i, upper_j = block_ij_to_element_ij(upper_i, upper_j, m)
    lower_i, lower_j = block_ij_to_element_ij(lower_i, lower_j, m)
 
    data = np.hstack([np.ravel(diag_data), np.ravel(upper_data), np.ravel(lower_data)])
    i = np.hstack([np.ravel(diag_i), np.ravel(upper_i), np.ravel(lower_i)])
    j = np.hstack([np.ravel(diag_j), np.ravel(upper_j), np.ravel(lower_j)])
 
    return sparse.csr_matrix((data, (i, j))), E, G


def primal(CASE, uc,etac,wc,paramc, dtc, urc):
# returns approximate Newton updates for u, eta, w

    # compute dudt and derivative of right hand side
    dudt = (uc[1:] - uc[:-1]) / dtc[:,np.newaxis]
    [dfdu,dfdut] = ddu(CASE, uc, paramc)

    # compute Residuum
    [ru, reta, rw] = residuum(CASE,uc,etac,wc,dudt,dfdut,dtc,urc,paramc)

    
    # compute sparse Schur matrix
    [Schur,E,G] = sparseSchurmatrix(etac, dudt, dfdu, dtc)

    
    #solve linear system Schur*dw = b
    tmp = (E*rw[:-1,np.newaxis,:]).sum(2) \
        + (G*rw[1:,np.newaxis,:]).sum(2) \
        + dudt*reta[:,np.newaxis]
    b = -np.ravel(ru) + np.ravel(tmp)
    solv = spsolve(Schur,b)

    # compute Newton updates
    dw = solv.reshape([-1,m])
    GTdw = (G*dw[:,:,np.newaxis]).sum(1)
    ETdw = (E*dw[:,:,np.newaxis]).sum(1)
    du = rw -np.vstack([np.zeros([1,m]), GTdw]) \
            -np.vstack([ETdw, np.zeros([1,m])])
    deta = reta -(dudt * dw).sum(1) 
    
    # Newton update for u, eta, w
    Gu = uc + du
    Geta  = etac + deta
    Gw = wc + dw  

    return Gu,Geta,Gw, ru, reta, rw

def primal_convergence(ru, reta, rw, tol):
    norm_res = (np.ravel(ru)**2).sum() \
             + (np.ravel(reta)**2).sum() \
             + (np.ravel(rw)**2).sum() 
    norm_res = np.sqrt(norm_res)
    if norm_res < tol :
        bool = 1
    else:
        bool = 0
    return norm_res, bool

def adjoint(Jc,uc,etac,wc,u_adjc,eta_adjc,w_adjc,Guc,Getac,Gwc,muc):
# uses global variables: J,u,eta,w,u_adj,eta_adj,w_adj,Gu,Geta,Gw, mu

    # compute adjoint updates
    dJdu = np.array(Jc.diff(uc).todense()).reshape(u_adjc.shape)
    
    Gu_u_adj = (Guc * u_adjc).sum()
    Geta_eta_adj = (Getac * eta_adjc).sum()
    Gw_w_adj = (Gwc * w_adjc).sum()
    # wrt u
    dGu_du = np.array((Gu_u_adj).diff(uc)).reshape(u_adjc.shape)
    dGeta_du = np.array((Geta_eta_adj).diff(uc)).reshape(u_adjc.shape)
    dGw_du = np.array((Gw_w_adj).diff(uc)).reshape(u_adjc.shape)
    dNdu = dJdu + dGu_du + dGeta_du + dGw_du
    # wrt eta
    dGu_deta = np.array((Gu_u_adj).diff(etac)).reshape(eta_adjc.shape)
    dGeta_deta = np.array((Geta_eta_adj).diff(etac)).reshape(eta_adjc.shape)
    dGw_deta = np.array((Gw_w_adj).diff(etac)).reshape(eta_adjc.shape)
    dNdeta = dGu_deta + dGeta_deta + dGw_deta
    # wrt w
    dGu_dw = np.array((Gu_u_adj).diff(wc)).reshape(w_adjc.shape)
    dGeta_dw = np.array((Geta_eta_adj).diff(wc)).reshape(w_adjc.shape)
    dGw_dw = np.array((Gw_w_adj).diff(wc)).reshape(w_adjc.shape)
    dNdw = dGu_dw + dGeta_dw + dGw_dw
    # wrt mu
    dGu_dmu = np.array(Gu_u_adj.diff(muc))
    dGeta_dmu = np.array(Geta_eta_adj.diff(muc))
    dGw_dmu = np.array(Gw_w_adj.diff(muc))
    
    #compute reduced gradient
    red_grad = dGu_dmu + dGeta_dmu + dGw_dmu

    return dNdu, dNdeta, dNdw, red_grad.reshape(Jc.shape)


def adjoint_convergence(uad,etaad,wad,Nu,Neta,Nw,tol):
    norm_res = (np.ravel(uad - Nu)**2).sum() \
                 + (np.ravel(etaad - Neta)**2).sum() \
                 + (np.ravel(wad - Nw)**2).sum()
    norm_res = np.sqrt(norm_res)
    if norm_res < tol :
        bool = 1
    else :
        bool = 0
    return norm_res, bool


#switch cases
#CASE = 'vanderpol' 
CASE = 'lorenz' 


#initialize optimization loop 
ERR=''
maxNiter = 1  # number of inner primal/adjoint Newton updates
maxOiter = 1000  # number of optimization steps
atol = 1e-7   # absolute tolerance for primal and adjoint convergence
redtol = 1e-7   # absolute tolerance of reduced gradient
stepsize = 1e-1  # step size for design updates


# initialize Van-der-Pol setting
if CASE == 'vanderpol':
    mu = 0.9
    param = mu
    u0 = np.array([0.5,0.5])
    dt,T = 0.01,10
    tmp=int(T / dt)
    t = 30 + dt * np.arange(tmp)
    f = lambda u, t: vanderpol(u, mu)
    # init optimization
    target = 2.8

elif CASE == 'lorenz':
    params = np.linspace(27, 28, 5)
    param = params[0]
    u0 = np.array([0.5,0.5,0.5])
    dt,T = 0.01, 10.0
    t = 30 + dt * np.arange(int(T / dt))
    f = lambda u, t: lorenz(u, param)
    target = 22.5
else :
    print('ERROR: case unknown')
    stop


# run up to t[0]
N0 = int(t[0] / (t[-1]-t[0]) * t.size)
u0 = odeint(f, u0, np.linspace(0, t[0], N0+1))[-1]

# compute a reference trajectory
print('CASE = ', CASE)
print('Reference trajectory with param = ', param)
ur = np.array(odeint(f, u0, t-t[0]))
dt = t[1:] - t[:-1]
tmp = np.hstack((t[:,np.newaxis], ur))
outputfile(tmp, 'ur_' + CASE + str(np.round(param,1)) + '.dat')


u = ur.copy()
N,m = u.shape[0],u.shape[1]   # number of time steps t_0, ..., t_N-1 and state dimension
M = m*N + (N-1) + m*(N-1) #number of variables u, eta, w 
print('N,M = ', N,M)


#open files for output
costfile=open('cost_'+CASE+'_T'+str(T)+ '_dt'+str(dt[0])+'.dat','w')
redgradfile=open('redgrad_'+CASE+'_T'+str(T)+ '_dt'+str(dt[0])+'.dat','w')
adjresfile=open('adjres_'+CASE+'_T'+str(T)+ '_dt'+str(dt[0])+'.dat','w')
primresfile=open('primres_'+CASE+'_T'+str(T)+ '_dt'+str(dt[0])+'.dat', 'w')


# initialize primal variables and paramater
u = np.array(u)
eta = np.zeros((u[1:]).shape[0])
w = np.array(np.zeros((u[1:]).shape))
param = np.array(param)

# initialize adjoint variables
u_adj = np.zeros(u.shape)
eta_adj = np.zeros(eta.shape)
w_adj = np.zeros(w.shape)

param_old = param
red_grad_old = 0.0

# optimization loop
for iter in range(maxOiter):
    print('param = ', param)
    
    
        
    # Newton-loop
    for Niter in range(maxNiter):
    
    
        # break derivative dependencies
        param = np.array(np.base(param))
        u = np.array(np.base(u))
        eta = np.array(np.base(eta))
        w = np.array(np.base(w))
        u_adj = np.array(np.base(u_adj))
        eta_adj = np.array(np.base(eta_adj))
        w_adj = np.array(np.base(w_adj))


        # compute update of primal variables and compute residuum
        [Gu,Geta,Gw, Ru,Reta,Rw] = primal(CASE, u,eta,w,param, dt, ur)
        
        # compute costfunction
        [meanz,J] = costfunction(CASE, u,target)
        #print('J', J)
    
        # compute update of adjoint variables and reduced gradient 
        [dNdu, dNdeta, dNdw, red_grad] = adjoint(J,u,eta,w,u_adj,eta_adj,w_adj,Gu,Geta,Gw,param)
        redgradfile.write('%.40f \n' %red_grad)

        #check adjoint and primal convergence
        [adj_res, adj_conv] = adjoint_convergence(u_adj,eta_adj, w_adj, dNdu, dNdeta, dNdw, atol)
        adjresfile.write('%.40f \n' %(adj_res))
        print('iter, adj_res', iter, adj_res)

        [prim_res, prim_conv] = primal_convergence(Ru,Reta,Rw, atol)
        primresfile.write('%.40f \n' %(prim_res))
        print('iter, prim_res', iter, prim_res)
      
   
    
        # update adjoint variables 
        u_adj = dNdu
        eta_adj = dNdeta
        w_adj = dNdw

        # update primal variables
        u = Gu
        eta = Geta
        w = Gw  
    
        ##print memory consumption 
        print(using('Newton '+str(iter)))
    
        #pypl.figure(1)
        #pypl.xlabel('iteration')
        #pypl.ylabel('residuum')
        #pypl.semilogy(iter, prim_res,'o')
 
        #pypl.figure(2)
        #pypl.xlabel('time')
        #pypl.ylabel('u-residuum')
        #pypl.semilogy(t[1:], np.sqrt((Rw**2).sum(1))[:])
 
   
    #print error message in case of convergence failure
    #if iter == maxNiter-1:
    #    ERR = ERR + 'NO CONVERGENCE AT param = ' +str(param)+ \
    #              'primal residuum ' + str(prim_res) +        \
    #              'adjoint residuum ' + str(adj_res) + '\n'
    
   

    # update design 
    Hk = 1.0
    param = param - stepsize * Hk * red_grad
       

    # check convergence
    if (adj_conv and prim_conv):
          print('')
          print('Primal and Adjoint converged!!')
          print('Be happy and go home!')
          print('')
          #break
          if red_grad < redtol:
              print('')
              print('Optimization converged!!')
              print('')
              break
     
    # compute time scale
    tau = t.copy()
    dtau = dt * (1.0+eta)
    tau[1:] = t[0] + np.cumsum(dtau)

    # evaluate costfunction and gradient
    [meanz,J] = costfunction(CASE, u,target)
    costfile.write('%.40f %.40f \n' %(meanz,J))
    print('MeanZ, Cost J = ',meanz, J)
    print('red_grad = ', red_grad)
    
    # output trajectory and time scale
    #tmp = np.hstack((tau[:,np.newaxis], u))
    #outputfile(tmp, 'u_'+ CASE + str(np.round(param,1))+'.dat')

# print error messages
print(ERR)

#close output files
costfile.close()
redgradfile.close()
adjresfile.close()
primresfile.close()

## plot costfunction
#pypl.figure(1)
#pypl.xlabel('rho')
#pypl.ylabel('costfunction')
#char = 'target '+str(target)
#pypl.title(char)
#pypl.plot(params[1:],Jlist)

pypl.show(block=True)



## plot costfunction
#pypl.figure(1)
#pypl.xlabel('mu')
#pypl.ylabel('costfunction')
#char = 'target '+str(target)
#pypl.title(char)
#pypl.plot(mus,Jlist)
#
#pypl.show(block=True)





#def Schurmatrix(mu):
#    # lower left blocks
#    E=np.zeros((N-1, 2,2))
#    G = np.zeros((N-1,2,2))
#    f = np.zeros((N-1,2,1))
#    for i in range(N-1):
#        E[i] = - (1.0 + eta[i]) / dt[i] * np.eye(2) - 0.5 * dfdu[i]
#        G[i] = (1.0 + eta[i]) / dt[i] * np.eye(2) - 0.5 * dfdu[i+1]
#        f[i] = dudt[i].reshape(2,1)
#    
#    # assemble 
#    B = np.zeros((2*(N-1),2*N + (N-1)))
#    for i in range(1,N):
#        lz = 2 * i #Zeilen
#        ls = 3 * i #Spalten
#        B[lz-2:lz, ls-3:ls-1] = E[i-1]
#        B[lz-2:lz, ls-1] = f[i-1].reshape(2,)
#        B[lz-2:lz, ls:ls+2] = G[i-1]
#    
#    # compute symmetric Schurmatrix
#    Schur = dot(B, B.T)
#    
#    return Schur, B



    ## assemble full KKT-matrix
    #KKT = np.zeros((M,M))
    ##upper left:
    #KKT[0:2,0:2] = np.eye(2)   
    #for i in range(1,N):
    #    l = 3 * i
    #    #KKT[l-1, l-3:l-1] = - 1.0 / dt[i-1] * w[i-1].T
    #    KKT[l-1, l-1] = 1.0
    #    #KKT[l:l+2, l-1] = 1.0 / dt[i-1] * w[i-1]
    #    KKT[l:l+2, l:l+2] = np.eye(2)
    ## lower left:
    #l+=1
    #for i in range(1,N):
    #    lz = l+ 2 * i #Zeilen
    #    ls = 3 * i #Spalten
    #    KKT[lz-1:lz+1, ls-3:ls-1] = E[i-1]
    #    KKT[lz-1:lz+1, ls-1] = f[i-1].reshape(2,)
    #    KKT[lz-1:lz+1, ls:ls+2] = G[i-1]
    ## compute symmetric part
    #KKT = KKT + KKT.T - diag(KKT.diagonal())

    ##solve the linear system KKT* x = b
    #solv = linalg.solve(KKT,res)

    ##assemble update vectors du, deta, dw
    #du = np.zeros(u.shape)
    #deta = np.zeros(eta.shape)
    #dw = np.zeros(w.shape)
    #for i in range(N-1):
    #    du[i,0] = solv[3*i]
    #    du[i,1] = solv[3*i+1]
    #    deta[i] = solv[3*i+2]
    #du[N-1,0] = solv[3*(N-1)]
    #du[N-1,1] = solv[3*(N-1)+1]
    #l = 3*(N-1)+1
    #for i in range(1,N):
    #    dw[i-1,0] = solv[l+2*i-1]
    #    dw[i-1,1] = solv[l+2*i]

