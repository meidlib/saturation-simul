#cw light driven two-level system
import time
import math
from scipy import integrate
#from scipy import signal
#from scipy.integrate import odeint
#from scipy.integrate import simps
from scipy.integrate import ode
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import gc


#pool=mp.Pool(processes=2)
#Transition frequency in MHz
w_0= 217105181895*1e3*1e-6
w_1=400
hbar=1.0545718e-34
dipol=5*3.3564*1e-4*1e-30
Ip=5e0
Im=1e-1
rabip=1.*dipol*Ip/(1*hbar)
rabim=1.*dipol*Im/(1*hbar)
rabi2p=1*dipol*Ip/hbar
rabi2m=1*dipol*Im/hbar
gamma=1e2
e_0=8.854187*1e-12
c=299792458
kb=1.38064852*1e-23
T=300*100
m=(2.014101+1.00794)*1.66054e-27
#Steps=100 -> 25 secs of calculation
ss=np.logspace(0,2,40)
detp=(ss-1)*50
d=-detp[1:]
detm=d[::-1]
dettune=np.concatenate((detm,detp),axis=0)
detune=dettune[20:58]
#print(detune)
#number of velocity groups / length=101
log_arrv=np.logspace(0,1.0455,51)
vp=(log_arrv-1)*10
xv=-vp[1:]
vm=xv[::-1]
dv=np.concatenate((vm,vp),axis=0)
nv=len(dv)
#Integration time and steps
t_start = 0
t_final = 1e-1
delta_t = 1e-3
t=np.arange(t_start,t_final, delta_t)
#print(1/(rabim+rabip))
vel=[]
i=0
k=0

def velocity():
    i=0
    while i in range (0,nv):
        #print(i)
        v=(2/301)*np.sqrt(kb*T/m)*(3*(i+.5-nv/2))
        vel.append(v)
        i=i+1
        
    return vel
velocity()

    
def position (t,i):
    return vel[i]*t
def interac(t, detune, i):
    return dipol*I*np.cos(1j*((w_0+detune)*t-((w_0/c)*position(t,i))))/hbar
#print(vel)
                          
def model(t,state, detune,k):
    #print(vel[k])
    #rabi=10
 
#   2 phase conjugated electic field rate equation with 2 ground states
    #   1t ground state p11
    dp11dt = (1j*rabip/2)*(state[2]-state[3])+(1j*rabim/2)*(state[4]-state[5])+2*gamma*(state[1])
    #   excited state p22
    dp22dt = (-1j*rabip/2)*(state[2]-state[3])+(-1j*rabim/2)*(state[4]-state[5])+(-1j*rabi2p/2)*(state[7]-state[8])+(-1j*rabi2m/2)*(state[9]-state[10])-4*gamma*state[1]
    #dp22dt = (-1j*rabip/2)*(state[2]-state[3])+(-1j*rabim/2)*(state[4]-state[5])-2*gamma*state[1]
    dp12pdt = ((1j*(detune-(w_1/2)-((w_0+(w_1/2))*vel[k])/c)-gamma)*state[2])+(1j*rabip/2*(state[0]-state[1]))-1j*(rabi2p/2)*state[12]
    dp21pdt = ((-1j*(detune-(w_1/2)-((w_0+(w_1/2))*vel[k])/c)-gamma)*state[3])-(1j*rabip/2*(state[0]-state[1]))+1j*(rabi2p/2)*state[11]
    dp12mdt=((1j*(detune-(w_1/2)+((w_0+(w_1/2))*vel[k])/c)-gamma)*state[4])+(1j*rabim/2*(state[0]-state[1]))-1j*(rabi2m/2)*state[12]
    dp21mdt = ((-1j*(detune-(w_1/2)+((w_0+(w_1/2))*vel[k])/c)-gamma)*state[5])-(1j*rabim/2*(state[0]-state[1]))+1j*(rabi2m/2)*state[11]
    #   Second ground state p33
    dp33dt=(1j*rabi2p/2)*(state[7]-state[8])+(1j*rabi2m/2)*(state[9]-state[10])+2*gamma*state[1]
    dp32pdt = ((1j*(detune+(w_1/2)-((w_0-(w_1/2))*vel[k])/c)-gamma)*state[7])+(1j*rabi2p/2*(state[6]-state[1]))-1j*(rabip/2)*state[11]
    dp23pdt = ((-1j*(detune+(w_1/2)-((w_0-(w_1/2))*vel[k])/c)-gamma)*state[8])-(1j*rabi2p/2*(state[6]-state[1]))+1j*(rabip/2)*state[12]
    dp32mdt=((1j*(detune+(w_1/2)+((w_0-(w_1/2))*vel[k])/c)-gamma)*state[9])+(1j*rabi2m/2*(state[6]-state[1]))-1j*(rabim/2)*state[11]
    dp23mdt =((-1j*(detune+(w_1/2)+((w_0-(w_1/2))*vel[k])/c)-gamma)*state[10])-(1j*rabi2m/2*(state[6]-state[1]))+1j*(rabim/2)*state[12]
    #   Off diagonal between the 2 ground states
    dp13dt=1*(1j*(((rabip/2)*state[2]+(rabim/2)*state[4])-((rabi2p/2)*state[8]+(rabi2m/2)*state[10])))-(state[11]*(gamma+1j*(w_1)))
    dp31dt=1*(-1j*(((rabip/2)*state[3]+(rabim/2)*state[5])-((rabi2p/2)*state[7]+(rabi2m/2)*state[9])))-(state[12]*(gamma-1j*(w_1)))
    

    dstatedt=[dp11dt,dp22dt,dp12pdt,dp21pdt,dp12mdt,dp21mdt,dp33dt,dp32pdt,dp23pdt,dp32mdt,dp23mdt,dp13dt,dp31dt]
    return dstatedt

def integration(detune,k):
    #r = integrate.complex_ode(model).set_integrator('vode', method='bdf')
    r=ode(model).set_integrator('zvode', method='bdf')
    t_start = 0
    t_final = 2e-1
    delta_t = 1e-4
    num_steps = int((t_final-t_start)/delta_t+1)

    rho11_t_zero = 1
    rho22_t_zero = 0
    rho12p_t_zero = 0
    rho21p_t_zero = 0
    rho12m_t_zero = 0
    rho21m_t_zero = 0
    rho33_t_zero = 1.0
    rho32p_t_zero = 0
    rho23p_t_zero = 0
    rho32m_t_zero = 0
    rho23m_t_zero = 0
    rho13_t_zero = 0
    rho31_t_zero = 0
    r.set_initial_value([rho11_t_zero,rho22_t_zero,rho12p_t_zero,rho21p_t_zero,rho12m_t_zero,rho21m_t_zero,rho33_t_zero,rho32p_t_zero,rho23p_t_zero,rho32m_t_zero,rho23m_t_zero,rho13_t_zero,rho31_t_zero],t_start)
    r.set_f_params(detune,k)

    t = np.zeros((int(num_steps), 1))
    rho11 = np.zeros((int(num_steps), 1),dtype=complex)
    rho22 = np.zeros((int(num_steps), 1),dtype=complex)
    rho12p =np.zeros((int(num_steps), 1),dtype=complex)
    rho21p =np.zeros((int(num_steps), 1),dtype=complex)
    rho12m =np.zeros((int(num_steps), 1),dtype=complex)
    rho21m =np.zeros((int(num_steps), 1),dtype=complex)
    rho33 = np.zeros((int(num_steps), 1),dtype=complex)
    rho32p =np.zeros((int(num_steps), 1),dtype=complex)
    rho23p =np.zeros((int(num_steps), 1),dtype=complex)
    rho32m =np.zeros((int(num_steps), 1),dtype=complex)
    rho23m =np.zeros((int(num_steps), 1),dtype=complex)
    rho13 =np.zeros((int(num_steps), 1),dtype=complex)
    rho31 =np.zeros((int(num_steps), 1),dtype=complex)
            
    rho11[0] = rho11_t_zero
    rho22[0] = rho22_t_zero
    rho12p[0] = rho12p_t_zero
    rho21p[0] = rho21p_t_zero
    rho12m[0] = rho12m_t_zero
    rho21m[0] = rho21m_t_zero
    rho33[0] = rho33_t_zero
    rho32p[0] = rho32p_t_zero
    rho23p[0] = rho23p_t_zero
    rho32m[0] = rho32m_t_zero
    rho23m[0] = rho23m_t_zero
    rho13[0] = rho13_t_zero
    rho31[0] = rho31_t_zero

    k = 1
    while r.successful() and k < num_steps:
        r.integrate(r.t + delta_t)

        t[k] = r.t
        rho11[k] = r.y[0]
        rho22[k] = r.y[1]
        rho12p[k] = r.y[2]
        rho21p[k] = r.y[3]
        rho12m[k] = r.y[4]
        rho21m[k] = r.y[5]
        rho33[k] = r.y[6]
        rho32p[k] = r.y[7]
        rho23p[k] = r.y[8]
        rho32m[k] = r.y[9]
        rho23m[k] = r.y[10]
        rho13[k] = r.y[11]
        rho31[k] = r.y[12]
        k += 1

    #IP=((rho12m[:,0].imag-rho21m[:,0].imag)-(rho12p[:,0].imag-rho21p[:,0].imag))/(Ip+Im)
        #print(k)

    IP12=(rho12m[:,0].imag-rho21m[:,0].imag)/(Im)
    IP32=(rho32m[:,0].imag-rho23m[:,0].imag)/(Im)
    IP=IP12+IP32
    #pop=rho11[:,0].real
    #plt.plot(t,pop)

    #plt.plot(t,rho12.real,"y-")
    #plt.plot(t,rho12.imag,"b-")
    #plt.plot(t,rho22.real,"k-")
    #plt.plot(t,IP12,"k-")
    #plt.plot(RP, "--")
    #plt.plot(peaks,RP[peaks], "--")
    #plt.plot(t,rho21.real,"k-")
    #plt.plot(t,rho13.imag,"b--")
    #plt.plot(t,rho21.imag,"r--")
    #area=simps(y[:,0], dx=delta_t )
    #area=simps(rho21[:,0].imag-rho12[:,0].imag, dx=5*1e-3 )
    #area=simps(rho12[:,0].real[peaks], dx=5*1e-3 )
    #area=np.trapz(IP[peaks], dx=5*1e-3 )
    #print(len(IP))
    #area=np.mean(IP[80:])
    #area=IP[-1]
    #print(area)
    plt.show()
    #return [IP12[-1],IP13[-1]]
    return IP[-1]


#integration(-1000)
i=0
liste=[]
#file.write("%s\t%s\n"%('detune(MHz)','intensity'))
def spectrum(k):
    i=0
    print(k)
    while i in range (0,len(detune)):
        res=integration (detune[i],k)
        liste.append(res)
        #file.write("%f\t%f\n"%(detune[i],res))
        i=i+1
    li=np.array(liste)
    #print(res)
    #field_abs=(w_0/c)*dipol*li*I
    
    plt.plot(detune,liste,"y--")
    plt.show()
    print("done")
    return liste
    

dopp=[]
spec=[]


def doppler(k):
        #f=(2/w_D)*np.sqrt(np.log(2)/np.pi)*np.exp(-4*np.log(2)*(detune[i]/w_D)**2)
    #f=np.exp(-(m*(vel[i])**2)/(2*kb*T))
    f=np.sqrt(m/(2*np.pi*kb*T))*np.exp(-(m*(vel[k])**2)/(2*kb*T))
    return f
ress=[]
ress1=[]
res1=[]
ress2=[]
res2=[]
spec12=[]
spec13=[]
def function_12(detune,k=0):
    i=0
    k=0
    result=integration(detune[i],k)*doppler(k)
    return result
def function_21(i,k):
    k=0
    i=0
    #print("k",k,"result",vel[k])
    return (integration(detune[i],k)[1]*doppler(k))
def sat_spectrum():
    ran=np.arange(40,61,1)
    for k in range (0,nv):
        #print("in k loop")
        print( "time", time.time()-time_s)
        pool=mp.Pool(processes=4)
        print("k",k,"result",vel[k])
        ans=[pool.apply_async(integration,args=(detune[x],k,))for x in range(0,len(detune))]
        #resu1=[p.get()*doppler(k) for p in ans]
        resu1=[p.get() for p in ans]
        pool.close()
        pool.join()
        nn=doppler(k)
        print(k,'doppler amp',nn)
        test1=np.multiply(resu1,nn)
        #plt.plot(detune, test1)
        #print(len(test1),test1)
        #print(resu1[:])
        res1.append(test1)
        plt.show()
            #print(len(res))
        #k=k+1

    res1_=np.array(res1)
    #print(len(res1_))
    step1=np.split(res1_,len(dv),axis=0)
    ress1=np.sum(step1,axis=0)
    plt.plot(detune, ress1[0],"o-")
    #dat=np.array([detune,ress1])
    #dat=dat.T
    #np.savetxt('pow_different_split_nv=21.txt', dat, delimiter = ',')
    time_e=time.time()
    duration=(time_e-time_s)/60
    frac=math.modf(duration)
    print('time for simulation: ',frac[1],'mins',frac[0]*60,'secs' )
    plt.show()
       
        
        #plt.plot(ress(k))
        #plt.show()
    #print(len(res))



#spectrum(80)

#field()
time_s=time.time()
print('simulation started at',time_s)

if __name__ ==  '__main__':
    sat_spectrum()

"""
print(vel)
f=[]
for i in range (nv):
    f.append(doppler(i))
plt.plot(vel,f,"o-")
plt.show()


#file.close()
#print(len(detune))
"""
