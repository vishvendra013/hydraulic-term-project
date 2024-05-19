from re import X
import autograd as ad 

from autograd import grad, jacobian

import autograd.numpy as np

#Equations of at nodes
e1= lambda x: x[0] + 0*x[1] + 0*x[2] + 0*x[3] + 0*x[4] + x[5] + 0*x[6] - 2.5
e2= lambda x: -1*x[0] + x[1] + 0*x[2] + 0*x[3] + 0*x[4] + 0*x[5] + x[6] + 0
e3= lambda x: 0*x[0] + -1*x[1] + x[2] + 0*x[3] + 0*x[4] + 0*x[5] + 0*x[6] + 0.5
e4= lambda x: 0*x[0] + 0*x[1] + -1*x[2] + -1*x[3] + 0*x[4] + 0*x[5] + 0*x[6] + 1
e5= lambda x: 0*x[0] + 0*x[1] + 0*x[2] + x[3] + -1*x[4] + 0*x[5] + -1*x[6] + 1
#Equation of the loops
e6= lambda x: 0*(x[0]**2) + 130703.32*(x[1]**2) + 43567.77*(x[2]**2) - 130703.32*(x[3]**2) + 0*(x[4]**2) + 0*(x[5]**2) - 330842.79*(x[6]**2) + 0
e7= lambda x: 10163.49*(x[0]**2) + 0*(x[1]**2) + 0*(x[2]**2) + 0*(x[3]**2) - 130703.32*(x[4]**2) - 10338.84*(x[5]**2) + 330842.79*(x[6]**2) + 0
#jacobian of above functions
jac_e1=jacobian(e1)
jac_e2=jacobian(e2)
jac_e3=jacobian(e3)
jac_e4=jacobian(e4)
jac_e5=jacobian(e5)
jac_e6=jacobian(e6)
jac_e7=jacobian(e7)


i=0
error=1000
tol= 0.0001
maxiter=550

# M x N Matrix 
M=7
N=7


x_0 =np.array([1,1,1,1,1,1,1],dtype=float).reshape(N,1)

#Newton_Raphson method for calculating the discharge(Q) values in each pipe
while np.any(abs(error)>tol) and i<maxiter:
  fun_evaluate= np.array([e1(x_0),e2(x_0),e3(x_0),e4(x_0),e5(x_0),e6(x_0),e7(x_0)]).reshape(M,1)
  flat_x_0  = x_0.flatten()
  jac=np.array([jac_e1(flat_x_0),jac_e2(flat_x_0),jac_e3(flat_x_0),jac_e4(flat_x_0),jac_e5(flat_x_0),jac_e6(flat_x_0),jac_e7(flat_x_0)])
  jac=jac.reshape(N,M)

  x_new = x_0 - np.linalg.inv(jac)@fun_evaluate

  error= x_new - x_0

  x_0=x_new

  print(i)
  print(error)
  print("--------------------------")

  i=i+1

print("The Solution is")
print(x_new)
print("Discharge at AB",x_new[0],"m3/s")
print("Discharge at BC",x_new[1],"m3/s")
print("Discharge at CD",x_new[2],"m3/s")
print("Discharge at DE",x_new[3],"m3/s")
print("Discharge at EF",x_new[4],"m3/s")
print("Discharge at FA",x_new[5],"m3/s")
print("Discharge at BE",x_new[6],"m3/s")

#making array for length, diameter & elvation 
l=np.array([600,600,200,600,600,200,200])
d=np.array([0.25,0.15,0.10,0.15,0.15,0.20,0.20])
elv=np.array([30,25,20,20,22,25])
f=0.2
head_init=15

#Calculation of head
print("head at node 1 is = [" ,head_init,"]m")
for i in range(5):
  hf_nxt = (8 * f * l[i]*(x_new[i])**2) / ( 9.81 * d[i]**5 * 3.14 ** 2)
  if elv[i]>elv[i+1]:
    head=head_init-hf_nxt

    #print("hf_next is",hf_nxt);    `       `   R4DXFRESESZDWQ1`XQQE3RESC`
    print("head at node ",i+2,"is = " ,head,"m")
  else:
    head=head_init+hf_nxt
    print("head at node ",i+2,"is = " ,head,"m")
    head_init=head
