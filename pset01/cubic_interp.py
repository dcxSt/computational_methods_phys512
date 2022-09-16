import numpy as np

fun=np.exp

xi=np.linspace(-2,2,11)
yi=fun(x)

xfine=np.linspace(xi[1],xi[-2],1001)
ytrue=fun(xfine)
y_interp=np.zeros(xfine.shape)
for i in range(len(x)):
    # The index of the left value of the interp interval
    ind=np.max(np.where(x[i]>=xi)[0]) 
    # Get four values 
    x_use=xi[ind-1:ind+3]
    y_use=yi[ind-1:ind+3]
    pars=np.polyfit(x_use,y_use,3)
    pred=np.polyval(pars,x[i])
    y_interp[i]=pred

print("Err = ", np.std(y_interp-ytrue))

