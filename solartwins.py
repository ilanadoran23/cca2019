from astropy.table import Table
import matplotlib.pyplot as plt
import numpy as np
import random
import math
import statistics
import scipy 
from condensation_temperature import * 

fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 15
fig_size[1] = 12
plt.rcParams["figure.figsize"] = fig_size

t= Table.read('solar_twins_data.fits') #fits file as table 
for i, words in enumerate(t['Fe']):
    t['Fe'][i] = 0
t

def star_table(star):
    for i, txt in enumerate(t['star_name']):
        if txt == star:
            tbl = t[i] #inputted star's row
            tbl

    star_elements =[]
    elnames = tbl.columns[3:64]
    for n in elnames:
        if len(n) < 3 :
            star_elements.append(n)
            star_elements #list of elements in that star
    
    star_abundance = []
    for n in star_elements:
        star_abundance.append(tbl[n])
        star_abundance #list of element abundances
        
    star_con_temp = []
    for n in star_elements:
        star_con_temp.append(tc_map[n])
        star_con_temp #condensation temperatures for stellar elements
    
    star_error_elements = []
    for r in elnames:
        if len(r) > 3 :
            star_error_elements.append(r) #list of elements recorded in star

    el_error = []
    for k in star_error_elements:
        el_error.append(tbl[k])
        el_error #list of error values for elements
        
    for x, txt in enumerate(star_abundance):
        if (math.isnan(txt) == True):
            del star_elements[x]
            del star_abundance[x]
            del star_con_temp[x]
            del el_error[x]
    
    star_table = Table([star_elements, star_abundance, el_error, star_con_temp], names=('Element', 'Abundance', 'Abundance Error','Condensation Temp')) #table of temperature vs abundance for elements 
    return star_table

#function for returning the best slope and intercept using linear algebra : Hogg eq 5 
#[m b] = [A^T C^-1 A]^-1 [A^T C^-1 Y]
def find_m_b(x,y,err): 
    #C 
    errorsq = np.square(err)
    C = np.diag(errorsq)
    
    #A
    xb = ([1] * len(x))
    mata = []   
    for z, txt in enumerate(x):
        mata.append(x[z])
        mata.append(xb[z])
    A= np.matrix(mata).reshape((len(x), 2))
    
    #plugging in 
    At = np.transpose(A)
    invC = np.linalg.inv(C)
    pt1 = np.dot(At, np.dot(invC,A))
    invpt1= np.linalg.inv(pt1)
    pt2 = np.dot(At, np.dot(invC, y)).T
    cov = np.dot(invpt1, pt2)
        
    m_= float(cov[0])
    b_= float(cov[1])
    return m_,b_ 

#jackknife method for determining other possible values of m and b 
def jackknifemb(_tp,_ab,_er):
    N=1000
    l=list(np.copy(_tp))
    k=list(np.copy(_ab))
    s=list(np.copy(_er))
    jackm= []
    jackb= [] 
    h=0
    
    #leaving out one point from data set and calculating m, b for each instance
    while h<N:
        w = random.randint(0, (len(_tp)-1))
        del l[w]
        del k[w]
        del s[w] #removing one data set from lists 
    
        jk_mb = find_m_b(l,k,s)
        jk_m = jk_mb[0]
        jk_b = jk_mb[1]

        jackm.append(jk_m) #alternate m values
        jackb.append(jk_b) #alternate b values
            
        l=list(np.copy(_tp)) #adding value back in for next round 
        k=list(np.copy(_ab)) 
        s=list(np.copy(_er))
        h=h+1 
        
    return jackm, jackb

def stellar_abundance_plot(star): 
    
    table = star_table(star)
    temp= np.array(table.columns[3])
    abund = np.array(table.columns[1])
    error = np.array(table.columns[2])
    elements = np.array(table.columns[0])
    
    plt.ioff()
    fig, ax = plt.subplots()

    ax.scatter(temp, abund)
    ax.set_xlabel('Tc',fontsize='xx-large', family='sans-serif')
    ax.set_ylabel('Element Abundance', fontsize='xx-large', family='sans-serif')
    ax.set_title('Temperature vs Element Abundance for {0}'.format(star), fontsize= 'xx-large', family='sans-serif')

    #point labels    
    for i, txt in enumerate(elements):
            ax.annotate(txt, xy=(temp[i], abund[i]), xytext=(-13,-6), 
                textcoords='offset points', ha='center', va='bottom')
    
    #alternate best fit lines
    jk= jackknifemb(temp, abund, error)
    for i, txt in enumerate (jk[0]):
        plot_xs = np.arange(0, 1750, .1)
        ax.plot(plot_xs, jk[0][i] * plot_xs + (jk[1][i]), color = 'lightgray', linewidth=0.1)

    #error bars
    ax.errorbar(temp, abund, yerr= error, fmt='o', color='black',
                 ecolor='lightsteelblue', elinewidth=3,capsize=0)
    
    #line of best fit m,b values
    mb = find_m_b(temp, abund, error) 
    plot_xs = np.arange(0, 1750, .1)
    ax.plot(plot_xs, (mb[0]) * plot_xs + (mb[1]), color='teal')

    fig.savefig(star+'.png')
    plt.close(fig)

def abund_plot_noCO(star): 
    table = star_table(star)
    temp= np.array(table.columns[3])
    abund = np.array(table.columns[1])
    error = np.array(table.columns[2])
    elements = np.array(table.columns[0])
    
    C_O_removed_error = [] #lists without C or O data -- outliers
    C_O_removed_temp = []
    C_O_removed_abund = []
    for h, name in enumerate(elements):
        if name != 'C':
            if name != 'O':
                C_O_removed_error.append(error[h])
                C_O_removed_temp.append(temp[h])
                C_O_removed_abund.append(abund[h])
    
    plt.ioff()
    fig, ax = plt.subplots()

    #point labels
    for i, txt in enumerate(elements):
            ax.annotate(txt, xy=(temp[i], abund[i]), xytext=(-13,-6),
                textcoords='offset points', ha='center', va='bottom')
    
    #alternate best fit lines  
    jk= jackknifemb(C_O_removed_temp, C_O_removed_abund, C_O_removed_error)
    for i, txt in enumerate (jk[0]):
        plot_xs = np.arange(0, 1750, .1)
        ax.plot(plot_xs, jk[0][i] * plot_xs + (jk[1][i]), color = 'lightgray', linewidth=0.1)
    
    #error bars 
    for u, name in enumerate(elements): #plotting points, with C and O in different colors
        if name == 'C':
            ax.errorbar(temp[u], abundance[u], yerr= error[u], fmt='o', color='blue',
                 ecolor='lightsteelblue', elinewidth=3, capsize=0)
        elif name == 'O' :
            ax.errorbar(temp[u], abundance[u], yerr= error[u], fmt='o', color='blue',
                 ecolor='lightsteelblue', elinewidth=3, capsize=0)
        else:
            ax.errorbar(C_O_removed_temp, C_O_removed_abund, yerr= C_O_removed_error, fmt='o', color='black',
                 ecolor='lightsteelblue', elinewidth=3, capsize=0)
    
    #plot labels
    ax.set_xlabel('Tc',fontsize='xx-large', family='sans-serif')
    ax.set_ylabel('Element Abundance', fontsize='xx-large', family='sans-serif')
    ax.set_title('Temperature vs Element Abundance for {0}'.format(star), fontsize= 'xx-large', family='sans-serif')
    
    #line of best fit m,b values
    mb = find_m_b(C_O_removed_temp, C_O_removed_abund, C_O_removed_error)    
    plot_xs = np.arange(0, 1750, .1)
    ax.plot(plot_xs, (mb[0]) * plot_xs + (mb[1]), color='teal') 
    
    fig.savefig(star+'noco.png')
    plt.close(fig)

#chi squared, Hogg 2010 eq 7 :  [Y - AX]^T C^-1 [Y - AX]
def chisquared(param, x, y, erro): 
    for h, txt in enumerate(y): #removing nan values
        if (math.isnan(txt) == True):
            del x[h]
            del y[h]
            del erro[h]
    
    #A
    ab = ([1] * len(x))
    Amat = []
    for z, txt in enumerate(x):
        Amat.append(x[z])
        Amat.append(ab[z])  
    A= np.array(Amat).reshape((len(x), 2)) 

    #C
    errorsq = np.square(erro)
    C = np.diag(errorsq)
    invsC = np.linalg.inv(C)

    #plugging in 
    AT= np.transpose(A)
    part1 = np.dot(AT, np.dot(invsC, A))
    invprt1= np.linalg.inv(part1)
    part2 = np.dot(AT, np.dot(invsC, y)).T
    X = np.dot(invprt1, part2)
    [X[0], X[1]] = param #for optimization - m and b 
    
    AX = np.dot(A,X)
    yax = (y - AX)
    yaxT = np.transpose(yax)
    yaxTinvsC = np.dot(yaxT, invsC)

    chisq = (np.dot(yaxTinvsC, yax))
    return (chisq)

def covmatrix(x, y, error): #Hogg 2010 eq 18     
    #removing nan data 
    for h, txt in enumerate(y):
        if (math.isnan(txt) == True):
            del x[h]
            del y[h]
            del error[h]
    
    #C    
    errororsq = np.square(error)
    errororC = np.diag(errororsq)
    abu = ([1] * len(x))
    
    #A 
    axer = np.copy(x)
    matri = []   
    for z, txt in enumerate(axer):
        matri.append(axer[z])
        matri.append(abu[z])        
    aa= np.matrix(matri).reshape((len(x), 2))
    
    #transpose of A and inverse of C, then plugged in 
    Att = np.transpose(aa)
    inverrororC = np.linalg.inv(errororC)
    inbrackets = np.dot(Att, np.dot(inverrororC, aa))
    
    covmatrix = np.linalg.inv(inbrackets)
    #covmatrix = [σ^2m, σmb, σmb, σ^2b]
    return covmatrix

def standardslopeerror(x, y, err):
    for h, txt in enumerate(y):
        if (math.isnan(txt) == True):
            del x[h]
            del y[h]
            del err[h]
    #C       
    errorsq = np.square(err)
    errorC = np.diag(errorsq)
    
    #A
    abu = ([1] * len(x))
    atemper = np.copy(x)
    matri = []
    for z, txt in enumerate(atemper):
        matri.append(atemper[z])
        matri.append(abu[z])
    aa= np.matrix(matri).reshape((len(x), 2))
    
    #plugging in 
    Att = np.transpose(aa)
    inverrorC = np.linalg.inv(errorC)
    prt1 = np.dot(Att, np.dot(inverrorC,aa))
    invt1= np.linalg.inv(prt1)
    prt2 = np.dot(Att, np.dot(inverrorC, y)).T
    covar = np.dot(invt1, prt2)
        
    _m_= float(covar[0])
    _b_= float(covar[1]) #standard slope, intercept values found with linalg 
    
    inbrackets = np.dot(Att, np.dot(inverrorC, aa))
    sserror = np.linalg.inv(inbrackets)
    #sserror = [σ^2m, σmb, σmb, σ^2b]
    sse = np.sqrt(sserror[0,0]) #standard slope error
    return sse

def standardintercepterror(x,y, err):
    for h, txt in enumerate(y):
        if (math.isnan(txt) == True):
            del x[h]
            del y[h]
            del err[h]
    #C       
    errorsq = np.square(err)
    errorC = np.diag(errorsq)
    
    abu = ([1] * len(x))
    atemper = np.copy(x)
    matri = []
    for z, txt in enumerate(atemper):
        matri.append(atemper[z])
        matri.append(abu[z])
    aa= np.matrix(matri).reshape((len(x), 2))
    
    Att = np.transpose(aa)
    inverrorC = np.linalg.inv(errorC)
    prt1 = np.dot(Att, np.dot(inverrorC,aa))
    invt1= np.linalg.inv(prt1)
    prt2 = np.dot(Att, np.dot(inverrorC, y)).T
    covar = np.dot(invt1, prt2)
        
    _m_= float(covar[0])
    _b_= float(covar[1]) #standard slope, intercept values found with linalg 
    
    inbrackets = np.dot(Att, np.dot(inverrorC, aa))
    sserror = np.linalg.inv(inbrackets)
    #sserror = [σ^2m, σmb, σmb, σ^2b]
    sie = np.sqrt(sserror[1,1]) #standard slope error
    return sie

def error_table(tp, ab, er):
    jackm = jackknifemb(tp, ab, er)[0]
    jackb = jackknifemb(tp, ab, er)[1]
    
    slopeer = standardslopeerror(tp,ab,er)
    interer= standardintercepterror(tp,ab,er)
    slopeintercept = find_m_b(tp,ab,er)
    
    error_type = ['slope', 'intercept']
    a = [slopeintercept[0], slopeintercept[1]]
    c = [statistics.stdev(jackm),statistics.stdev(jackb)]
    d = [slopeer, interer]
    tab = Table([error_type,a, c, d], names=('error type', 'value','standard dev', 
                                              'linear algebra uncertainty'))
    return tab

def residuals(star):
    table = star_table(star)
    temp= np.array(table.columns[3])
    abund = np.array(table.columns[1])
    error = np.array(table.columns[2])
    
    mborig = find_m_b(temp, abund, error)
    m = mborig[0]
    b = mborig[1]

    predicted_values = [] #abundance values from slope
    pv = 0 
    for u in temp: 
        pv = (m*u) + b
        predicted_values.append(pv)
        pv = 0

    prev = np.array(predicted_values)
    abu = np.array(abund)
    diff = abu - prev #difference between slope and measured values
    return diff

def abudiff(star): #plots for abundance differences     
    table = star_table(star)
    temp= np.array(table.columns[3])
    error = np.array(table.columns[2])
    elements = np.array(table.columns[0])
    
    diff = residuals(star)
    plt.ioff()
    fig, ax = plt.subplots()

    ax.scatter(temp, diff)
    ax.set_xlabel('Tc',fontsize='xx-large', family='sans-serif')
    ax.set_ylabel('Tc-Corrected Abundance', fontsize='xx-large', family='sans-serif')
    ax.set_title('Temperature vs Abundance for {0}'.format(star), fontsize= 'xx-large', family='sans-serif')

    #point labels
    for i, txt in enumerate(elements):
            ax.annotate(txt, xy=(star_con_temp[i], diff[i]), xytext=(-13,-6),
                textcoords='offset points', ha='center', va='bottom')

    jk= jackknifemb(temp, diff, error)
    for i, txt in enumerate (jk[0]):
        plot_xs = np.arange(0, 1750, .1)
        ax.plot(plot_xs, jk[0][i] * plot_xs + (jk[1][i]), color = 'lightgray', linewidth=0.1)

    #error bars
    ax.errorbar(temp, diff, yerr= error, fmt='o', color='black',
                 ecolor='lightsteelblue', elinewidth=3,capsize=0)

    #line of best fit m,b values
    mb = find_m_b(temp, diff, error)
    plot_xs = np.arange(0, 1750, .1)
    ax.plot(plot_xs, (mb[0]) * plot_xs + (mb[1]), color='teal')

    fig.savefig('tcremoved'+ star + '.png')
    plt.close(fig)


if __name__ == "__main__":
    mbvalues = []
    starrow = []
    x0 = [0,.1]

    for i, txt in enumerate(t['star_name']):
        tabl = find_stellar_abundances(txt)
        temperature= tabl.columns[3]
        elementab = tabl.columns[1]
        aberrors =tabl.columns[2]

    #mbvalues is an array (slope, intercept, standard deviation of slope & intercept, standard slope error, intercept error, chi squared)
        starrow.append(find_m_b(temperature,elementab, aberrors))
        jack = jackknifemb(np.array(temperature),np.array(elementab),np.array(aberrors))
        standarddev = (statistics.stdev(jack[0]),statistics.stdev(jack[1]))
        starrow.append(standarddev)
        errors = (standardslopeerror(temperature,elementab,aberrors), standardintercepterror(temperature,elementab,aberrors))
        starrow.append(errors)
        starrow.append(chisquared(x0, temperature, elementab, aberrors))
    
        mbvalues.append(starrow)
        starrow =[]
    
        plt.figure()
        #stellar_abundance_plot(txt)
        abudiff(txt)
