# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 12:05:44 2015

@author: irnakat
"""
import numpy as np
import mpmath as mpm
import os
# carbon copy of mt_rekursi
def mt_rekursi(res,tebal,freq):
    k = np.sqrt(1j*2.*np.pi*4.*np.pi*1e-7*(freq/res))
    t = tebal
    
    c = k[-2]/k[-1]
    for i in range(len(res)-2,0,-1):
        c = k[i-1]/k[i]*mpm.coth(k[i]*t[i]+mpm.acoth(c))
    c = 1./k[0]*mpm.coth(k[0]*t[0]+mpm.acoth(c))
#    if c.real!=c.real and c.imag!=c.imag:
#	print 'res',res[:5]
    return c
    
# carbon copy of mt_maju_rekursi
def mt_maju_rekursi(freq,res,thick):
    rhoa = np.zeros((len(freq)))
    phas = np.zeros((len(freq)))
    for i in range(len(freq)):
        c = mt_rekursi(res,thick,freq[i])
        rhoa[i] = 2.*np.pi*freq[i]*4.*np.pi*1e-7*np.abs(c**2)
        phas[i] = mpm.atan(c.real/-c.imag)
        
    return freq,rhoa,phas

# carbon copy of maju_cantik_mundur_cantik
def maju_cantik_mundur_cantik(calculID,freq,rhoa,phas,maxiter_depth=10,maxiter_rho=50,convcrit=0,demo=False):
    """
    Zohdy reborn
    """
    from IPython.display import clear_output,display
    import matplotlib as mpl
    mpl.use('Agg')
    import matplotlib.pyplot as plt
    import time,sys
    from copy import deepcopy
    import plotly.plotly as py
    import plotly.graph_objs as go
    import plotly.tools as tls
    from plotly import tools

    # sign-in to plot.ly server
    py.sign_in('irnaka','jqqy1w0m86')
    
    ts = 0.01 if demo else 0.
        
    rhoi = np.zeros((len(freq)))
    dhi = np.zeros((len(freq)))    
    
    for i in range(len(freq)):
        rhoi[i] = rhoa[i]*((np.pi/(2.*phas[i])-1.))
        dhi[i] = np.sqrt(rhoa[i]*1./freq[i]/2./np.pi/(4.*np.pi*1e-7))
    rhoi = np.abs(rhoi)
    tebal = np.diff(dhi)*1.2
    
    f0,rh0,ph0 = mt_maju_rekursi(freq,rhoi,tebal)
    
    print 'initial model done!'
    # perbaikan gizi ke arah depth
    eror = []
    erorpha = []
    erorrho = []
    for i in range(maxiter_depth):
        tebal = tebal*0.9
        f0,rh0,ph0 = mt_maju_rekursi(freq,rhoi,tebal)
        erorrho.append(np.sum((np.log(rhoa)-np.log(rh0))**2))
        erorpha.append(np.sum(np.log(phas/ph0))**2)
        if convcrit==0:
            eror.append((erorrho[-1]+erorpha[-1])/2.)
        elif convcrit==1:
            eror.append(erorrho[-1])
        elif convcrit==2:
            eror.append(erorpha[-1])
        if len(eror)>1 and eror[-1]>eror[-2]:
            print('Depth convergence has been reached!')
            tebal = tebal/0.9
            f0,rh0,ph0 = mt_maju_rekursi(freq,rhoi,tebal)
            iterdepth = i-1
            eror.pop(-1)
            erorpha.pop(-1)
            erorrho.pop(-1)
            break
        if i==maxiter_depth-1:
            iterdepth = i
    
    # perbaikan gizi ke arah rho
    fr = [];r = [];p = [];l = [];er = [];ri =[]
    for i in range(maxiter_rho):
	print 'rhoi\n',rhoi
        for j in range(len(freq)):
            rhotemp = deepcopy(rhoi)
	    if np.log(rh0[j])>0.:
    	        rhoi[j] = rhoi[j]* (np.log(rhoa[j])/np.log(rh0[j]))
	    else:
		rhoi[j] = rhoi[j]* (np.log(rhoa[j])-np.log(rh0[j]))

        f0,rh0,ph0 = mt_maju_rekursi(freq,rhoi,tebal)
	if i==0:
	    f0old = deepcopy(f0)
	    rh0old = deepcopy(rh0)
	    ph0old = deepcopy(ph0)
	    rhoiold = deepcopy(rhoi)
        erorrho.append(np.sum((np.log(rhoa)-np.log(rh0))**2))
        erorpha.append(np.sum(np.log(phas/ph0))**2)
        if convcrit==0:
            eror.append((erorrho[-1]+erorpha[-1])/2.)
        elif convcrit==1:
            eror.append(erorrho[-1])
        elif convcrit==2:
            eror.append(erorpha[-1])
        if eror[-1]>eror[-2] and eror[-3]>eror[-2]:
            fr.append(f0old)
            r.append(rh0old)
            p.append(ph0old)
            ri.append(rhoiold)
            er.append(eror[-2])
            l.append(len(eror)-2)
            #if len(l)==3:
            #    break
        if i==maxiter_rho-1:
            fr.append(f0old)
            r.append(rh0old)
            p.append(ph0old)
            ri.append(rhoiold)
            er.append(eror[-1])
            l.append(len(eror)-1)
        rhoiold = deepcopy(rhoi)
        f0old = deepcopy(f0)
        rh0old = deepcopy(rh0)
        ph0old = deepcopy(ph0)

    
    # finding global minima
    rhl = deepcopy(rh0)
    phl = deepcopy(ph0)
    minIndex = er.index(min(er))
    f0 = fr[minIndex]
    rh0 = r[minIndex]
    ph0 = p[minIndex]
    rhoi = ri[minIndex]
    
    # create the figure
    f = plt.figure(figsize=(16.,8.))

    # plot rho apparent
    ax1 = f.add_subplot(2,3,(1,2))
    ax1.plot(f0,rhl,'b',label='$\\rho_{app_{calc,it = %d}}$'%(maxiter_rho+iterdepth+1))
    ax1.plot(freq,rhoa,'go',label='$\\rho_{app_{obs}}$')
    ax1.plot(f0,rh0,'r',label='$\\rho_{app_{calc,min_{global}}}$\n$\\epsilon = %.4f$'%(er[minIndex]))
    boundx = np.exp(0.01*(np.log(np.max(f0)/np.min(f0))))
    ax1.set_xlim(np.exp(np.log(np.min(f0))-boundx),np.exp(np.log(np.max(f0))+boundx))
    ax1.set_yscale('log')
    ax1.set_xscale('log')
    ax1.set_title('Apparent resistivity and Phase')
    ax1.set_ylabel('apparent resistivity ($\\rho_{app}$)')
    ax1.set_xlabel('frequency (Hz)')
    ax1.grid(True,which='both')
    ax1.legend(loc='upper left',fancybox=True,framealpha=0.4)

    # plot phase
    ax1t = ax1.twinx()
    ax1t.plot(f0,np.degrees(phl),'b--')
    ax1t.plot(f0,np.degrees(phas),'gx')
    ax1t.plot(f0,np.degrees(ph0),'r--',label='$\\theta_{app_{calc,min_{global}}}$')
    ax1t.set_ylabel('phase')
    ax1t.set_xlim(np.exp(np.log(np.min(f0))-boundx),np.exp(np.log(np.max(f0))+boundx))
    ax1t.legend(loc='lower right',fancybox=True,framealpha=0.4)

    # plot iteration error
    ax2 = f.add_subplot(2,3,(4,5))
    ax2.plot(np.arange(1,iterdepth+i+3),eror,'g',label='rho iter (mean)')
    ax2.plot(np.arange(1,iterdepth+2),eror[:iterdepth+1],'r',label='thick iter (mean)')
    ax2.plot(np.arange(1,iterdepth+i+3),erorrho,'g-.',label='rho iter ($\\rho_{app}$)')
    ax2.plot(np.arange(1,iterdepth+2),erorrho[:iterdepth+1],'r',label='thick iter ($\\rho_{app}$)')
    ax2.plot(np.arange(1,iterdepth+i+3),erorpha,'g--',label='rho iter ($\\theta$)')
    ax2.plot(np.arange(1,iterdepth+2),erorpha[:iterdepth+1],'r--',label='thick iter ($\\theta$)')
    ax2.scatter(l[minIndex]+1,er[minIndex],s=400,c='blue',marker=(5, 2),label='global minima ($\\rho_{app}$)')
    ax2.set_xlim(1,maxiter_rho+iterdepth)
    ax2.set_xlabel('iteration')
    ax2.set_ylabel('error')
    ax2.grid(True,which='both')
    ax2.set_yscale('log')
    ax2.legend(loc='best',fancybox=True,framealpha=0.4)

    # plot depth profile
    resist,kedalaman = gambar_model(rhoi,tebal)
    ax3 = f.add_subplot(2,3,(3,6))
    ax3.plot(resist,kedalaman)
    ax3.set_xscale('log')
    ax3.set_ylabel('depth (m)')
    ax3.set_xlabel('resistivity ($\\rho$)')
    ax3.grid(True,which='both')
    ax3.set_ylim(kedalaman[0],kedalaman[-1])
    ax3.invert_yaxis()

    # deleting old data
    basedir = '/media/serverdisk//MTZohdy'
    remove_trash(basedir)

    # export the figure
    plt.tight_layout()
    plt.savefig(basedir+'/images/out'+calculID+'.png')

    # export numeric data
    np.savetxt(basedir+'/outputs/data_'+calculID+'.txt',(freq,rhoa,rh0,phas,ph0),fmt='%1.5e')
    np.savetxt(basedir+'/outputs/iter_'+calculID+'.txt',(np.arange(1,iterdepth+i+3),eror,erorrho,erorpha),fmt='%1.5e')
    np.savetxt(basedir+'/outputs/out_'+calculID+'.txt',(resist,kedalaman),fmt='%1.5e')
    
    # converting to plot.ly figure
    """
    trace1 = go.Scatter(x = f0,y = rhl, 
                        mode='lines', 
                        name='calc $\\rho_{app}$,last iter',
                        xaxis = 'x1',
                        yaxis = 'y1')
    trace2 = go.Scatter(x = freq,y = rhoa, 
                        mode='markers', 
                        name='obs $\\rho_{app}$',
                        xaxis = 'x1',
                        yaxis = 'y1')
    trace3 = go.Scatter(x = f0,y = rh0, 
                        mode='lines', 
                        name='calc $\\rho_{app}$,$min_{glob}$',
                        xaxis = 'x1',
                        yaxis = 'y1')
    trace4 = go.Scatter(x = f0,y = np.degrees(phl), 
                        mode='lines', 
                        name='calc $\\theta$,last iter',
                        xaxis = 'x1',
                        yaxis = 'y2')
    trace5 = go.Scatter(x = freq,y = np.degrees(phas), 
                        mode='markers', 
                        name='obs $\\theta$',
                        xaxis = 'x1',
                        yaxis = 'y2')
    trace6 = go.Scatter(x = f0,y = np.degrees(ph0), 
                        mode='lines', 
                        name='calc $\\theta$,$min_{glob}$',
                        xaxis = 'x1',
                        yaxis = 'y2')

    trace7 = go.Scatter(x = np.arange(1,iterdepth+maxiter_rho+3), y = eror,
                        mode='lines',
                        marker = dict(
                            line = dict(
                                color = 'rgb(0,255,0)'
                                )),
                        name = '$\\rho_{app}$ iter rho',
                        xaxis = 'x3',
                        yaxis = 'y3')
    trace8 = go.Scatter(x = np.arange(1,iterdepth+2), y = eror[:iterdepth+1],
                        mode='lines',
                        marker = dict(
                            line = dict(
                                color = 'rgb(255,0,0)'
                                )),
                        name = '$\\rho_{app}$ iter thickness',
                        xaxis = 'x3',
                        yaxis = 'y3')

    trace9 = go.Scatter(x = np.arange(1,iterdepth+maxiter_rho+3), y = erorpha,
                        mode='lines',
                        marker = dict(
                            line = dict(
                                color = 'rgb(0,255,0)'
                                )),
                        name = '$\\theta$ iter rho',
                        xaxis = 'x3',
                        yaxis = 'y3')
    trace10 = go.Scatter(x = np.arange(1,iterdepth+2), y = erorpha[:iterdepth+1],
                        mode='lines',
                        marker = dict(
                            line = dict(
                                color = 'rgb(255,0,0)'
                                )),
                        name = '$\\theta$ iter thickness',
                        xaxis = 'x3',
                        yaxis = 'y3')

    trace11 = go.Scatter(x = resist,y = kedalaman,
                         mode = 'lines',
                         xaxis = 'x4',
                         yaxis = 'y4')

    data = [trace1,trace2,trace3,trace4,trace5,trace6,trace7,trace8,trace9,trace10,trace11]

    layout = go.Layout(
        yaxis = dict(
            title = 'Apparent Resistivity',
            domain = [0.55,1.0],
            type = 'log'
            ),
        yaxis2 = dict(
            title = 'Phase',
            domain = [0.55,1.0],
            overlaying = 'y',
            side = 'right'
            ),
        yaxis3 = dict(
            title = 'Error',
            domain = [0.0,0.45],
            type = 'log'
            ),
        xaxis = dict(
            title = 'Frequency (Hz)',
            domain = [0.0,0.45],
            type = 'log'
            ),
        xaxis3 = dict(
            title = 'Number iteration',
            domain = [0.0,0.45],
            anchor = 'y3'
            ),
        xaxis4 = dict(
            title = 'Resistivity',
            domain = [0.55,1.0],
            type = 'log'
            ),
        yaxis4 = dict(
            title = 'Depth',
            domain = [0.55,1.0],
            anchor = 'x4',
            autorange = 'reversed'
            ),
        showlegend=False
        )
    fig = go.Figure(data=data,layout=layout)
    """
    resist,kedalaman = gambar_model(rhoi,tebal)
    trace1 = go.Scatter(x = f0,y = rhl,
                        mode='lines',
                        name='rho app on last iteration')
    trace2 = go.Scatter(x = freq,y = rhoa,
                        mode='markers',
                        name='rho app observed')
    trace3 = go.Scatter(x = f0,y = rh0,
                        mode='lines',
                        name='rho app on global minimum')
    trace4 = go.Scatter(x = f0,y = np.degrees(phl),
                        mode='lines',
                        name='theta on last iteration')
    trace5 = go.Scatter(x = freq,y = np.degrees(phas),
                        mode='markers',
                        name='theta observed')
    trace6 = go.Scatter(x = f0,y = np.degrees(ph0),
                        mode='lines',
                        name='theta calc global minimum')
    trace7 = go.Scatter(x = np.arange(1,iterdepth+maxiter_rho+3), y = eror,
                        mode='lines',
                        marker = dict(
                            line = dict(
                                color = 'rgb(0,255,0)'
                                )),
                        name = 'mean on rho iteartion')
    trace8 = go.Scatter(x = np.arange(1,iterdepth+2), y = eror[:iterdepth+1],
                        mode='lines',
                        marker = dict(
                            line = dict(
                                color = 'rgb(255,0,0)'
                                )),
                        name = 'mean on thickness iteration')

    trace9 = go.Scatter(x = np.arange(1,iterdepth+maxiter_rho+3), y = erorpha,
                        mode='lines',
                        marker = dict(
                            line = dict(
                                color = 'rgb(0,255,0)'
                                )),
                        name = 'phase on rho iteration')
    trace10 = go.Scatter(x = np.arange(1,iterdepth+2), y = erorpha[:iterdepth+1],
                        mode='lines',
                        marker = dict(
                            line = dict(
                                color = 'rgb(255,0,0)'
                                )),
                        name = 'phase on thickness iteration')

    trace11 = go.Scatter(x = np.arange(1,iterdepth+maxiter_rho+3), y = erorrho,
                        mode='lines',
                        marker = dict(
                            line = dict(
                                color = 'rgb(0,255,0)'
                                )),
                        name = 'rho on rho iteration')
    trace12 = go.Scatter(x = np.arange(1,iterdepth+2), y = erorrho[:iterdepth+1],
                        mode='lines',
                        marker = dict(
                            line = dict(
                                color = 'rgb(255,0,0)'
                                )),
                        name = 'rho on thickness iteration')

    trace13 = go.Scatter(x = l[minIndex]+1, y = er[minIndex],
                         mode = 'markers',
                         name = 'global minima')

    trace14 = go.Scatter(x = resist,y = kedalaman,
                         mode = 'lines',
                         name = 'depth vs resistivity')

    fig = tools.make_subplots(rows=3, cols=2,
                              specs=[[{},{'rowspan':3}],
                                     [{},None],
                                     [{},None]],
                              print_grid = True)
    fig.append_trace(trace1,1,1)
    fig.append_trace(trace2,1,1)
    fig.append_trace(trace3,1,1)
    fig.append_trace(trace4,2,1)
    fig.append_trace(trace5,2,1)
    fig.append_trace(trace6,2,1)
    fig.append_trace(trace7,3,1)
    fig.append_trace(trace8,3,1)
    fig.append_trace(trace9,3,1)
    fig.append_trace(trace10,3,1)
    fig.append_trace(trace11,3,1)
    fig.append_trace(trace12,3,1)
    fig.append_trace(trace13,3,1)
    fig.append_trace(trace14,1,2)

    fig['layout']['xaxis1'].update(title='Frequency (Hz)',
                                   type='log')
    fig['layout']['xaxis3'].update(title='Frequency (Hz)',
                                   type='log')
    fig['layout']['xaxis4'].update(title='Number of Iteration')
    fig['layout']['xaxis2'].update(title='Resistivity',
                                   type='log')

    fig['layout']['yaxis1'].update(title='Apparent Resistivity',
                                   type='log')
    fig['layout']['yaxis3'].update(title='Phase')
    fig['layout']['yaxis4'].update(title='Error',
                                   type='log')
    fig['layout']['yaxis2'].update(title='Depth (m)',
                                   autorange='reversed')

    fig['layout'].update(title='inversion result',
                         showlegend=False)

    try:
        plot_url_freq_rho_phase = py.plot(fig,filename='MTZohdy_output',auto_open=False)
        
        print tls.get_embed(plot_url_freq_rho_phase)
        plotly_success = True
    except Exception:
        plotly_success = False
    return f0,rh0,ph0,rhoi,tebal,plotly_success

# carbon copy of gambar_model
def gambar_model(res,thick):
    import pylab as plt
    resist = np.reshape(np.transpose(np.concatenate((np.array([res]), \
                                                     np.array([res])))),(len(res)*2,1))
    nr = len(resist)
    kedalaman = np.zeros((nr,1))
    dalam = np.reshape(np.transpose(np.concatenate((np.array([np.cumsum(thick)]), \
                                                     np.array([np.cumsum(thick)])))), \
                       (len(np.cumsum(thick))*2,1))
    kedalaman[0] = 0.
    kedalaman[1:-1] = dalam
    kedalaman[-1] = dalam[-2]+5000.
    return resist,kedalaman

# removing unused file
def remove_trash(basedir):
    import os
    import time

    # deleting old output files
    moddir = basedir+os.sep+'images'
    filelist = os.listdir(moddir)
    TWODAYS = 60.*60.*24.*2
    for i in filelist:
        if time.time()-os.stat(moddir+os.sep+i).st_mtime>TWODAYS:
            os.remove(moddir+os.sep+i)
    moddir = basedir+os.sep+'outputs'
    filelist = os.listdir(moddir)
    TWODAYS = 60.*60.*24.*2
    for i in filelist:
        if time.time()-os.stat(moddir+os.sep+i).st_mtime>TWODAYS:
            os.remove(moddir+os.sep+i)
