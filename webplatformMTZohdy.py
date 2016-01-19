import web
from web import form
import web
import MTZohdy as MT
import numpy as np
import os
import string
import random

render = web.template.render('/media/serverdisk/MTZohdy/templates/')

urls = ('/', 'index','/images/(.*)','images')
app = web.application(urls, globals())

myform = form.Form( 
    form.Textbox('Max Depth Iteration',
                 form.regexp('\d+','Must be a positive integer')),
    form.Textbox('Max Rho Iteration',
                 form.regexp('\d+','Must be a positive integer')),
    form.Textarea('Period',
                  size='50'),
    form.Textarea('Rhoa',
                  size='50'),
    form.Textarea('Phase',
                  size='50'),
    form.Radio('Convergence criteria',['both','resistivity','phase'],
               value='both'))

calculIDform = form.Form(
    form.Textbox('Calculation ID',
                 form.regexp('^[a-z]{18,}$','Must be 18 lowercase characters')))

class images:
    def GET(self,name):
        ext = name.split(".")[-1] # Gather extension

        cType = {
            "png":"images/png",
            "jpg":"images/jpeg",
            "gif":"images/gif",
            "ico":"images/x-icon"            }

        if name in os.listdir('images'):  # Security
            web.header("Online MT Inversion", cType[ext]) # Set the Header
            return open('images/%s'%name,"rb").read() # Notice 'rb' for reading images
        else:
            raise web.notfound()    

class index:
    def GET(self): 
        form = myform()
        form2 = calculIDform()
        # make sure you create a copy of the form by calling it (line above)
        # Otherwise changes will appear globally
        return render.formtest(form,form2,calculIDerror=False)

    def POST(self): 
        form = myform() 
        form2 = calculIDform()
        form3 = web.input()
        print'form testing'
        print form3
        #if not form.validates() and not form2.validates(): 
        #    return render.formtest(form,form2,calculIDerror=False)
        if form3['Calculation ID']=='' and form3['maxiterthick']=='':
            return render.formtest(form,form2,calculIDerror=False)
        else:
            # form.d.boe and form['boe'].value are equivalent ways of
            # extracting the validated arguments from the form.
            print 'calculID form :'+form3['Calculation ID']
            if form3['Calculation ID']==None or form3['Calculation ID']=='':
                # creating random ID
                calculID = ''.join(random.choice(string.ascii_lowercase) for i in range(18))
                """
                # obtain values from webform
                iterdepth = int(form3['Max Depth Iteration'].value)
                iterrho = int(form['Max Rho Iteration'].value)
                period_string = form.d.Period.split()
                freq = 1./np.array([np.float(i) for i in period_string])
                rhoa_string = form.d.Rhoa.split()
                rhoa = np.array([np.float(i) for i in rhoa_string])
                phase_string = form.d.Phase.split()
                phase = np.radians(np.array([np.float(i) for i in phase_string]))
                if form['Convergence criteria'].value == 'both':
                    convcrit=0
                elif form['Convergence criteria'].value == 'resistivity':
                    convcrit=1
                elif form['Convergence criteria'].value == 'phase':
                    convcrit=2
                """
                # obtain values from webform                                                                                                                                                                                                    
                iterdepth = int(form3['maxiterthick'])
                iterrho = int(form3['maxiterrho'])
                period_string = form3['period'].split()
                freq = 1./np.array([np.float(i) for i in period_string])
                rhoa_string = form3['rhoa'].split()
                rhoa = np.array([np.float(i) for i in rhoa_string])
                phase_string = form3['phase'].split()
                phase = np.radians(np.array([np.float(i) for i in phase_string]))
                if form3['convcrit'] == 'both':
                    convcrit=0
                elif form3['convcrit'] == 'resistivity':
                    convcrit=1
                elif form3['convcrit'] == 'phase':
                    convcrit=2

                f0,rh0,ph0,rhoi,tebal,plotly_success = MT.maju_cantik_mundur_cantik(calculID, \
                                                            freq,rhoa,phase,iterdepth,iterrho,convcrit)
                if plotly_success:
                    return render.imdisplay(str(calculID), \
                                        'images/out'+calculID+'.png', \
                                        'http://104.196.29.119:8000/data_'+calculID+'.txt', \
                                        'http://104.196.29.119:8000/iter_'+calculID+'.txt', \
                                        'http://104.196.29.119:8000/out_'+calculID+'.txt',plotlyplot=True)
                    #return web.seeother('https://plot.ly/~irnaka/107.embed')
                else:
                    return render.imdisplay(str(calculID), \
                                        'images/out'+calculID+'.png', \
                                        'http://104.196.29.119:8000/data_'+calculID+'.txt', \
                                        'http://104.196.29.119:8000/iter_'+calculID+'.txt', \
                                        'http://104.196.29.119:8000/out_'+calculID+'.txt',plotlyplot=False)
            else:
                # check ID existence
                import os
                
                moddir = 'outputs/'
                filelist = os.listdir(moddir)
                try:
                    calculID = form3['Calculation ID']
                    print 'active calculation ID : '+calculID
                    tmp = filelist.index('out_'+calculID+'.txt')
                    return render.imdisplay(str(calculID), \
                                'images/out'+calculID+'.png', \
                                'http://104.196.29.119:8000/data_'+calculID+'.txt', \
                                'http://104.196.29.119:8000/iter_'+calculID+'.txt', \
                                'http://104.196.29.119:8000/out_'+calculID+'.txt',plotlyplot=False)
                except ValueError:
                    return render.formtest(form,form2,calculIDerror=True)
        
if __name__=="__main__":
    web.internalerror = web.debugerror
    app.run()
    
