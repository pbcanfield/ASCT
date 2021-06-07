NEURON {
  POINT_PROCESS IZH
  NONSPECIFIC_CURRENT vv
  RANGE a,b,c,d,e,f,I,vv,thresh, vr, vt, vpeak, aACH, cACH, dACH,alphaShutdown, bACH, ACHshutdown, aMin, aMax, g, vrACH, k, Cap
}
UNITS {
	(mV) = (millivolt)
    (nA) = (nanoamp)
	(pA) = (picoamp)
	(uS) = (microsiemens)
	(nS) = (nanosiemens)
}

INITIAL {
u = uinit
net_send(0,1)
}

PARAMETER {  
  k = 0.0011 (nA/mV2) :(1/mV*megaohm) 
: So pranit used pA aesthetically then divides k b d and u by 1000 to get them back to be in nA which is what neuron uses and understands for point processes.
: The model can still be used in nA but you have to overlook the low values
  a = 0.01 (1/ms)
  b = 0.0002 (uS)
  c = -65 (mV)
  d = .001 (nA)

 
  vpeak= 30 (mV)
  vv = 0 (mV)
  vr = - 70 (mV)
  vt = - 45 (mV)

  a_OLM = 0.002
  ACH = 1 
  dACH = 0 (nA)
  cACH = 0 (mV)
  vrACH = 0 (mV)
  Cap = 1000
  uinit = 0 (nA)
	
  alphaShutdown = 1
  
  aMin = 0.01
  aMax = 0.05
  bACH = 1.25
  
  aACH = 0.025
  g = 1.333
  :The equations to calculate g and aACH aACH = aMin * bACH g = 1 / (bACH * (1 - (aMin/aMax)))
  ACHshutdown = 0
}

STATE { u }

ASSIGNED {
  }

BREAKPOINT {
  SOLVE states METHOD derivimplicit
  vv = -(k*(v - (vr + vrACH * (-1+ ACH) ))*(v - vt) - u)
  
  a_OLM = 0.023*ACH^2 - 0.022*ACH + 0.002
}

DERIVATIVE states {
    u' = (a_OLM * ACHshutdown + a *alphaShutdown)*(b*(v - (vr + vrACH * (-1+ ACH) ))-u)
	:u' =  a *(b*(v - (vr + vrACH * (-1+ ACH) ))-u)
    :u' = ( alphaShutdown*a +  ACHshutdown *(g * -aACH) / ((-1+ACH) - g * bACH) )*(b*(v - (vr + vrACH * (-1+ ACH) ))-u)
}

NET_RECEIVE (w) {
  if (flag == 1) {
    WATCH (v>vpeak) 2
  } else if (flag == 2) {
    net_event(t)
    v = c + cACH * (-1+ACH) 
    u = u+d + dACH * (1-ACH)
  }
}


