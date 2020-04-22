import numpy as np
from datetime import datetime,timedelta
from sys import stdout
from astropy.io import fits
import sunpy.map, pfsspy, glob, h5py
from sunpy.coordinates import sun
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

DefaultPath = "./ExampleData"

def extract_br(m):
	br = m.data
	br = br - np.nanmean(br)
	# GONG maps have their LH edge at -180deg, so roll to get it at 0deg
	br = np.roll(br, np.int((m.meta['CRVAL1'] + 180)/np.abs(m.meta['cdelt1'])), axis=1)
	br = np.nan_to_num(br)
	return br*1e5 # Gauss to nT

def gong2pfsspy(dt,rss=2.5,nr=60,ret_magnetogram=False,data_dir=DefaultPath) :
	#Input : dt (datetime object), rss (source surface radius in Rs)
	YYYYMMDD = f'{dt.year}{dt.month:02d}{dt.day:02d}'
	gongpath=f'mrzqs{YYYYMMDD}/'
	filepath = sorted(glob.glob(f'{data_dir}/mrzqs{YYYYMMDD[2:]}*.fits'))[0]
	stdout.write(f"Read in {filepath}\r")
	# Sunpy 1.03 error : need to fix header of gong maps to include units
	am = fits.open(filepath)[0]
	am.header.append(('CUNIT1','degree')) 
	am.header.append(('CUNIT2','degree'))                                   
	gong_map = sunpy.map.Map(am.data,am.header)
	br_gong = extract_br(gong_map)
	if ret_magnetogram : return br_gong
	peri_input = pfsspy.Input(br_gong, nr, rss,dtime=dt)
	peri_output = pfsspy.pfss(peri_input)

	return peri_output

def adapt2pfsspy(dt,rss=2.5,nr=60,adapt_source="GONG",realization="mean",
				 ret_magnetogram=False,data_dir=DefaultPath):
	#Input : dt (datetime object), rss (source surface radius in Rs)    
	YYYYMMDD = f'{dt.year}{dt.month:02d}{dt.day:02d}'
	A = {"GONG": 3, "HMI" : 4}.get(adapt_source)
	filepath=sorted(glob.glob(f"{data_dir}/adapt40{A}*{YYYYMMDD}*.fts"))[0]
	stdout.write(f"Read in {filepath}\r")
	adapt_map = fits.open(filepath)
	if realization == "mean" : br_adapt_ = np.mean(adapt_map[0].data,axis=0)
	elif isinstance(realization,int) : br_adapt_ = adapt_map[0].data[realization,:,:]
	else : raise ValueError("realization should either be 'mean' or type int ") 
	 # Interpolate to Strumfric Grid (const lat spacing -> const cos(lat) spacing)
	lat_dr = np.linspace(-90,90,br_adapt_.shape[0])
	lon_dr = np.linspace(0,360,br_adapt_.shape[1])
	clat_dr = np.sin(np.radians(lat_dr))
	clat_dr_interp = np.linspace(clat_dr[0],clat_dr[-1],len(clat_dr))
	br_adapt = np.zeros(br_adapt_.shape)
	for ind,_ in enumerate(lon_dr) :
		interper = interp1d(clat_dr,br_adapt_[:,ind])
		br_adapt[:,ind]=interper(clat_dr_interp)
	
	br_adapt -= np.mean(br_adapt)
	br_adapt *= 1e5 # G -> nT
	if ret_magnetogram : return br_adapt
	peri_input = pfsspy.Input(br_adapt, nr, rss,dtime=dt)
	peri_output = pfsspy.pfss(peri_input)

	return peri_output

def derosa2pfsspy(dt,rss=2.5,nr=60,
				  ret_magnetogram=False, 
				  data_dir=DefaultPath) :
	#Input : dt (datetime object), rss (source surface radius in Rs)
	YYYYMMDD = f'{dt.year}{dt.month:02d}{dt.day:02d}'
	dr_times=['_000400.h5','_060432.h5','_120400.h5','_180328.h5']
	dr_time=dr_times[np.argmin(np.abs(dt.hour-np.array([0,6,12,18])))]
	filepath = f'{data_dir}/Bfield_{YYYYMMDD}{dr_time}'
	f = h5py.File(filepath, 'r')
	stdout.write(f"Read in {filepath}\r")
	fdata=f['ssw_pfss_extrapolation']
	br_dr = fdata['BR'][0,:,:,:]*1e5
	nr_dr = fdata['NR']
	nlat_dr = fdata['NLAT']
	nlon_dr = fdata['NLON']
	r_dr = fdata['RIX'][0]
	lat_dr = fdata['LAT'][0]
	c_dr = np.cos(np.radians(90 - lat_dr))
	lon_dr = fdata['LON'][0]
	date=fdata['MODEL_DATE']
	magnetogram = br_dr[0,:,:]
	
	 # Interpolate to Strumfric Grid (const lat spacing -> const cos(lat) spacing)
	clat_dr = np.sin(np.radians(lat_dr))
	clat_dr_interp = np.linspace(clat_dr[0],clat_dr[-1],len(clat_dr))
	br_dr_interp = np.zeros(magnetogram.shape)
	for ind,_ in enumerate(lon_dr) :
		interper = interp1d(clat_dr,magnetogram[:,ind])
		br_dr_interp[:,ind]=interper(clat_dr_interp)
	br_dr_interp -= np.mean(br_dr_interp) # Remove Mean offest for pfsspy FFT based PFSS extrap 
	if ret_magnetogram : return (magnetogram,br_dr_interp)
	peri_input = pfsspy.Input(br_dr_interp, nr, rss,dtime=dt)
	peri_output = pfsspy.pfss(peri_input)	
	return peri_output

def hmi2pfsspy(dt,rss=2.5,nr=60,ret_magnetogram=False,data_dir=DefaultPath) :
	#Input : dt (datetime object), rss (source surface radius in Rs)
	YYYYMMDD = f'{dt.year}{dt.month:02d}{dt.day:02d}'
	filepath = glob.glob(f'{data_dir}/hmi.mrdailysynframe_small_720s.{YYYYMMDD}*.fits')[0]
	stdout.write(f"Read in {filepath}\r")
	hmi_fits = fits.open(filepath)[0]
	hmi_fits.header['CUNIT2'] = 'degree'
	for card in ['HGLN_OBS','CRDER1','CRDER2','CSYSER1','CSYSER2'] :
		hmi_fits.header[card] = 0

	hmi_map = sunpy.map.Map(hmi_fits.data,hmi_fits.header)
	hmi_map.meta['CRVAL1'] = 120 + sun.L0(time=hmi_map.meta['T_OBS']).value
	br_hmi = extract_br(hmi_map)
	if ret_magnetogram : return br_hmi
	peri_input = pfsspy.Input(br_hmi, nr, rss,dtime=dt)
	peri_output = pfsspy.pfss(peri_input)
	return peri_output


def plot_output(po,source="adapt",adapt_source="GONG",no_cbar=False,figax=None) :

	if source == "adapt" : map_name = f"A{adapt_source}"
	else : map_name = source
	L0=sunpy.coordinates.sun.L0(time=po.dtime).value
	r_,s_,p_ = (po.grid.rg,po.grid.sg,po.grid.pg)
	rss = np.exp(r_)[-1]
	br_ph = po.bg[:,:,0,2].T
	br_ss = po.source_surface_br


	if figax is None : fig,axes=plt.subplots(ncols=2,figsize=(20,7))
	else : 
		fig,axes = figax
		assert isinstance(fig,plt.Figure), "figax[0] must be plt.figure instance" 
		assert isinstance(axes,(list,np.ndarray)), "figax[1] must be list of 2 axes objects "
		
	axes[0].set_title(f"Photospheric Br : {map_name} {str(po.dtime)}")
	c1=axes[0].pcolormesh(np.degrees(p_),s_,br_ph,
					  cmap='bwr',
					  vmin=-500e3,
					  vmax=500e3,
					  rasterized=True
				  )
	if not no_cbar : _=plt.colorbar(c1,orientation='horizontal',
									label="Br [nT]",ax=axes[0])
	axes[0].set_xlim([0,360])
	axes[0].set_ylim([-1,1])
	if L0 -60 < 0 or L0 + 60 > 360 :
		axes[0].axvspan(0,(L0+60) % 360, color="Grey", alpha=0.2)
		axes[0].axvspan((L0-60)%360,360, color="Grey", alpha=0.2)
	else : axes[0].axvspan((L0-60) % 360,(L0+60) % 360,color="Grey",alpha=0.2)

	axes[1].set_title(f"Source Surface Br :{map_name} {str(po.dtime)}" )
	c2=axes[1].pcolormesh(np.degrees(p_),s_,br_ss,
					  cmap='bwr',
					  vmin=-np.nanmax(br_ss),
					  vmax=np.nanmax(br_ss),
					 )
	_=po.plot_pil(ax=axes[1])

	axes[1].set_xlim([0,360])
	axes[1].set_ylim([-1,1])
	axes[1].text(10,0.75,f'Rss : {rss} $R_\odot$',fontsize=20)
	if L0 -60 < 0 or L0 + 60 > 360 :
		axes[1].axvspan(0,(L0+60) % 360, color="Grey", alpha=0.2)
		axes[1].axvspan((L0-60)%360,360, color="Grey", alpha=0.2)
	else : axes[1].axvspan((L0-60) % 360,(L0+60) % 360,color="Grey",alpha=0.2)
	if not no_cbar : _=plt.colorbar(c2,orientation='horizontal',
									label="Br [nT]",ax=axes[1])
	
	if figax is None : return (fig,axes)
