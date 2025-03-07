# KISS_GP_on_GeoGals
Applying KISS-GP methods to analyse galaxy data


The main code is `KISSGP+NUTS.py`

To run it, specify as flags a galaxy ID (two available; NGC3351 and NGC1300), a metallicity diagnostic (three available: O3N2, N2S2Ha, and Scal), and a length for the monte carlo chains

For example:
`python3 KISSGP+NUTS.py -g 'NGC3351' -d 'O3N2' -l 10` 

is a good test to make sure everything is working.
