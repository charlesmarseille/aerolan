***COSQM AOD CORRELATION FROM AERONET DATA***

This code reads cosqm data and aod data and correlates one to the other 
with some corrections on cosqm data, such as cloud screening (local variance threshold)
and time of year offset (day of week and season).
Of course, AOD values are from the lunar photometers of aeronet network.
Thus, correlation is between a given aod data point and the associated cosqm closest time value.
In this way, moon is always present in the cosqm data. (general ZNSB is higher and the correlation
is good for moon conditions, not confirmed for moonless nigths) 

File structure:
- AOD .lev15 (level 1.5==cloud screening, explained on aeronet website) files for each location
- COSQM .txt files from each cosqm location (wget command to have all .txt files, explained in 
another text file)


! Some data from cosqm are corrupted from wget command (maybe network of Teide peak), so 
the LoadDataCorrupt() function removes the broken lines from each corrupt text file !
