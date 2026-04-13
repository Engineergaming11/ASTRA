There are two different catalogs neccessary for platesolving given the present optics that we have.

For the Celestron (Schmitt Casagrand) we will need the 5200 index. Particularly we need 5204, not all 48 need to be installed given your location.
For the instance of Tucson Arizona, we will be using 5204-08 through 5204-39. You can automate the install on the cli using
for i in(seq i 8 39) do; wget url-5204-${i}.fits; done 

For the Celestron refracting 
We may use the entirety of the catalog but we can use xxx
likewise you can use
for i in(seq i 8 39) do; wget https://astrometry.net/data/4200/index-4206-${i}.fits; done

