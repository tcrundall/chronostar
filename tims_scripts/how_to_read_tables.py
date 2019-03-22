# coding: utf-8
from astropy.table import Table
mytable = Table.read('banyan_data.txt', format='ascii')
mytable.info
