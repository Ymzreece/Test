#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 02:16:39 2024

@author: mingzeyan
"""

deck = 46
# %%
deck -= 1
# %%
broo = 1
diaa = 1
heaa = 3
spaa = 4

bro_left = 13 - broo
dia_left = 13 - diaa
hea_left = 13 - heaa
spa_left = 13 - spaa
# %%
bro_buy = 30
dia_buy = 28
hea_buy = 33
spa_buy = 33

bro_sell = 24
dia_sell = 20
hea_sell = 28
spa_sell = 28


pro = 100
# %%
b_bro_ev = pro*bro_left/deck - bro_buy*(1-bro_left/deck)
b_dia_ev = pro*dia_left/deck - dia_buy*(1-dia_left/deck)
b_hea_ev = pro*hea_left/deck - hea_buy*(1-hea_left/deck)
b_spa_ev = pro*spa_left/deck - spa_buy*(1-spa_left/deck)

s_bro_ev = bro_sell*(1-bro_left/deck) - pro*bro_left/deck
s_dia_ev = dia_sell*(1-dia_left/deck) - pro*dia_left/deck 
s_hea_ev = hea_sell*(1-hea_left/deck) - pro*hea_left/deck 
s_spa_ev = spa_sell*(1-spa_left/deck) - pro*spa_left/deck

a_bro = [b_bro_ev,s_bro_ev]
a_dia = [b_dia_ev,s_dia_ev]
a_hea = [b_hea_ev,s_hea_ev]
a_spa = [b_spa_ev,s_spa_ev]
