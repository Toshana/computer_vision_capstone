# -*- coding: utf-8 -*-
"""
Created on Tue Nov 01 13:59:12 2016

@author: Toshana
"""

import kivy
#kivy.require('1.0.6') # replace with your current kivy version !

from kivy.app import App
from kivy.uix.label import Label


class MyApp(App):

    def build(self):
        return Label(text='Hello world')


if __name__ == '__main__':
    MyApp().run()