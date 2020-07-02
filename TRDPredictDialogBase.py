# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version Oct 26 2018)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

###########################################################################
## Class TRDPredictDialogBase
###########################################################################

class TRDPredictDialogBase ( wx.Dialog ):

	def __init__( self, parent ):
		wx.Dialog.__init__ ( self, parent, id = wx.ID_ANY, title = u"TRD Predict", pos = wx.DefaultPosition, size = wx.Size( 655,278 ), style = wx.DEFAULT_DIALOG_STYLE )

		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

		gbSizer1 = wx.GridBagSizer( 0, 0 )
		gbSizer1.SetFlexibleDirection( wx.BOTH )
		gbSizer1.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		self.m_staticText1 = wx.StaticText( self, wx.ID_ANY, u"TRD Model", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText1.Wrap( -1 )

		gbSizer1.Add( self.m_staticText1, wx.GBPosition( 0, 0 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.fpModel = wx.FilePickerCtrl( self, wx.ID_ANY, wx.EmptyString, u"Select the TRD Model file", u"Pytorch model file(*.pth)|*.pth", wx.DefaultPosition, wx.DefaultSize, wx.FLP_DEFAULT_STYLE|wx.FLP_SMALL )
		gbSizer1.Add( self.fpModel, wx.GBPosition( 0, 1 ), wx.GBSpan( 1, 8 ), wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5 )

		self.m_staticText2 = wx.StaticText( self, wx.ID_ANY, u"Input Image", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText2.Wrap( -1 )

		gbSizer1.Add( self.m_staticText2, wx.GBPosition( 1, 0 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.fpInput = wx.FilePickerCtrl( self, wx.ID_ANY, wx.EmptyString, u"Select input image", u"Image file(*.jpg;*.jpeg;*.bmp;*.png)|*.jpg;*.jpeg;*.bmp;*.png|All files (*.*)|*.*", wx.DefaultPosition, wx.DefaultSize, wx.FLP_DEFAULT_STYLE|wx.FLP_SMALL )
		gbSizer1.Add( self.fpInput, wx.GBPosition( 1, 1 ), wx.GBSpan( 1, 8 ), wx.ALL|wx.EXPAND|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.m_staticText3 = wx.StaticText( self, wx.ID_ANY, u"Classes num", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText3.Wrap( -1 )

		gbSizer1.Add( self.m_staticText3, wx.GBPosition( 2, 0 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.txtClassesNum = wx.TextCtrl( self, wx.ID_ANY, u"1", wx.DefaultPosition, wx.DefaultSize, 0 )
		gbSizer1.Add( self.txtClassesNum, wx.GBPosition( 2, 1 ), wx.GBSpan( 1, 2 ), wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5 )

		self.m_staticText4 = wx.StaticText( self, wx.ID_ANY, u"Block Size", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText4.Wrap( -1 )

		gbSizer1.Add( self.m_staticText4, wx.GBPosition( 2, 3 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.txtBlockSize = wx.TextCtrl( self, wx.ID_ANY, u"608", wx.DefaultPosition, wx.DefaultSize, 0 )
		gbSizer1.Add( self.txtBlockSize, wx.GBPosition( 2, 4 ), wx.GBSpan( 1, 2 ), wx.ALL|wx.EXPAND|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.m_staticText5 = wx.StaticText( self, wx.ID_ANY, u"Block Overlap", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText5.Wrap( -1 )

		gbSizer1.Add( self.m_staticText5, wx.GBPosition( 2, 6 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.txtBlockOverlap = wx.TextCtrl( self, wx.ID_ANY, u"197", wx.DefaultPosition, wx.DefaultSize, 0 )
		gbSizer1.Add( self.txtBlockOverlap, wx.GBPosition( 2, 7 ), wx.GBSpan( 1, 2 ), wx.ALL|wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5 )

		self.m_staticText6 = wx.StaticText( self, wx.ID_ANY, u"Score Thresh", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText6.Wrap( -1 )

		gbSizer1.Add( self.m_staticText6, wx.GBPosition( 3, 0 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.txtScoreThresh = wx.TextCtrl( self, wx.ID_ANY, u"0.7", wx.DefaultPosition, wx.DefaultSize, 0 )
		gbSizer1.Add( self.txtScoreThresh, wx.GBPosition( 3, 1 ), wx.GBSpan( 1, 2 ), wx.ALL, 5 )

		self.m_staticText7 = wx.StaticText( self, wx.ID_ANY, u"IOU Thresh", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText7.Wrap( -1 )

		gbSizer1.Add( self.m_staticText7, wx.GBPosition( 3, 3 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.txtIouThresh = wx.TextCtrl( self, wx.ID_ANY, u"0.5", wx.DefaultPosition, wx.DefaultSize, 0 )
		gbSizer1.Add( self.txtIouThresh, wx.GBPosition( 3, 4 ), wx.GBSpan( 1, 2 ), wx.ALL, 5 )

		self.m_staticText8 = wx.StaticText( self, wx.ID_ANY, u"Center Dis Thresh", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText8.Wrap( -1 )

		gbSizer1.Add( self.m_staticText8, wx.GBPosition( 3, 6 ), wx.GBSpan( 1, 1 ), wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.txtCenterDisThresh = wx.TextCtrl( self, wx.ID_ANY, u"0.1", wx.DefaultPosition, wx.DefaultSize, 0 )
		gbSizer1.Add( self.txtCenterDisThresh, wx.GBPosition( 3, 7 ), wx.GBSpan( 1, 2 ), wx.ALL, 5 )

		bSizer1 = wx.BoxSizer( wx.HORIZONTAL )

		bSizer1.SetMinSize( wx.Size( -1,75 ) )

		bSizer1.Add( ( 0, 0), 1, wx.EXPAND|wx.ALIGN_CENTER_VERTICAL, 5 )

		self.btnPredict = wx.Button( self, wx.ID_ANY, u"Predict", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer1.Add( self.btnPredict, 0, wx.ALL|wx.ALIGN_CENTER_VERTICAL, 5 )


		gbSizer1.Add( bSizer1, wx.GBPosition( 4, 0 ), wx.GBSpan( 1, 9 ), wx.ALIGN_CENTER_VERTICAL|wx.EXPAND, 5 )


		self.SetSizer( gbSizer1 )
		self.Layout()

		self.Centre( wx.BOTH )

		# Connect Events
		self.btnPredict.Bind( wx.EVT_BUTTON, self.btnPredictOnButtonClick )

	def __del__( self ):
		pass


	# Virtual event handlers, overide them in your derived class
	def btnPredictOnButtonClick( self, event ):
		event.Skip()


