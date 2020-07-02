import wx
import TRDPredictDialogBase
import subprocess
import os

# Implementing TRDPredictDialogBase
class TRDPredictDialog( TRDPredictDialogBase.TRDPredictDialogBase ):
	def __init__( self, parent ):
		TRDPredictDialogBase.TRDPredictDialogBase.__init__( self, parent )
		self.predictProcess = None

	# Handlers for TRDPredictDialogBase events.
	def btnPredictOnButtonClick( self, event ):
		# TODO: Implement btnPredictOnButtonClick
		if self.predictProcess != None and self.predictProcess.poll() is None:
			dlg = wx.MessageDialog(self,"正在检测图像……")
			dlg.ShowModal()
			return

		modelPath = self.fpModel.GetPath()
		imgPath = self.fpInput.GetPath()
		blockSize = self.txtBlockSize.GetValue()
		blockOverlap = self.txtBlockOverlap.GetValue()
		numClasses = self.txtClassesNum.GetValue()
		scoreThresh = self.txtScoreThresh.GetValue()
		iouThresh = self.txtIouThresh.GetValue()
		ctThresh = self.txtCenterDisThresh.GetValue()
		# E:\\SourceCode\\Python\\TRD\\
		dirname,_ = os.path.split(os.path.abspath(__file__)) 
		predictPy = os.path.join(dirname,'predict.py')
		cmdStr = 'python "%s" -m "%s" -i "%s" -iz %s -o %s -c %s -st %s -it %s -ct %s '
		cmdStr = cmdStr%(predictPy,modelPath, imgPath,blockSize,blockOverlap,numClasses,scoreThresh,iouThresh,ctThresh)
		self.predictProcess = subprocess.Popen(cmdStr)

if __name__ == '__main__':
	app = wx.App()
	dlg = TRDPredictDialog(None)
	dlg.ShowModal()