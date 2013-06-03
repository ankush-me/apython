import sys, math, time
import threading

vals  = {}
lock  = threading.RLock()

def runInThread(names, mins, maxes, res=None, appName='easyInput'):
    from PyQt4 import QtCore, QtGui

    class sliderInput:
        def __init__(self, names, mins, maxes, res, appName):
            self.widget = QtGui.QWidget()
            self.width  = 500
            self.height = 100
            self.widget.setWindowTitle(appName)
        
            self.vals    = {}
            
            M = max([len(n) for n in names])
        
            if res==None:
                res = []
                for(mi,Ma) in zip(mins, maxes):
                    res.append((Ma-mi)/10.0)
            for n,mi,ma,r in zip(names, mins, maxes, res):
                self.addSlider(n,mi,ma,r,M)

            # update global vals
            global vals
            lock.acquire()
            vals = self.vals
            lock.release()

            self.width = max(500, 10*M + 450)
            self.widget.setGeometry(100,100,self.width, self.height)
            self.widget.show()


        def addSlider(self, name, vmin, vmax, res, MAX_NAME):
            slider = QtGui.QSlider(QtCore.Qt.Horizontal, self.widget)
            slider.setFocusPolicy(QtCore.Qt.NoFocus)
            slider.setGeometry(10+10*MAX_NAME, self.height-50, self.width-100, 20)
            slider.setMinimum(0)
            m = int(math.ceil((vmax-vmin)/res))
            slider.setMaximum(m)

            nlabel = QtGui.QLabel(self.widget)
            nlabel.setText(name)
            nlabel.setGeometry(10, self.height-50, 10*MAX_NAME, 20)

            vlabel = QtGui.QLabel(self.widget)
            vlabel.setText(str(vmin))
            vlabel.setGeometry(15+10*MAX_NAME+self.width-100,self.height-50, 100, 20)

            def update(v):
                global vals
                val = vmin + v*res
                vlabel.setText(str(val))
                self.vals[name] = val

                # update global variable
                lock.acquire()
                vals = self.vals
                lock.release()

            slider.valueChanged[int].connect(update)
            self.vals[name]    = vmin
            self.height += 30

    app =  QtGui.QApplication([appName])
    ei  = sliderInput(names, mins, maxes, res, appName)
    app.exec_()


def easyInput(names, mins, maxes, res=None, appName='appName'):
    """
    main interface function to the user.
    specify UNIQUE names, mins, maxes, res[olutions] and appName
    """
    #thread_helper.queueCommand(runInThread, [names, mins, maxes, res, appName])
    qt_thread = threading.Thread(target=runInThread, args=[names, mins, maxes, res, appName], name="pyqt_thread")
    qt_thread.start()


def getVals():
    """
    return the vals set through sliders.
    """
    global vals
    lock.acquire()
    v = vals
    lock.release()
    return v



if __name__=='__main__':
    names = ['a','b']
    mins  = [0,10]
    maxes = [5,100]
    res   = [1, 10]
    easyInput(names, mins, maxes, res)
    
    while True:
        print getVals()
        time.sleep(1)
    
