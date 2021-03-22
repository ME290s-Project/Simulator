try:
    import tkinter
except ImportError:
    import Tkinter as tkinter 
import time 



class AnimationWindow(tkinter.Tk):
    ''' Base class for tkinter animation windows
    Creates window and binds keyboard events to react
     '''
    def __init__(self, width, height, windowtitle = 'animation engine'):
        tkinter.Tk.__init__(self)
        self.wm_title(windowtitle)
        self.bind('<KeyPress>',lambda event: self.keypress(*self._keyevent(event)))
        self.bind('<KeyRelease>',lambda event: self.keyrelease(*self._keyevent(event)))
        self.canvas = tkinter.Canvas(self,width = width, height = height, 
                        background = 'black', borderwidth = 0)
        self.canvas.pack() 
        self.set_frame_rate(30)
        self.gfxupdate_starttime = time.perf_counter()
        self.graphics_update_dt = 0.0 
        self.continue_animation = True 
        self.setup() 
        self.after(10,self._frame_tick) 
    
    def set_frame_rate(self, frame_rate):
        self.frame_rate = frame_rate
        self.frame_time = 1/ frame_rate

    def _frame_tick(self):
        now = time.perf_counter() 
        dt = now - self.gfxupdate_starttime
        self.graphics_update_dt += dt 
        if self.graphics_update_dt > self.frame_time:
            self.graphics_update_dt -= self.frame_time 
            # if self.graphics_update_dt >= self.frame_time:
            #     print("Gfx update too slow to reach {:d} fps!".format(self.frame_rate))
            if self.continue_animation:
                self.draw() 
        self.gfxupdate_starttime = now 
        self.after(1000 // (self.frame_rate * 2),self._frame_tick)

    def _keyevent(self, event):
        c = event.char 
        if not c or ord(c) > 255:
            c = event.keysym
        return c, (event.x, event.y) 

    def stop(self):
        self.continue_animation = False

    def setup(self):
        pass

    def draw(self):
        pass

    def keypress(self, char, mouseposition):
        pass

    def keyrelease(self, char, mouseposition):
        pass
    