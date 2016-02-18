"""Classes for managing a simulation.
"""
import sys
import os.path
import time
import cPickle
import shutil
import warnings

class Controller(object):
    """Base class for managing a simulation.
    
    Your subclass must override the record() method.
    """
    
    def __init__(self, lat, outSteps, totalSteps, logFileName, cpFilename, cpInterval):
        """lat -- an LB lattice object (or any self contained object
        with a step(int numsteps) method and a property/attribute int
        time_step.
        
        outSteps -- timesteps between recording events

        totalSteps -- total no. timesteps
        
        logFileName -- file name for a log 
        
        cpFilename -- file name to which to save checkpoints

        cpInterval -- time in seconds between checkpoints
        """
        self.lat = lat
        self.outSteps = outSteps
        self.totalSteps = totalSteps
        self.logFile = file(logFileName, 'a')
        self.cpFilename = cpFilename
        self.cpInterval = cpInterval
        self.ready = True
        self.lastCPTime = time.time()
        
        return
    
    def log(self, msg):
        """Write and flush the message to the logfile, prepending the date
        & time"""
        self.logFile.write('%s: %s\n' % (time.ctime() , msg))
        self.logFile.flush()
        return

    def record(self):
        """You MUST override this"""
        raise NotImplementedError("Must override the record() method")
    
    def checkpoint(self):
        """Pickle ourself to the file specified in self.cpFilename"""
        import pdb
        pdb.set_trace()
        self.log("Checkpointing...")
        # Write the cp to a temporary file so we don't clobber an existing
        # cp file (yet)
        tmp = self.cpFilename + '.tmp'
        cpFile = file(tmp, 'wb')
        p = cPickle.Pickler(cpFile, protocol=2)
          
        p.dump(self)
        cpFile.close()
        
        # replace any existing cpFile with the new one
        if os.path.exists(self.cpFilename):
            os.remove(self.cpFilename)
        shutil.move(tmp, self.cpFilename)
        self.log("                ...done")
        self.lastCPTime = time.time()
        
        return
    
    def isRecordStep(self):
        """Return True if it's time to write results; False
        otherwise. The criteria is just for evenly spaced results.
        
        """
        return (self.lat.time_step % self.outSteps == 0)
    
    def isFinished(self):
        """Return True if the simulation has run for long enough;
        False otherwise. This just checks the number of steps against
        that specified on instantiation.
        
        """
        return self.lat.time_step >= self.totalSteps

    def isCheckpointStep(self):
        """Return True if we should perform a checkpoint; False
        otherwise. This just checks the elapsed time since the last
        CP.
        
        """

        return (time.time() - self.lastCPTime) > self.cpInterval
    
    def step(self):
        """Advance the simulation by one timestep.  Record data if at
        an output timestep.  Checkpoint if we've been running long
        enough.
        
        """
        self.lat.step()

        if self.isRecordStep():
            self.log('Time step %d of %d' %
                     (self.lat.time_step, self.totalSteps))
            self.record()
            pass
        
        if self.isCheckpointStep():
	    self.checkpoint()
            pass
        return
    
    def run(self):
        """Run the simulation with the control parameters specified.
        
        """
        self.lastCPTime = time.time()
        while not self.isFinished():
            self.step()
            continue
        return

    def __getstate__(self):
        """Returns the dict that is pickled"""
        d = self.__dict__.copy()
        d['logFile'] = self.logFile.name
        return d
    
    def __setstate__(self, d):
        """Restores self.__dict__ from the pickle'd dictionary"""
        if d['logFile'] == sys.stdout.name:
            d['logFile'] = sys.stdout
        elif d['logFile'] == sys.stderr.name:
            d['logFile'] = sys.stderr
        elif os.path.exists(d['logFile']):
            d['logFile'] = file(d['logFile'], 'a')
        else:
            d['ready'] = False
            warnings.warn('Cannot find logfile: "%s". Call fixpaths.' % \
                          d['logFile'])
            d.pop('logFile')
            d.pop('cpFilename')
            
            #raise cPickle.UnpicklingError('Cannot find logfile: "%s"' % d['logFile'])
        
        self.__dict__ = d
        try:
            self.log("Restored self from checkpoint file '%s'" % \
                     self.cpFilename)
        except AttributeError:
            # swallow a failure of the log.
            pass
        
        return

    @classmethod
    def restoreFromCheckpoint(cls, cpFilename, logFilename):
        """A class method for restoring a simulation from disk."""
        new = cPickle.load(file(cpFilename))
        if not getattr(new, 'ready', False):
            new.fixpaths(cpFilename, logFilename)
            
        return new
    
    def fixpaths(self, cpFilename, logFilename):
        """Fix up the paths; it's a private method so don't be calling
        this from outside the class.
        
        """
        if not getattr(self, 'ready', False):
            if not os.path.exists(logFilename):
                raise ValueError('Cannot find logfile: "%s"' % logFileName)
            if not os.path.exists(cpFilename):
                raise ValueError('Cannot find checkpoint file: "%s"' % \
                                 cpFilename)
            
            self.logFile = file(logfilename, 'a')
            self.cpFilename = cpFilename
            self.ready = True
            
            self.log('Completed restoration from checkpoint file "%s"' % \
                     cpFilename)

    
    pass

    
class BaseController(Controller):
    """As for Controller, except this keeps all its output in one directory.

    Still must override record().
    """
    
    def __init__(self, lat, baseName, outSteps, totalSteps, cpInterval):
        """lat -- an LB lattice object (or any self contained
        object with a step() method and a
        property/attribute int time_step.
        
        baseName -- the name of the directory in which to store
        things, if it doesn't exist, it is created.
        
        outSteps -- timesteps between recording events

        totalSteps -- total no. timesteps
        
        cpInterval -- time in seconds between checkpoints
        """
        self.baseName = baseName
        
        if os.path.exists(baseName):
            assert os.path.isdir(baseName)
        else:
            os.mkdir(baseName)
            pass
        
        Controller.__init__(self, lat, outSteps, totalSteps,
                            os.path.join(self.baseName, 'log'),
                            os.path.join(self.baseName, 'run.checkpoint'),
                            cpInterval)
        return

    def open(self, name, mode='r', buffering=1):
        """Open a file in the output directory. If name is a relative path, it is
        assumed to be relative to the base directory for this run. Other
        arguments options as for __builtins__.open.
        """
        if not os.path.isabs(name):
            name = os.path.join(self.baseName, name)
        return open(name, mode, buffering)

    def fixpaths(self, base):
        """Fix up the paths; it's a private method so don't be calling
        this from outside the class.
        

        """
        return Controller.fixpaths(self,
                                   os.path.join(base, 'log'),
                                   os.path.join(base, 'run.checkpoint'))
    
    @classmethod
    def restoreFromCheckpoint(cls, baseName):
        """A class method for restoring a simulation from disk."""
        new = cPickle.load(file(os.path.join(baseName, 'run.checkpoint')))
        new.fixpaths(baseName)
        return new
