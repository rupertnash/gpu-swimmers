class System:
    def __init__(self, lat):
        self.lat = lat
        self.things = []
        
    def AddThing(self, th):
        self.things.append(th)

    def Step(self):
        self.lat.ZeroForce()
        for th in self.things:
            th.AddForces(self.lat)
        self.lat.Step()
        self.lat.CalcHydro()
        for th in self.things:
            th.Move(self.lat)
    
